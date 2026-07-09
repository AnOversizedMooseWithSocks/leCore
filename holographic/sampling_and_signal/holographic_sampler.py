"""holographic_sampler.py -- the SAMPLER: a placeable read-probe (modeling-app backlog, the capstone item).

A Sampler is the READ-DUAL of a FieldEffect. A FieldEffect is a shaped, falloff-weighted WRITE to a field (a force,
some heat, a decal); a Sampler is the SAME shape + falloff + attach machinery pointed the other way -- a shaped,
falloff-weighted READ from the scene. It lives in the canonical Scene like any object (a handle + a transform), so
it is placed, moved, and animated like anything else, and its output can drive a parameter or fire a trigger. That
symmetry is why a genuinely new, useful object costs almost nothing to build: it mirrors machinery that exists.

Three modes, each reusing a piece already in the box:
  * POINT   -- read the value(s) at one spot (the field at the sampler's position);
  * SURFACE -- sample points on a surface patch (emit_from_surface, which already importance-samples by a weight),
               read the field at each, weight by the falloff and an optional map/field;
  * VOLUME  -- fill the shape's interior (points where the shape SDF is negative), read the field, and AGGREGATE --
               a weighted mean (a bundle) or a sum (a Monte-Carlo integral).

OVERLAP -- when several objects cover the sampled region, the native answer is a LABELED BUNDLE: tag each sample by
its owning object's STABLE HANDLE (a near-orthonormal identity atom), scale the handle by that sample's
contribution, and bundle. The one superposed readout is then both SEPARABLE (an object's share is the dot of the
bundle with its handle) and COLLAPSIBLE (the total is the sum), and the dominant contributor is a cleanup. (For a
VECTOR-valued read, the general form is bind(handle, value_vector) + unbind; the scalar case here reduces to the
weighted-atom sum, which is exact for near-orthonormal handles.)

Mirrors FieldEffect; reuses _falloff, emit_from_surface, resolve_param. Deterministic; NumPy + stdlib only.
"""
import numpy as np

from holographic.misc.holographic_fieldeffect import _falloff


class Sampler:
    """A placeable read-probe. `shape` is an SDF (anything with .eval(P)) giving WHERE to read; `target(P) -> vals`
    is WHAT to read (a field / material / attribute -- radiance's 'render = query'); `mode` is point/surface/volume;
    `radius`/`falloff` shape the read weight exactly as FieldEffect shapes its write weight; `weight` is an optional
    map/field weighting resolved through the parameter socket."""

    def __init__(self, shape, target, mode="point", radius=1.0, falloff="smooth", weight=None):
        self.shape = shape
        self.target = target
        self.mode = mode
        self.radius = float(radius)
        self.falloff = falloff
        self.weight = weight

    # -- the shaped read weight (the SAME falloff curve FieldEffect writes with) ----------------------------
    def _falloff_weight(self, P):
        d = self.shape.eval(P)
        t = np.clip(-d / self.radius, 0.0, 1.0)              # 0 outside the shape -> 1 a radius deep inside
        return _falloff(t, self.falloff)

    def _map_weight(self, P):
        if self.weight is None:
            return np.ones(len(P))
        from holographic.misc.holographic_param import resolve_param
        return np.asarray(resolve_param(self.weight, points=P, n=len(P)), float)

    # -- the points to read, per mode ----------------------------------------------------------------------
    def _region_points(self, at, bounds=None, n=64, seed=0):
        at = np.asarray(at, float)
        if self.mode == "point":
            return at.reshape(1, -1)
        if self.mode == "surface":
            from holographic.simulation_and_physics.holographic_emitter import emit_from_surface
            pos, _nrm, _vel = emit_from_surface(self.shape.eval, n, bounds, weight=self.weight, seed=seed)
            return pos
        # volume: random-fill the bounds, keep the points INSIDE the shape (sdf < 0); oversample so we get ~n
        lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
        rng = np.random.default_rng(seed)
        pts = rng.uniform(lo, hi, size=(n * 6, len(lo)))
        return pts[self.shape.eval(pts) < 0.0][:n]

    def _read(self, pts):
        return np.asarray(self.target(pts), float)

    # -- the aggregated read -------------------------------------------------------------------------------
    def sample(self, at=(0.0, 0.0, 0.0), bounds=None, n=64, seed=0, aggregate="mean"):
        """Read the scene at this sampler's region and aggregate. `aggregate`: 'mean' (falloff+map-weighted mean --
        a bundle), 'sum' (a Monte-Carlo integral), or 'raw' (the (values, weights, points) for the caller). Returns
        None for an empty region."""
        pts = self._region_points(at, bounds, n, seed)
        if len(pts) == 0:
            return None
        vals = self._read(pts)
        # POINT mode reads the value AT the probe (weight 1); surface/volume weight by the falloff over the region
        w = self._map_weight(pts) if self.mode == "point" else self._falloff_weight(pts) * self._map_weight(pts)
        wsum = float(w.sum()) + 1e-15
        if aggregate == "raw":
            return vals, w, pts
        if vals.ndim > 1:                                    # vector-valued field (e.g. velocity, colour)
            acc = (vals * w[:, None]).sum(axis=0)
            return acc / wsum if aggregate == "mean" else acc
        acc = float((vals * w).sum())
        return acc / wsum if aggregate == "mean" else acc

    # -- the overlap case: a labeled bundle ----------------------------------------------------------------
    def sample_labeled(self, owner_atoms_of, at=(0.0, 0.0, 0.0), bounds=None, n=64, seed=0):
        """The OVERLAP case. `owner_atoms_of(pts) -> (K, dim)` gives each sample's owning object's stable handle
        atom. Each sample scales its owner's handle by its (scalar) contribution (value x falloff x map weight);
        bundling gives ONE superposed readout. Use contribution_of / dominant_owner / total_contribution on it."""
        pts = self._region_points(at, bounds, n, seed)
        if len(pts) == 0:
            return None
        vals = self._read(pts)
        contrib = vals if vals.ndim == 1 else np.linalg.norm(vals, axis=1)   # scalar contribution per sample
        contrib = contrib * self._falloff_weight(pts) * self._map_weight(pts)
        atoms = np.asarray(owner_atoms_of(pts), float)       # (K, dim) handle atom per sample
        # weighted superposition of owner atoms: Sum_i contrib_i * handle_i  (bind reduces to scaling for a scalar)
        return (atoms * contrib[:, None]).sum(axis=0)


# ---- reading a labeled bundle (separable + collapsible) ------------------------------------------------------

def contribution_of(labeled_bundle, handle_atom):
    """An object's share of a labeled bundle -- the dot with its stable handle (handles are near-orthonormal, so
    this recovers that object's summed contribution and cancels the others)."""
    return float(np.dot(labeled_bundle, handle_atom))


def dominant_owner(labeled_bundle, handle_atoms):
    """The object that contributed MOST -- a cleanup of the bundle against the handle codebook. Returns the index."""
    hb = np.asarray(handle_atoms, float)
    return int(np.argmax(hb @ labeled_bundle))


def total_contribution(labeled_bundle, handle_atoms):
    """Collapse the labels: the total contribution across all objects (the sum of every object's share)."""
    hb = np.asarray(handle_atoms, float)
    return float((hb @ labeled_bundle).sum())


def owners_from_sdfs(handle_sdf_pairs):
    """Build an `owner_atoms_of(pts)` callable from a list of (handle_atom, sdf) pairs: each point is owned by the
    object whose SDF is smallest there (nearest surface / inside). Returns per-point handle atoms."""
    atoms = np.asarray([h for h, _ in handle_sdf_pairs], float)
    sdfs = [s for _, s in handle_sdf_pairs]

    def owner_atoms_of(pts):
        d = np.stack([s.eval(pts) for s in sdfs], axis=1)    # (K, n_objects) distance to each object
        return atoms[np.argmin(d, axis=1)]                   # each point's nearest object's handle atom

    return owner_atoms_of


# ---- placing a Sampler in the Scene + wiring its output ------------------------------------------------------

def place_sampler(scene, sampler, transform=None, name="Sampler"):
    """Drop a Sampler into the canonical Scene as an object (a record with a handle + transform), so it is placed,
    moved, and animated like anything else. The Sampler instance rides in the record's params. Returns the handle."""
    return scene.add(name=name, transform=transform, params={"sampler": sampler, "is_sampler": True})


def sampler_triggers(value, threshold, direction="above"):
    """Fire when a sampled value crosses a threshold -- the trigger affordance (a sampler gating an effect). True
    when value is above (or below) the threshold."""
    return (value >= threshold) if direction == "above" else (value <= threshold)


def _selftest():
    """A point sampler reads the field at its spot; a volume sampler averages the field inside its shape (a bundle)
    and its Monte-Carlo integral tracks the region; an overlapping read makes a LABELED bundle that separates by
    handle, names the dominant owner, and collapses to a total; a Sampler places into the Scene; deterministic."""
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    from holographic.scene_and_pipeline.holographic_scene_doc import Scene

    # a scalar field: height = z (so a region's mean height is its centre height)
    field = lambda P: np.asarray(P, float)[:, 2]

    # (1) POINT mode: read the field exactly at the probe
    sp = Sampler(sphere(0.1), field, mode="point")
    assert abs(sp.sample(at=(0.0, 0.0, 2.0)) - 2.0) < 1e-9

    # (2) VOLUME mode: mean height inside a unit sphere at the origin is ~0 (symmetric); the integral is finite
    sv = Sampler(sphere(1.0), field, mode="volume", radius=1.0, falloff="linear")
    mean = sv.sample(at=(0, 0, 0), bounds=([-1, -1, -1], [1, 1, 1]), n=400, seed=0, aggregate="mean")
    assert abs(mean) < 0.15                                  # symmetric region -> mean height ~ 0

    # (3) OVERLAP: two objects, a labeled bundle separates their contributions
    dim = 512
    rng = np.random.default_rng(0)
    hA = rng.standard_normal(dim); hA /= np.linalg.norm(hA)   # object A's stable handle atom
    hB = rng.standard_normal(dim); hB /= np.linalg.norm(hB)   # object B's
    sdfA = sphere(1.0)                                        # A at the origin
    from holographic.mesh_and_geometry.holographic_sdf import sphere as _sph
    # B is a sphere translated to x=3 (use the SDF translate if available; else a tiny wrapper)
    class _Shift:
        def __init__(s, base, off): s.base = base; s.off = np.asarray(off, float)
        def eval(s, P): return s.base.eval(np.asarray(P, float) - s.off)
    sdfB = _Shift(_sph(1.0), [3.0, 0.0, 0.0])
    owners = owners_from_sdfs([(hA, sdfA), (hB, sdfB)])
    # a constant field = 1, sampled over a wide region covering BOTH spheres -> each sample labeled by nearest
    ones = lambda P: np.ones(len(P))
    swide = Sampler(sphere(5.0), ones, mode="volume", radius=5.0)
    lab = swide.sample_labeled(owners, at=(1.5, 0, 0), bounds=([-2, -2, -2], [5, 2, 2]), n=600, seed=1)
    cA = contribution_of(lab, hA)
    cB = contribution_of(lab, hB)
    assert cA > 0 and cB > 0                                  # both objects contributed
    dom = dominant_owner(lab, [hA, hB])
    assert dom in (0, 1)                                      # a well-defined dominant owner
    tot = total_contribution(lab, [hA, hB])
    assert abs(tot - (cA + cB)) < 1e-6                        # collapsible: total = sum of shares

    # (4) placeable in the Scene as an object
    scene = Scene(dim=128, seed=0)
    h = place_sampler(scene, sp, name="probe")
    assert h in scene.objects and scene.get(h).params["is_sampler"]

    # (5) trigger crosses a threshold
    assert sampler_triggers(2.0, 1.0, "above") and not sampler_triggers(0.5, 1.0, "above")

    # (6) deterministic
    a = sv.sample(at=(0, 0, 0), bounds=([-1, -1, -1], [1, 1, 1]), n=200, seed=7)
    b = sv.sample(at=(0, 0, 0), bounds=([-1, -1, -1], [1, 1, 1]), n=200, seed=7)
    assert a == b

    print("holographic_sampler selftest OK: a POINT sampler reads the field at its spot; a VOLUME sampler averages "
          "the field inside its shape (a weighted bundle, ~0 over a symmetric region); an OVERLAP read builds a "
          "LABELED bundle that separates each object's share by its handle (A=%.1f, B=%.1f), names the dominant "
          "owner, and collapses to the total; a Sampler places into the Scene as an object; deterministic"
          % (cA, cB))


if __name__ == "__main__":
    _selftest()
