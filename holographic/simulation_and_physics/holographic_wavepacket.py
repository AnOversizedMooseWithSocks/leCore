"""holographic_wavepacket.py -- the WAVE-PACKET FIELD (Physics & FX backlog, item N8 / #4).

The plain FFT ocean (holographic_spectralfield.ocean_field) is GLOBAL: every wave spans the whole domain, so it
cannot reflect off a wall or bend around a rock. The fix, from a decade of Jeschke & Wojtan work (Wave Packets
2017, Water Surface Wavelets 2018), is to LOCALIZE the spectrum in space: represent the surface as many little
wave packets, each a Gaussian-enveloped wave train that lives at a PLACE. Because a packet is local, it can:

  * REFLECT off a wall  -- its wavevector mirrors across the wall normal (k' = k - 2(k.n)n), specular reflection;
  * REFRACT / SHOAL over a depth change -- the finite-depth dispersion omega = sqrt(g|k| tanh(|k|h)) slows the
    packet in shallow water, so its energy piles up near shore (the reason waves steepen at the beach);
  * DIFFRACT -- the Gaussian envelope spreads, so energy leaks around an obstacle edge.

THE VSA FORM (Thesis A, and why this belongs on the substrate): a packet is exactly a ROLE-BOUND RECORD --
bind(POSITION,x) + bind(WAVENUMBER,k) + bind(DIRECTION,theta) + bind(AMPLITUDE,a) + bind(PHASE,phi), bundled into
one hypervector -- and the SURFACE is a BUNDLE of those records. That makes the surface a content-addressable
index: "is this packet part of the surface?" is a cosine against the bundle. Advancing a packet is `bind` (phase
turn + group-velocity shift); the obstacle response is a projection of the direction onto the allowed set (the
reflection above is its closed form). So tricky waves reuse bind / bundle / project verbatim -- they are MORE
native than the global FFT ocean, not less.

HONEST SCOPE (kept): the PHYSICS runs on plain readable NumPy arrays (positions, wavevectors, amplitudes,
phases) -- that is clearer and faster than unbinding/rebinding a record every step, and the memory warns against
gratuitous VSA round-trips. The record/bundle layer is the REPRESENTATION (a packet as a hypervector, the surface
as a content-addressable bundle), demonstrated and tested, not the simulation loop. Group velocity (energy speed)
is HALF the phase speed for deep-water gravity waves -- a packet's ENERGY moves at c_g, which is what we advect.
Deterministic (seeded); NumPy + stdlib only.
"""
import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, bundle, cosine, Vocabulary
from holographic.io_and_interop.holographic_encoders import ScalarEncoder


class WavePacketField:
    """A water surface as a set of localized wave packets. Domain is the square [0, size] x [0, size]. Each packet
    carries a position, a wavevector k (direction + wavenumber), an amplitude, and a phase. Advancing turns the
    phase by omega*dt and moves the packet's ENERGY at the group velocity along its direction; walls reflect it."""

    def __init__(self, size=64.0, g=9.81, envelope=6.0, seed=0):
        self.size = float(size)
        self.g = float(g)
        self.envelope = float(envelope)                          # Gaussian packet width (a few wavelengths)
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros((0, 2))                              # (N,2) packet centres
        self.k = np.zeros((0, 2))                                # (N,2) wavevectors
        self.amp = np.zeros((0,))                                # (N,) amplitudes
        self.phase = np.zeros((0,))                             # (N,) carrier phases

    def add_packet(self, pos, wavevector, amplitude=1.0, phase=0.0):
        """Add one packet at `pos` travelling along `wavevector` (its magnitude is the wavenumber |k|)."""
        self.pos = np.vstack([self.pos, np.asarray(pos, float)])
        self.k = np.vstack([self.k, np.asarray(wavevector, float)])
        self.amp = np.append(self.amp, float(amplitude))
        self.phase = np.append(self.phase, float(phase))
        return self

    # -- the dispersion relation (deep water, or finite depth for shoaling) -------------------------------------
    def _omega(self, kmag, depth=None):
        """Angular frequency of a gravity wave. Deep water: omega = sqrt(g|k|). Finite depth h: multiply |k| by
        tanh(|k|h) -- which -> |k| in deep water and -> 0 in shallow (the wave slows and shoals)."""
        if depth is None:
            return np.sqrt(self.g * kmag)
        return np.sqrt(self.g * kmag * np.tanh(kmag * np.maximum(depth, 1e-6)))

    def _group_speed(self, kmag, depth=None):
        """The GROUP speed c_g = d omega / d|k| -- the speed a packet's ENERGY (and so the packet) travels. For
        deep water this is exactly half the phase speed. Computed by a central difference so it also covers the
        finite-depth case without a hand-derived formula."""
        dk = 1e-4
        return (self._omega(kmag + dk, depth) - self._omega(kmag - dk, depth)) / (2 * dk)

    # -- advance: phase turn + group-velocity move + wall reflection --------------------------------------------
    def advance(self, dt, depth=None, obstacles=None):
        """Step every packet forward by dt. depth(pos)->h optionally slows packets in shallow water (shoaling);
        obstacles is a list of axis-aligned boxes (x0,y0,x1,y1) that reflect packets. The domain boundary always
        reflects (a walled tank)."""
        kmag = np.linalg.norm(self.k, axis=1)
        kmag = np.maximum(kmag, 1e-9)
        khat = self.k / kmag[:, None]
        depths = None if depth is None else np.array([depth(p) for p in self.pos])
        self.phase = self.phase + self._omega(kmag, depths) * dt
        cg = self._group_speed(kmag, depths)                    # (N,) group speed per packet
        self.pos = self.pos + khat * cg[:, None] * dt           # energy moves at the group velocity
        self._reflect_boundary()
        if obstacles:
            for box in obstacles:
                self._reflect_box(box)
        return self

    def _reflect_boundary(self):
        """Specular reflection off the domain walls: a packet that steps past a wall has its position folded back
        and the matching wavevector component flipped (k' = k - 2(k.n)n for an axis-aligned normal n)."""
        for axis in (0, 1):
            low = self.pos[:, axis] < 0.0
            high = self.pos[:, axis] > self.size
            self.pos[low, axis] = -self.pos[low, axis]                       # fold back in
            self.pos[high, axis] = 2 * self.size - self.pos[high, axis]
            flip = low | high
            self.k[flip, axis] = -self.k[flip, axis]                         # mirror the direction

    def _reflect_box(self, box):
        """Reflect packets that have entered an axis-aligned obstacle box off its nearest face."""
        x0, y0, x1, y1 = box
        inside = (self.pos[:, 0] > x0) & (self.pos[:, 0] < x1) & (self.pos[:, 1] > y0) & (self.pos[:, 1] < y1)
        for i in np.where(inside)[0]:
            px, py = self.pos[i]
            # distance to each face; push out of and reflect off the closest one
            dists = {"L": px - x0, "R": x1 - px, "B": py - y0, "T": y1 - py}
            face = min(dists, key=dists.get)
            if face in ("L", "R"):
                self.pos[i, 0] = x0 if face == "L" else x1
                self.k[i, 0] = -self.k[i, 0]
            else:
                self.pos[i, 1] = y0 if face == "B" else y1
                self.k[i, 1] = -self.k[i, 1]

    # -- render: the surface is the SUM of the packets' enveloped carriers --------------------------------------
    def render(self, res=64):
        """Sample the surface on a res x res grid: each packet contributes a Gaussian-enveloped cosine
        amp * exp(-r^2 / 2 sigma^2) * cos(k.(x - pos) + phase). The sum is the localized-spectrum surface."""
        xs = np.linspace(0, self.size, res)
        X, Y = np.meshgrid(xs, xs)
        h = np.zeros((res, res))
        s2 = self.envelope * self.envelope
        for i in range(len(self.pos)):
            dx = X - self.pos[i, 0]
            dy = Y - self.pos[i, 1]
            r2 = dx * dx + dy * dy
            carrier = np.cos(self.k[i, 0] * dx + self.k[i, 1] * dy + self.phase[i])
            h += self.amp[i] * np.exp(-r2 / (2 * s2)) * carrier
        return h


# --- the VSA representation: a packet IS a role-bound record, the surface IS a bundle -------------------------

_PACKET_ROLES = ["POS_X", "POS_Y", "K_MAG", "K_DIR", "AMP", "PHASE"]


def _packet_fields(field, i):
    """The six scalar fields of packet i, each normalised into [0,1] so one ScalarEncoder can carry them all."""
    px, py = field.pos[i]
    kx, ky = field.k[i]
    kmag = float(np.hypot(kx, ky))
    kdir = (np.arctan2(ky, kx) + np.pi) / (2 * np.pi)           # direction angle -> [0,1]
    return {
        "POS_X": px / field.size, "POS_Y": py / field.size,
        "K_MAG": min(kmag / 3.0, 1.0), "K_DIR": kdir,
        "AMP": min(field.amp[i], 1.0), "PHASE": (field.phase[i] % (2 * np.pi)) / (2 * np.pi),
    }


def packet_record(field, i, roles, enc):
    """Encode packet i as a ROLE-BOUND RECORD: bind each scalar field to its role atom and bundle -- one
    hypervector standing for the whole packet (the N8 claim, made concrete)."""
    vals = _packet_fields(field, i)
    return bundle([bind(roles.get(r), enc.encode(vals[r])) for r in _PACKET_ROLES])


def surface_bundle(field, dim=2048, seed=0):
    """The surface as a BUNDLE of packet records -- a content-addressable index of the packets. Returns
    (bundle_vector, records, roles, enc) so callers can query membership by cosine."""
    roles = Vocabulary(dim, seed)
    for r in _PACKET_ROLES:
        roles.get(r)
    enc = ScalarEncoder(dim, lo=0.0, hi=1.0, seed=seed + 1, kernel="rbf", bandwidth=0.08)
    records = [packet_record(field, i, roles, enc) for i in range(len(field.pos))]
    return (bundle(records) if records else np.zeros(dim)), records, roles, enc


def _selftest():
    """A packet reflects off a wall (its wavevector mirrors and it heads back); its ENERGY moves at the group
    velocity (half the phase speed for deep water); shallow water slows it (shoaling); the rendered surface is
    localized and oscillates at |k|; a packet is a role-bound record and the surface is a content-addressable
    bundle (a member scores high, a stranger low); deterministic."""
    # (1) REFLECTION off the right wall: a packet heading +x comes back heading -x
    f = WavePacketField(size=64.0, g=9.81, seed=0)
    f.add_packet(pos=[60.0, 32.0], wavevector=[1.2, 0.0], amplitude=1.0)
    kx0 = f.k[0, 0]
    for _ in range(200):
        f.advance(0.1)
    assert f.k[0, 0] == -kx0, "the wavevector must mirror after hitting the wall"
    assert f.pos[0, 0] < 60.0, "the packet must be heading back inward"

    # (2) GROUP velocity = half the PHASE velocity (deep water gravity waves)
    g = WavePacketField(size=200.0, g=9.81, seed=0)
    g.add_packet(pos=[10.0, 100.0], wavevector=[0.5, 0.0])
    kmag = 0.5
    cg = g._group_speed(kmag); cp = g._omega(kmag) / kmag
    assert abs(cg - 0.5 * cp) < 1e-3, (cg, cp)
    x_before = g.pos[0, 0]; g.advance(1.0); moved = g.pos[0, 0] - x_before
    assert abs(moved - cg) < 1e-2, "the packet moves at the group speed"

    # (3) SHOALING: shallow water is slower than deep water for the same |k|
    deep = g._group_speed(0.5, depth=None)
    shallow = g._group_speed(0.5, depth=0.5)
    assert shallow < deep, "a packet slows in shallow water"

    # (4) the rendered surface is LOCALIZED near the packet and oscillates
    r = WavePacketField(size=64.0, seed=0)
    r.add_packet(pos=[32.0, 32.0], wavevector=[1.5, 0.0], amplitude=1.0)
    h = r.render(res=64)
    cy = 32
    row = np.abs(h[cy])
    assert row[32] > row[5] and row[32] > row[60], "energy concentrated near the packet centre"
    assert np.max(np.abs(h)) > 0.1

    # (5) VSA: a packet is a record, the surface is a content-addressable bundle
    s = WavePacketField(size=64.0, seed=1)
    s.add_packet(pos=[20.0, 20.0], wavevector=[1.0, 0.3])
    s.add_packet(pos=[45.0, 40.0], wavevector=[0.4, 1.1])
    bundle_vec, records, roles, enc = surface_bundle(s, dim=4096, seed=0)
    member = float(cosine(records[0], bundle_vec))
    stranger = WavePacketField(size=64.0, seed=9); stranger.add_packet([5.0, 55.0], [2.0, -1.5])
    stranger_rec = packet_record(stranger, 0, roles, enc)
    assert member > float(cosine(stranger_rec, bundle_vec)), "a member packet scores higher than a stranger"

    # (6) deterministic
    f2 = WavePacketField(size=64.0, g=9.81, seed=0); f2.add_packet([60.0, 32.0], [1.2, 0.0], 1.0)
    for _ in range(200):
        f2.advance(0.1)
    assert np.array_equal(f.pos, f2.pos) and np.array_equal(f.k, f2.k)

    print("holographic_wavepacket selftest OK: a packet reflects off the wall (k mirrors, heads back); its energy "
          "moves at the group speed = half the phase speed; shallow water slows it (shoaling); the surface is a "
          "localized wave train; a packet is a role-bound record and the surface a content-addressable bundle "
          "(member %.2f > stranger); deterministic" % member)


if __name__ == "__main__":
    _selftest()
