"""holographic_roomacoustic.py -- A6: GEOMETRIC ROOM ACOUSTICS. How a room echoes -- reflections and reverb.

WHY THIS EXISTS (Acoustics & Cymatics backlog, item A6 -- the last one)
----------------------------------------------------------------------
Clap in a bathroom and in a carpeted room: same clap, very different sound. The room's geometry sets WHEN the
echoes arrive (each reflection travels a longer path, so it is delayed by distance/c), and the wall materials
set HOW LOUD they come back (a hard tiled wall reflects almost everything, soft foam soaks it up) and thus how
long the reverberation tail lasts. This module computes both, reusing the acoustic reflectance/absorption from
A2 -- the acoustic twin of the light path tracer, swapping the optical BRDF for the acoustic reflect/absorb.

THE METHOD (readable, the two standard tools)
---------------------------------------------
  * EARLY REFLECTIONS -- the IMAGE-SOURCE METHOD (Allen & Berkley 1979). A reflection off a flat wall looks, to
    the listener, exactly like sound from a MIRROR-IMAGE of the source placed behind that wall. So mirror the
    source across each wall (and across pairs of walls for second bounces), and each image contributes one tap of
    the room impulse response at delay = image_distance / c and amplitude = (product of the wall reflection
    coefficients it bounced off) / distance (spherical spreading). The direct sound is the un-mirrored source.
  * REVERBERATION TIME -- SABINE'S formula RT60 = 0.161 * V / A, where V is the room volume and A = sum over
    surfaces of (absorption_coefficient * area). More (or more absorptive) surface area -> larger A -> shorter
    RT60. This is the tail: how many seconds for the sound to decay by 60 dB.

HONEST SCOPE (kept negative): geometric (ray) acoustics is a HIGH-FREQUENCY approximation -- it has no
diffraction or interference (the low-frequency wave field A3 is the complement, to be used in that regime).
Early reflections are computed to a low image-source order here (readable, exact for a shoebox); the full late
tail is summarised by Sabine rather than every high-order image. A shoebox (rectangular) room. Deterministic;
NumPy + stdlib. Wall reflectance/absorption reuses holographic_acoustic (A2).
"""
import numpy as np


class ShoeboxRoom:
    """A rectangular room. `size` = (Lx, Ly, Lz) in metres; wall absorption comes from a named material (via A2)
    or an explicit coefficient in [0,1] (0 = perfectly reflective, 1 = anechoic). Compute its reverberation time
    and its early-reflection impulse response between a source and a listener."""

    def __init__(self, size=(5.0, 4.0, 3.0), material="plaster", absorption=None, c=343.0):
        self.L = np.asarray(size, float)
        self.c = float(c)
        if absorption is not None:
            self.alpha = float(absorption)
        else:
            from holographic_acoustic import wall_absorption
            self.alpha = wall_absorption(material)
        self.material = material

    def volume(self):
        return float(np.prod(self.L))

    def surface_area(self):
        lx, ly, lz = self.L
        return float(2.0 * (lx * ly + ly * lz + lx * lz))

    def rt60(self):
        """Sabine reverberation time (seconds): RT60 = 0.161 * V / A, A = absorption * total wall area. Drops as
        the walls get more absorptive -- the reverb tail gets shorter."""
        A = self.alpha * self.surface_area()
        return 0.161 * self.volume() / max(A, 1e-9)

    def _reflectance(self):
        """Pressure reflection coefficient of a wall: sqrt(reflected ENERGY fraction) = sqrt(1 - alpha)."""
        return np.sqrt(max(0.0, 1.0 - self.alpha))

    def _walls(self):
        """The six faces as (axis, plane_position). Mirroring the source across one is one reflection."""
        return [(0, 0.0), (0, self.L[0]), (1, 0.0), (1, self.L[1]), (2, 0.0), (2, self.L[2])]

    @staticmethod
    def _mirror(p, axis, pos):
        q = np.array(p, float); q[axis] = 2.0 * pos - q[axis]; return q

    def reflections(self, source, listener, max_order=2):
        """All contributions to the room impulse response up to `max_order` bounces, as a list of dicts
        {delay, amplitude, order}. Order 0 is the direct sound; order 1 mirrors the source across each wall; order
        2 mirrors those images across a different wall. Delay = path/c, amplitude = product(reflectance)/path."""
        s = np.asarray(source, float); r = np.asarray(listener, float)
        rho = self._reflectance()
        taps = []
        d0 = np.linalg.norm(s - r)
        taps.append({"delay": d0 / self.c, "amplitude": 1.0 / max(d0, 1e-6), "order": 0})   # direct sound
        walls = self._walls()
        if max_order >= 1:
            for (ax, pos) in walls:
                img = self._mirror(s, ax, pos)
                d = np.linalg.norm(img - r)
                taps.append({"delay": d / self.c, "amplitude": rho / max(d, 1e-6), "order": 1})
        if max_order >= 2:
            for (ax1, pos1) in walls:
                img1 = self._mirror(s, ax1, pos1)
                for (ax2, pos2) in walls:
                    if (ax2, pos2) == (ax1, pos1):
                        continue                                    # don't re-reflect off the same wall in a row
                    img2 = self._mirror(img1, ax2, pos2)
                    d = np.linalg.norm(img2 - r)
                    taps.append({"delay": d / self.c, "amplitude": rho * rho / max(d, 1e-6), "order": 2})
        taps.sort(key=lambda t: t["delay"])
        return taps

    def impulse_response(self, source, listener, fs=8000, max_order=2, length_s=None):
        """The room impulse response sampled at `fs` Hz: an array where each reflection deposits its amplitude at
        its arrival time. Convolve a dry sound with this to hear the room. Length defaults to a bit past RT60."""
        taps = self.reflections(source, listener, max_order=max_order)
        length_s = length_s or max(self.rt60() * 1.2, taps[-1]["delay"] * 1.5, 0.05)
        n = int(length_s * fs) + 1
        rir = np.zeros(n)
        for t in taps:
            i = int(round(t["delay"] * fs))
            if 0 <= i < n:
                rir[i] += t["amplitude"]
        return rir, fs

    def direct_delay(self, source, listener):
        """When the direct sound arrives (seconds) = straight-line distance / speed of sound."""
        return float(np.linalg.norm(np.asarray(source, float) - np.asarray(listener, float)) / self.c)


def _selftest():
    """The direct sound arrives at the straight-line delay; first reflections arrive LATER at geometrically
    correct times; a more absorptive room has a shorter RT60 and a weaker reflected tail. Deterministic."""
    room = ShoeboxRoom(size=(6.0, 4.0, 3.0), absorption=0.05, c=343.0)
    src = (1.0, 2.0, 1.5); lis = (5.0, 2.0, 1.5)

    # (1) direct sound: |src-lis| = 4 m -> 4/343 s
    taps = room.reflections(src, lis, max_order=1)
    assert taps[0]["order"] == 0 and abs(taps[0]["delay"] - 4.0 / 343.0) < 1e-9

    # (2) every reflection arrives AFTER the direct sound (longer path) at a geometrically-correct delay
    assert all(t["delay"] >= taps[0]["delay"] - 1e-12 for t in taps)
    # the floor reflection (mirror across z=0): source image at z=-1.5, distance sqrt(4^2 + 3^2)=5 -> 5/343 s
    floor_img_dist = np.sqrt(4.0 ** 2 + (1.5 + 1.5) ** 2)
    assert any(abs(t["delay"] - floor_img_dist / 343.0) < 1e-6 for t in taps if t["order"] == 1)
    # reflections are quieter than the direct sound (they spread further and lost energy to the wall)
    assert all(t["amplitude"] < taps[0]["amplitude"] for t in taps if t["order"] == 1)

    # (3) RT60 drops as the walls get more absorptive (a tiled hall rings; a padded studio is dead)
    live = ShoeboxRoom(size=(6, 4, 3), absorption=0.03)     # hard, reflective
    dead = ShoeboxRoom(size=(6, 4, 3), absorption=0.45)     # soft, absorptive
    assert live.rt60() > dead.rt60() * 5                    # far longer reverb in the live room
    assert live.rt60() > 1.0 and dead.rt60() < 0.5          # rough real-world magnitudes

    # (4) the reflections come back LOUDER in the live room (same geometry/delays, higher reflectance)
    refl_live = sum(t["amplitude"] ** 2 for t in live.reflections(src, lis, max_order=2) if t["order"] >= 1)
    refl_dead = sum(t["amplitude"] ** 2 for t in dead.reflections(src, lis, max_order=2) if t["order"] >= 1)
    assert refl_live > refl_dead                             # a livelier room reflects more energy back
    rir, fs = live.impulse_response(src, lis, max_order=2)   # the sampled RIR is finite and has the direct sound
    assert np.isfinite(rir).all() and rir.max() > 0.0

    # (5) named materials flow through A2: a concrete room rings longer than a carpeted one
    concrete = ShoeboxRoom(size=(6, 4, 3), material="concrete")
    carpet = ShoeboxRoom(size=(6, 4, 3), material="carpet")
    assert concrete.rt60() > carpet.rt60()

    # (6) deterministic
    assert room.reflections(src, lis)[0]["delay"] == room.reflections(src, lis)[0]["delay"]
    print("holographic_roomacoustic selftest OK: direct sound at d/c; reflections later at geometric delays "
          "(floor bounce at %.1f ms); RT60 live %.2fs >> dead %.2fs; concrete rings longer than carpet"
          % (floor_img_dist / 343.0 * 1000, live.rt60(), dead.rt60()))


if __name__ == "__main__":
    _selftest()
