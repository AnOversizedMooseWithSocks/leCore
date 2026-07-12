"""Authoritative game shard: a deterministic fixed-timestep world tick for building games on leCore.

WHY THIS EXISTS (the gap): the engine already has every INGREDIENT of a game -- rigid/soft bodies,
CCD, spatial hashing, a distributed bus/farm, fork/merge worlds, durability -- but no faculty that
composes them into the thing a game SERVER actually is: an authoritative, deterministic, fixed-dt
tick over a set of entities, fed by an ordered player-command queue, answering clients with
area-of-interest snapshots and cheap deltas, and able to HAND OFF entities that cross a region
boundary so a massive world can be sharded across the existing distributed farm.

DESIGN CHOICES (each one a negative avoided):
  * FIXED dt, commands applied at a named tick -- variable dt is the classic source of
    non-reproducible multiplayer state (the "it desynced on the slow machine" bug). Determinism
    here is the engine's constitution, not a nicety: two shards fed the same command log MUST
    produce byte-identical state digests (deterministic lockstep verification for free).
  * COMMANDS, not method calls, mutate the world. Every mutation is a dict with
    (tick, player, seq); within a tick they apply sorted by (player, seq) -- arrival order over
    a network is NOT deterministic, so we never depend on it.
  * SPHERES ONLY in the built-in collision pass, culled by the existing spatial_hash_pairs
    (O(N) expected -- the 'cull, don't batch' primitive). Anything fancier (stacking, joints,
    CCD for bullets) should DELEGATE to rigid_body / time_of_impact per-entity; the shard is an
    orchestrator, not a second physics engine. KEPT NEGATIVE: an earlier sketch re-implemented
    impulse resolution here; rejected -- sibling-of-a-faculty is a discoverability tax.
  * DELTAS are computed against a snapshot the CLIENT names (baseline dict), not against hidden
    server history -- stateless on the wire, so any farm worker can answer any client.
  * hashlib (sha256) for the state digest, never hash() -- PYTHONHASHSEED discipline.

Massive scale story (the seam, documented on purpose): give each shard a `region` (axis-aligned
box). step() reports entities that LEFT the region in 'departed'; a coordinator moves them to the
neighbouring shard over the existing distributed_bus and despawns them here. AOI snapshots mean a
client only ever pays for what is near it, regardless of world size.
"""

import hashlib
import json

import numpy as np

# WHY delegate: the uniform-grid cull already exists and is tested; re-writing it here would be
# the exact "same wheel in a different costume" failure this codebase audits against.
from holographic.misc.holographic_fields import spatial_hash_pairs


class GameShard:
    """One authoritative region of a game world, stepped at a fixed dt.

    Entities are structure-of-arrays (ids, pos, vel, radius, inv_mass) -- NumPy-friendly and
    trivially serialisable. All mutation flows through submit()ed commands; step() advances one
    tick. See module docstring for the design trade-offs.
    """

    #: command ops the shard understands. Kept as data so /invoke callers can introspect it.
    OPS = ("spawn", "despawn", "impulse", "set_vel")

    def __init__(self, dt=1.0 / 60.0, seed=0, gravity=(0.0, 0.0, 0.0), region=None,
                 restitution=0.2):
        # WHY float64 throughout: bit-identical digests across machines matter more here than
        # memory -- a 1e-12 drift flips a collision tie and desyncs a lockstep peer.
        self.dt = float(dt)
        self.tick = 0
        self.gravity = np.asarray(gravity, dtype=np.float64)
        self.restitution = float(restitution)
        # region = (lo, hi) axis-aligned box, or None = unbounded single-shard world.
        self.region = None
        if region is not None:
            lo, hi = region
            self.region = (np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))
        self._rng = np.random.default_rng(seed)  # reserved for game-side randomness; seeded.
        # Structure-of-arrays state. ids are ints chosen by the caller (stable across shards).
        self.ids = []
        self.pos = np.zeros((0, 3), dtype=np.float64)
        self.vel = np.zeros((0, 3), dtype=np.float64)
        self.radius = np.zeros((0,), dtype=np.float64)
        self.inv_mass = np.zeros((0,), dtype=np.float64)
        self._queue = []  # pending commands: list of dicts

    # ------------------------------------------------------------------ commands

    def submit(self, cmd):
        """Queue one command dict: {'tick': int, 'player': str, 'seq': int, 'op': str, ...op args}.

        Commands for past ticks are rejected loudly (an authoritative server never rewrites
        history); future-tick commands wait in the queue. Returns the accepted command.
        """
        if cmd.get("op") not in self.OPS:
            raise ValueError("unknown op %r (known: %s)" % (cmd.get("op"), ", ".join(self.OPS)))
        if int(cmd.get("tick", self.tick)) < self.tick:
            raise ValueError("command for past tick %s (now %s)" % (cmd.get("tick"), self.tick))
        c = dict(cmd)
        c.setdefault("tick", self.tick)
        c.setdefault("player", "")
        c.setdefault("seq", 0)
        self._queue.append(c)
        return c

    def _index_of(self, eid):
        try:
            return self.ids.index(eid)
        except ValueError:
            return -1

    def _apply(self, c):
        op = c["op"]
        if op == "spawn":
            if self._index_of(c["id"]) >= 0:
                return  # idempotent: a re-sent spawn (network retry) must not duplicate.
            self.ids.append(int(c["id"]))
            self.pos = np.vstack([self.pos, np.asarray(c["pos"], dtype=np.float64)[None]])
            v = np.asarray(c.get("vel", (0.0, 0.0, 0.0)), dtype=np.float64)
            self.vel = np.vstack([self.vel, v[None]])
            self.radius = np.append(self.radius, float(c.get("radius", 0.5)))
            self.inv_mass = np.append(self.inv_mass, float(c.get("inv_mass", 1.0)))
        elif op == "despawn":
            i = self._index_of(c["id"])
            if i >= 0:
                self._remove(i)
        elif op == "impulse":
            i = self._index_of(c["id"])
            if i >= 0:
                # dv = J * inv_mass: an immovable (inv_mass 0) entity ignores impulses by math,
                # not by special case.
                self.vel[i] += np.asarray(c["j"], dtype=np.float64) * self.inv_mass[i]
        elif op == "set_vel":
            i = self._index_of(c["id"])
            if i >= 0:
                self.vel[i] = np.asarray(c["v"], dtype=np.float64)

    def _remove(self, i):
        del self.ids[i]
        keep = np.ones(len(self.radius), dtype=bool)
        keep[i] = False
        self.pos = self.pos[keep]
        self.vel = self.vel[keep]
        self.radius = self.radius[keep]
        self.inv_mass = self.inv_mass[keep]

    # ------------------------------------------------------------------ the tick

    def step(self):
        """Advance one fixed tick. Returns {'tick', 'n', 'departed', 'digest'}.

        Order inside a tick (fixed forever -- changing it is a constitution violation):
        1) apply this tick's commands sorted by (player, seq);
        2) semi-implicit Euler integrate;
        3) sphere-sphere positional separation + velocity response, pairs culled by
           spatial_hash_pairs and resolved in sorted-pair order (determinism over speed);
        4) region departure check.
        """
        due = [c for c in self._queue if c["tick"] == self.tick]
        self._queue = [c for c in self._queue if c["tick"] != self.tick]
        for c in sorted(due, key=lambda c: (str(c["player"]), int(c["seq"]))):
            self._apply(c)

        if len(self.ids):
            # Semi-implicit Euler: v first, then x -- the stable order for games (symplectic).
            self.vel += self.gravity * self.dt
            self.pos += self.vel * self.dt
            self._resolve_collisions()

        departed = []
        if self.region is not None and len(self.ids):
            lo, hi = self.region
            out = np.any((self.pos < lo) | (self.pos > hi), axis=1)
            departed = [self.ids[i] for i in np.flatnonzero(out)]

        self.tick += 1
        return {"tick": self.tick, "n": len(self.ids), "departed": departed,
                "digest": self.state_digest()}

    def _resolve_collisions(self):
        r = self.radius
        if len(r) < 2:
            return
        # Cull with the shared grid; the query radius is the largest possible pair reach.
        pairs = spatial_hash_pairs(self.pos, float(2.0 * r.max()))
        # WHY sort: the grid's pair order can depend on layout; resolving in sorted (i, j)
        # order pins the sequential Gauss-Seidel result so digests match across runs/machines.
        for i, j in sorted((int(a), int(b)) for a, b in pairs):
            d = self.pos[j] - self.pos[i]
            dist = float(np.linalg.norm(d))
            min_d = float(r[i] + r[j])
            if dist >= min_d or dist == 0.0:
                continue
            n = d / dist
            wi, wj = self.inv_mass[i], self.inv_mass[j]
            w = wi + wj
            if w == 0.0:
                continue
            # Positional projection (PBD-style): push out of overlap, mass-weighted.
            corr = (min_d - dist) * n / w
            self.pos[i] -= corr * wi
            self.pos[j] += corr * wj
            # Velocity: kill approach speed along the normal with restitution -- enough for
            # game-feel spheres; anything richer delegates to rigid_body per the module note.
            rel = float(np.dot(self.vel[j] - self.vel[i], n))
            if rel < 0.0:
                jimp = -(1.0 + self.restitution) * rel / w
                self.vel[i] -= jimp * wi * n
                self.vel[j] += jimp * wj * n

    def extract(self, eid):
        """Remove entity `eid` and return its full row (JSON-safe) -- the migration primitive.

        WHY not despawn+respawn by hand: a handoff must carry velocity/radius/inv_mass exactly, or
        the entity 'teleports' with subtly different physics on the destination shard -- a desync
        that no digest on EITHER shard would catch (each shard is self-consistent, the WORLD isn't).
        """
        i = self._index_of(eid)
        if i < 0:
            return None
        row = {"id": int(eid), "pos": self.pos[i].tolist(), "vel": self.vel[i].tolist(),
               "radius": float(self.radius[i]), "inv_mass": float(self.inv_mass[i])}
        self._remove(i)
        return row

    # ------------------------------------------------------------------ client views

    def snapshot(self, center=None, radius=None):
        """Full or area-of-interest view: {'tick', 'ids', 'pos', 'vel', 'radius'} (lists, JSON-safe).

        With center+radius only entities within `radius` of `center` are returned -- the
        interest-management primitive that makes a massive world affordable per client.
        """
        if len(self.ids) == 0:
            keep = np.zeros(0, dtype=bool)
        elif center is None or radius is None:
            keep = np.ones(len(self.ids), dtype=bool)
        else:
            c = np.asarray(center, dtype=np.float64)
            keep = np.linalg.norm(self.pos - c, axis=1) <= float(radius)
        idx = np.flatnonzero(keep)
        return {"tick": self.tick,
                "ids": [self.ids[i] for i in idx],
                "pos": self.pos[idx].tolist(),
                "vel": self.vel[idx].tolist(),
                "radius": self.radius[idx].tolist()}

    def delta_since(self, baseline, eps=1e-9):
        """Diff current state against a client-held snapshot: {'tick','added','removed','moved'}.

        'moved' lists (id, pos, vel) only for entities whose position changed by more than eps --
        the cheap wire format. Stateless: the baseline travels FROM the client, so any farm
        worker can answer (no per-client server memory to shard).
        """
        base_ids = list(baseline.get("ids", []))
        base_pos = {int(i): np.asarray(p, dtype=np.float64)
                    for i, p in zip(base_ids, baseline.get("pos", []))}
        cur = set(self.ids)
        added, moved = [], []
        for k, eid in enumerate(self.ids):
            if eid not in base_pos:
                added.append({"id": eid, "pos": self.pos[k].tolist(),
                              "vel": self.vel[k].tolist(), "radius": float(self.radius[k])})
            elif float(np.max(np.abs(self.pos[k] - base_pos[eid]))) > eps:
                moved.append({"id": eid, "pos": self.pos[k].tolist(),
                              "vel": self.vel[k].tolist()})
        removed = [int(i) for i in base_ids if int(i) not in cur]
        return {"tick": self.tick, "added": added, "removed": removed, "moved": moved}

    # ------------------------------------------------------------------ persistence & lockstep

    def state_digest(self):
        """sha256 of the canonical state -- the deterministic-lockstep verifier. Two shards fed
        the same command log at the same ticks MUST agree here, or one has desynced."""
        h = hashlib.sha256()
        h.update(json.dumps({"tick": self.tick, "ids": self.ids}, sort_keys=True).encode())
        for a in (self.pos, self.vel, self.radius, self.inv_mass):
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        return h.hexdigest()

    def save_state(self):
        """Plain-dict export (JSON-safe) for durability / handoff / fork-merge workflows."""
        return {"tick": self.tick, "dt": self.dt, "gravity": self.gravity.tolist(),
                "restitution": self.restitution,
                "region": None if self.region is None else [self.region[0].tolist(),
                                                            self.region[1].tolist()],
                "ids": list(self.ids), "pos": self.pos.tolist(), "vel": self.vel.tolist(),
                "radius": self.radius.tolist(), "inv_mass": self.inv_mass.tolist()}

    def load_state(self, d):
        """Restore from save_state(). Digest after load equals digest before save -- pinned in
        the selftest, because a lossy save is a silent desync factory."""
        self.tick = int(d["tick"])
        self.dt = float(d["dt"])
        self.gravity = np.asarray(d["gravity"], dtype=np.float64)
        self.restitution = float(d["restitution"])
        self.region = None
        if d.get("region") is not None:
            self.region = (np.asarray(d["region"][0], dtype=np.float64),
                           np.asarray(d["region"][1], dtype=np.float64))
        self.ids = [int(i) for i in d["ids"]]
        self.pos = np.asarray(d["pos"], dtype=np.float64).reshape(len(self.ids), 3)
        self.vel = np.asarray(d["vel"], dtype=np.float64).reshape(len(self.ids), 3)
        self.radius = np.asarray(d["radius"], dtype=np.float64)
        self.inv_mass = np.asarray(d["inv_mass"], dtype=np.float64)
        return self


def shard_key(pos, cell=64.0):
    """Which shard owns a position, for grid-sharding a massive world: integer (i, j, k) cell.

    Pure and stateless on purpose -- every node in the farm computes the same owner for the same
    position with no coordination round-trip.
    """
    p = np.asarray(pos, dtype=np.float64)
    return tuple(int(x) for x in np.floor(p / float(cell)))


def run_shard(commands, ticks, dt=1.0 / 60.0, seed=0, gravity=(0.0, 0.0, 0.0),
              region=None, restitution=0.2, state=None, aoi=None):
    """One-shot, JSON-in/JSON-out shard run -- the /invoke-callable face of GameShard.

    WHY this exists: the object-returning game_shard() faculty is perfect in-process, but an HTTP
    agent can only hold JSON. So: feed a (possibly saved) `state`, a command list, and a tick
    count; get back the final save_state() blob, the per-tick digests (lockstep audit trail),
    every departure, and an optional AOI snapshot {'center':..., 'radius':...}. Stateless on the
    wire -- the state blob travels with the client, so any farm worker can serve the next call.
    """
    s = GameShard(dt=dt, seed=seed, gravity=gravity, region=region, restitution=restitution)
    if state is not None:
        s.load_state(state)
    for c in commands:
        s.submit(dict(c))
    digests, departed = [], []
    for _ in range(int(ticks)):
        out = s.step()
        digests.append(out["digest"])
        departed.extend(out["departed"])
    snap = None
    if aoi is not None:
        snap = s.snapshot(center=aoi.get("center"), radius=aoi.get("radius"))
    return {"state": s.save_state(), "digests": digests, "departed": departed, "aoi": snap}


class ShardWorld:
    """A grid of GameShards with deterministic cross-shard entity migration -- the massive world.

    Each grid cell (see shard_key) owns one GameShard whose region is exactly that cell's box.
    tick() steps every shard in SORTED key order (determinism: dict iteration order must never
    matter), then migrates each departed entity to the shard that now owns its position, spawning
    it there for the NEXT tick. Shards are created lazily -- an empty universe costs nothing,
    which is what makes 'absolutely massive' affordable: cost tracks OCCUPIED cells, not world
    size.

    DISTRIBUTION SEAM (documented, not hidden): in-process, migration is a direct spawn command.
    Across machines, call tick(collect_only=True) to get the same migration rows as JSON payloads
    keyed by destination shard -- publish them on the distributed bus (topic per shard key) and
    let each remote node feed them to its own shard's submit(). The payload format is identical
    either way, so a world can move from one process to a farm without changing its data.

    KEPT NEGATIVE: stepping shards in parallel threads was rejected here -- BLAS thread pinning
    plus per-shard determinism does NOT compose into world determinism unless migration is
    barriered per tick anyway, so the parallel win belongs at the FARM level (one shard per
    worker), not inside this class.
    """

    def __init__(self, cell=64.0, dt=1.0 / 60.0, seed=0, gravity=(0.0, 0.0, 0.0),
                 restitution=0.2):
        self.cell = float(cell)
        self.dt = float(dt)
        self.seed = int(seed)
        self.gravity = tuple(float(g) for g in gravity)
        self.restitution = float(restitution)
        self.shards = {}   # key tuple -> GameShard
        self.owner = {}    # entity id -> key tuple (the world's routing table)
        self.tick_count = 0

    def _region_of(self, key):
        lo = np.asarray(key, dtype=np.float64) * self.cell
        return (lo, lo + self.cell)

    def _shard_at(self, key):
        s = self.shards.get(key)
        if s is None:
            # WHY seed mixing via sha256 (not hash()): every node must derive the SAME per-shard
            # seed for the same cell, machine-independently.
            h = hashlib.sha256(("%d:%r" % (self.seed, key)).encode()).digest()
            sub = int.from_bytes(h[:8], "big")
            s = GameShard(dt=self.dt, seed=sub, gravity=self.gravity,
                          region=self._region_of(key), restitution=self.restitution)
            s.tick = self.tick_count  # a late-created shard joins at the world's clock
            self.shards[key] = s
        return s

    def spawn(self, eid, pos, vel=(0.0, 0.0, 0.0), radius=0.5, inv_mass=1.0, player="", seq=0):
        """Route a spawn to the owning shard (command at the world's current tick)."""
        key = shard_key(pos, self.cell)
        self._shard_at(key).submit({"tick": self.tick_count, "player": player, "seq": seq,
                                    "op": "spawn", "id": int(eid), "pos": tuple(pos),
                                    "vel": tuple(vel), "radius": radius, "inv_mass": inv_mass})
        self.owner[int(eid)] = key
        return key

    def submit(self, cmd):
        """Route any command to the shard that owns cmd['id'] (spawns route by position)."""
        if cmd.get("op") == "spawn":
            return self.spawn(cmd["id"], cmd["pos"], vel=cmd.get("vel", (0, 0, 0)),
                              radius=cmd.get("radius", 0.5), inv_mass=cmd.get("inv_mass", 1.0),
                              player=cmd.get("player", ""), seq=cmd.get("seq", 0))
        key = self.owner.get(int(cmd.get("id", -1)))
        if key is None:
            raise KeyError("unknown entity id %r" % cmd.get("id"))
        c = dict(cmd)
        c.setdefault("tick", self.tick_count)
        return self.shards[key].submit(c)

    def tick(self, collect_only=False):
        """Step every shard once, then migrate departures. Returns
        {'tick', 'n', 'migrated' or 'handoffs', 'digest'}.

        collect_only=True skips the local re-spawn and instead returns the migration rows as
        {dest_key_str: [row, ...]} -- the exact payloads a bus transport publishes.
        """
        moved = []
        for key in sorted(self.shards):                     # determinism: sorted, always
            out = self.shards[key].step()
            for eid in out["departed"]:
                row = self.shards[key].extract(eid)
                if row is not None:
                    moved.append(row)
        self.tick_count += 1
        handoffs = {}
        for row in sorted(moved, key=lambda r: r["id"]):    # stable migration order
            dest = shard_key(row["pos"], self.cell)
            if collect_only:
                handoffs.setdefault(repr(dest), []).append(row)
                self.owner.pop(row["id"], None)             # ownership leaves this node
            else:
                self._shard_at(dest).submit({"tick": self.tick_count, "player": "", "seq": 0,
                                             "op": "spawn", "id": row["id"], "pos": row["pos"],
                                             "vel": row["vel"], "radius": row["radius"],
                                             "inv_mass": row["inv_mass"]})
                self.owner[row["id"]] = dest
        n = sum(len(s.ids) for s in self.shards.values())
        res = {"tick": self.tick_count, "n": n, "digest": self.world_digest()}
        if collect_only:
            res["handoffs"] = handoffs
        else:
            res["migrated"] = [r["id"] for r in sorted(moved, key=lambda r: r["id"])]
        return res

    def receive(self, rows):
        """Feed migration rows from a remote node (the other half of collect_only): spawn each on
        its owning local shard at the current tick. The bus transport's delivery hook."""
        for row in sorted(rows, key=lambda r: r["id"]):
            dest = shard_key(row["pos"], self.cell)
            self._shard_at(dest).submit({"tick": self.tick_count, "player": "", "seq": 0,
                                         "op": "spawn", "id": row["id"], "pos": row["pos"],
                                         "vel": row["vel"], "radius": row["radius"],
                                         "inv_mass": row["inv_mass"]})
            self.owner[row["id"]] = dest

    def snapshot(self, center, radius):
        """AOI across shard boundaries: query every shard whose cell box intersects the sphere --
        a player standing on a border sees both sides, with no seam."""
        c = np.asarray(center, dtype=np.float64)
        out = {"tick": self.tick_count, "ids": [], "pos": [], "vel": [], "radius": []}
        for key in sorted(self.shards):
            lo, hi = self._region_of(key)
            # closest point of the box to the sphere centre; cheap exact box/sphere test.
            d = np.linalg.norm(np.clip(c, lo, hi) - c)
            if d > radius:
                continue
            s = self.shards[key].snapshot(center=center, radius=radius)
            for f in ("ids", "pos", "vel", "radius"):
                out[f].extend(s[f])
        return out

    def world_digest(self):
        """sha256 over (key, shard digest) pairs in sorted key order -- the WORLD's lockstep
        verifier; two farms running the same command log must agree here."""
        h = hashlib.sha256()
        for key in sorted(self.shards):
            h.update(repr(key).encode())
            h.update(self.shards[key].state_digest().encode())
        return h.hexdigest()


def run_world(commands, ticks, cell=64.0, dt=1.0 / 60.0, seed=0, gravity=(0.0, 0.0, 0.0),
              restitution=0.2, aoi=None):
    """One-shot JSON face of ShardWorld: spawn/drive entities across a sharded world and return
    per-tick digests, every migration, and an optional cross-shard AOI snapshot."""
    w = ShardWorld(cell=cell, dt=dt, seed=seed, gravity=gravity, restitution=restitution)
    for c in commands:
        w.submit(dict(c))
    digests, migrated = [], []
    for _ in range(int(ticks)):
        out = w.tick()
        digests.append(out["digest"])
        migrated.extend(out["migrated"])
    snap = w.snapshot(aoi["center"], aoi["radius"]) if aoi else None
    return {"tick": w.tick_count, "n": sum(len(s.ids) for s in w.shards.values()),
            "shards": len(w.shards), "digests": digests, "migrated": migrated, "aoi": snap}


class BusShardHost:
    """One farm node's slice of a sharded game world, exchanging handoffs over the EXISTING bus.

    THE LAYERING (why this class is thin on purpose): the distributed system is the DATA layer --
    holographic_coordinator's own kept negative says non-monoid feedback steps (a game tick IS
    one) run WHOLE on one worker; the coordinator places work, the bus moves messages, presence
    says who's alive. The game world is the INTERACTION layer -- it only defines what one
    worker's stateful unit is. This adapter is the handshake between them and NOTHING more: it
    owns a set of cell keys, steps its ShardWorld with collect_only=True, publishes each handoff
    row on the bus topic 'game/<world_id>/shard/<i>,<j>,<k>' (one topic per destination cell --
    the bus's pattern matching IS the routing table), and subscribes to its OWN cells' topics,
    feeding arrivals to receive(). Local MessageBus and cross-machine DistributedBus are the SAME
    call -- transport is the bus's business, not this class's.

    KEPT NEGATIVE: an earlier sketch gave this class its own peer list and HTTP posts -- rejected
    as a straight duplicate of DistributedBus; if the bus needs a capability (acks, replay), it
    gets built IN the bus where every other faculty inherits it too.
    """

    def __init__(self, bus, world, own_keys, world_id="w0"):
        self.bus = bus
        self.world = world
        self.world_id = str(world_id)
        self.own_keys = set(tuple(int(x) for x in k) for k in own_keys)
        self._inbox = []
        self._unsubs = []
        for k in sorted(self.own_keys):
            # WHY one topic per cell (not one per node): ownership can move between nodes without
            # anyone else re-learning topology -- the topic is the address, the node is a tenant.
            self._unsubs.append(self.bus.subscribe(self.topic(k),
                                                   lambda m: self._inbox.append(m)))

    def topic(self, key):
        return "game/%s/shard/%s" % (self.world_id, ",".join(str(int(x)) for x in key))

    def owns(self, key):
        return tuple(key) in self.own_keys

    def tick(self):
        """One node tick: deliver queued arrivals, step own shards, publish departures. Returns
        {'tick', 'n', 'sent', 'received'}. Determinism: arrivals apply in sorted-id order via
        receive(); the bus's local delivery is the deterministic MessageBus."""
        # WHY arrivals apply AFTER this round's step: the in-process ShardWorld spawns a migrated
        # entity at tick_count AFTER the increment (it joins the NEXT round). Delivering before
        # the step gave the traveller one extra integration step vs the single-process reference
        # -- a real off-by-one caught by the parity selftest, kept loud here. Constraint that
        # follows (documented, not hidden): rounds are BARRIERED -- publish in round R, join at
        # R+1 -- so within a round the sender must tick before the receiver polls. The farm's
        # coordinator already runs barriered waves (run_waves); that is the placement to use.
        rows = [dict(msg.payload) for msg in self._inbox]
        self._inbox = []
        out = self.world.tick(collect_only=True)
        sent = 0
        for dest_repr, drows in sorted(out["handoffs"].items()):
            dest = tuple(int(x) for x in dest_repr.strip("()").split(","))
            for row in drows:
                if self.owns(dest):
                    self.world.receive([row])   # short-circuit: still ours, no wire needed
                else:
                    self.bus.publish(self.topic(dest), dict(row))
                    sent += 1
        if rows:
            self.world.receive(rows)     # arrivals join at the NEW tick_count (next round's step)
        return {"tick": self.world.tick_count, "n": sum(len(s.ids) for s in self.world.shards.values()),
                "sent": sent, "received": len(rows)}

    def close(self):
        for u in self._unsubs:
            u()


class WorldStreamer:
    """Per-client delta feed over a ShardWorld -- the logic behind the service's game SSE channel.

    WHY here and not in the service: the service is a thin route table by design; anything with a
    contract worth testing lives in a module where the selftest can pin it. The streamer holds one
    BASELINE SNAPSHOT PER SESSION and answers next_event(session, center, radius) with the delta
    since that client's last event -- first contact returns the full AOI as 'added' (a client
    joining mid-game must not need history). The baseline lives server-side ONLY for the lifetime
    of a stream connection; the stateless run_* faces remain the cross-worker path (a session is
    an optimisation, not an address).

    advance_per_event: if True (single-feed default) the streamer ticks the world once per event
    -- the demo/live loop. Run it False when something else owns the clock (a bus host round, a
    second stream) -- two advancing feeds would double-step the world; the selftest pins the
    False path's digest stability.
    """

    def __init__(self, world, advance_per_event=True):
        self.world = world
        self.advance = bool(advance_per_event)
        self._baselines = {}   # session -> last snapshot sent

    def next_event(self, session, center=None, radius=None):
        """One SSE event body: {'tick','added','removed','moved','digest','n'}."""
        if self.advance:
            self.world.tick()
        if center is not None and radius is not None:
            snap = self.world.snapshot(center=center, radius=float(radius))
        else:
            # whole-world view: concatenate every shard (small worlds / admin feeds).
            snap = {"tick": self.world.tick_count, "ids": [], "pos": [], "vel": [], "radius": []}
            for key in sorted(self.world.shards):
                s = self.world.shards[key].snapshot()
                for f in ("ids", "pos", "vel", "radius"):
                    snap[f].extend(s[f])
        base = self._baselines.get(session, {"ids": [], "pos": []})
        # Diff in snapshot space (not against live shard state): the baseline the CLIENT holds is
        # the last snapshot it was SENT, and AOI membership changes must appear as added/removed.
        base_pos = {int(i): p for i, p in zip(base["ids"], base["pos"])}
        added, moved = [], []
        for k, eid in enumerate(snap["ids"]):
            if eid not in base_pos:
                added.append({"id": eid, "pos": snap["pos"][k], "vel": snap["vel"][k],
                              "radius": snap["radius"][k]})
            elif max(abs(a - b) for a, b in zip(snap["pos"][k], base_pos[eid])) > 1e-9:
                moved.append({"id": eid, "pos": snap["pos"][k], "vel": snap["vel"][k]})
        cur = set(snap["ids"])
        removed = [int(i) for i in base["ids"] if int(i) not in cur]
        self._baselines[session] = snap
        return {"tick": self.world.tick_count, "added": added, "removed": removed,
                "moved": moved, "digest": self.world.world_digest(), "n": len(snap["ids"])}

    def drop(self, session):
        """Forget a client's baseline (its EventSource closed)."""
        self._baselines.pop(session, None)


def _selftest():
    # 1) DETERMINISM: identical command logs -> identical digests, tick by tick.
    def build():
        s = GameShard(dt=1.0 / 60.0, seed=0, gravity=(0, -9.8, 0))
        s.submit({"tick": 0, "player": "a", "seq": 0, "op": "spawn", "id": 1,
                  "pos": (0, 5, 0), "radius": 0.5})
        s.submit({"tick": 0, "player": "b", "seq": 0, "op": "spawn", "id": 2,
                  "pos": (0.3, 5.6, 0), "radius": 0.5})
        s.submit({"tick": 3, "player": "a", "seq": 1, "op": "impulse", "id": 1, "j": (2, 0, 0)})
        return s
    s1, s2 = build(), build()
    for _ in range(30):
        d1, d2 = s1.step(), s2.step()
        assert d1["digest"] == d2["digest"], "lockstep desync: same commands, different state"

    # ...and command ARRIVAL order must not matter (only (player, seq) does).
    s3 = GameShard(dt=1.0 / 60.0, seed=0, gravity=(0, -9.8, 0))
    s3.submit({"tick": 0, "player": "b", "seq": 0, "op": "spawn", "id": 2,
               "pos": (0.3, 5.6, 0), "radius": 0.5})
    s3.submit({"tick": 0, "player": "a", "seq": 0, "op": "spawn", "id": 1,
               "pos": (0, 5, 0), "radius": 0.5})
    s3.submit({"tick": 3, "player": "a", "seq": 1, "op": "impulse", "id": 1, "j": (2, 0, 0)})
    for _ in range(30):
        d3 = s3.step()
    assert d3["digest"] == d1["digest"], "arrival order leaked into state"

    # 2) COLLISION: two overlapping equal spheres separate to >= sum of radii (1e-9 slack).
    s = GameShard(seed=0)
    s.submit({"tick": 0, "player": "a", "seq": 0, "op": "spawn", "id": 1, "pos": (0, 0, 0),
              "radius": 0.5})
    s.submit({"tick": 0, "player": "a", "seq": 1, "op": "spawn", "id": 2, "pos": (0.4, 0, 0),
              "radius": 0.5})
    s.step()
    gap = float(np.linalg.norm(s.pos[1] - s.pos[0]))
    assert gap >= 1.0 - 1e-9, "overlap not resolved: %r" % gap

    # 3) AOI + DELTA: only the entity near the query center is returned; only the moved one
    #    appears in the delta.
    s = GameShard(seed=0)
    s.submit({"tick": 0, "player": "a", "seq": 0, "op": "spawn", "id": 1, "pos": (0, 0, 0)})
    s.submit({"tick": 0, "player": "a", "seq": 1, "op": "spawn", "id": 2, "pos": (100, 0, 0)})
    s.step()
    base = s.snapshot()
    aoi = s.snapshot(center=(0, 0, 0), radius=10)
    assert aoi["ids"] == [1], "AOI returned %r" % aoi["ids"]
    s.submit({"tick": 1, "player": "a", "seq": 0, "op": "set_vel", "id": 2, "v": (1, 0, 0)})
    s.step()
    d = s.delta_since(base)
    assert [m["id"] for m in d["moved"]] == [2] and not d["added"] and not d["removed"]

    # 4) REGION HANDOFF: an entity crossing the box shows up in 'departed'.
    s = GameShard(seed=0, region=((-1, -1, -1), (1, 1, 1)))
    s.submit({"tick": 0, "player": "a", "seq": 0, "op": "spawn", "id": 7, "pos": (0.9, 0, 0),
              "vel": (30, 0, 0)})
    out = s.step()
    assert out["departed"] == [7], "departure missed: %r" % out
    assert shard_key((0.9, 0, 0), cell=64.0) != shard_key((90.0, 0, 0), cell=64.0)

    # 5) SAVE/LOAD round-trip is digest-identical (a lossy save is a silent desync).
    dump = s.save_state()
    s2 = GameShard().load_state(dump)
    assert s2.state_digest() == s.state_digest(), "save/load changed state"

    # 6) run_shard (the /invoke face) matches the object path digest-for-digest, and is
    #    resumable from its own returned state blob.
    cmds = [{"tick": 0, "player": "a", "seq": 0, "op": "spawn", "id": 1, "pos": (0, 5, 0)}]
    r1 = run_shard(cmds, 10, gravity=(0, -9.8, 0))
    ref = GameShard(gravity=(0, -9.8, 0))
    ref.submit(dict(cmds[0]))
    for _ in range(10):
        ref_d = ref.step()["digest"]
    assert r1["digests"][-1] == ref_d, "run_shard diverged from GameShard"
    r2 = run_shard([], 5, gravity=(0, -9.8, 0), state=r1["state"])
    for _ in range(5):
        ref_d = ref.step()["digest"]
    assert r2["digests"][-1] == ref_d, "resume-from-state diverged"

    # 7) SHARD WORLD: an entity crossing a cell boundary migrates, keeps its exact velocity,
    #    and the world stays deterministic (two identical worlds agree digest-for-digest).
    def build_world():
        w = ShardWorld(cell=4.0, dt=0.1, seed=0)
        w.spawn(1, (3.5, 1.0, 1.0), vel=(2.0, 0.0, 0.0))    # crosses x=4 once, tick 3
        w.spawn(2, (1.0, 1.0, 1.0))
        return w
    w1, w2 = build_world(), build_world()
    saw_migration = False
    for _ in range(5):
        o1, o2 = w1.tick(), w2.tick()
        assert o1["digest"] == o2["digest"], "world lockstep desync"
        if 1 in o1["migrated"]:
            saw_migration = True
    assert saw_migration, "boundary crossing never migrated"
    assert w1.owner[1] != w1.owner[2], "entity 1 should live in a different cell now"
    k = w1.owner[1]
    i = w1.shards[k].ids.index(1)
    assert abs(w1.shards[k].vel[i][0] - 2.0) < 1e-9, "velocity not carried across handoff"
    # cross-shard AOI: a query centred between the two entities sees BOTH despite the seam.
    snap = w1.snapshot(center=(4.0, 1.0, 1.0), radius=6.0)
    assert sorted(snap["ids"]) == [1, 2], "AOI missed an entity across the boundary: %r" % snap["ids"]

    # 8) collect_only + receive() round-trip equals in-process migration (the bus seam contract).
    wa, wb = build_world(), build_world()
    for _ in range(5):
        ra = wa.tick()
        rb = wb.tick(collect_only=True)
        rows = [r for rs in rb["handoffs"].values() for r in rs]
        wb.receive(rows)
        assert wa.world_digest() == wb.world_digest(), "bus-transport path diverged from in-process"

    # 9) run_world (the /invoke face) reports the migration and matches the object path.
    r = run_world([{"op": "spawn", "id": 1, "pos": (3.5, 1.0, 1.0), "vel": (2, 0, 0)},
                   {"op": "spawn", "id": 2, "pos": (1.0, 1.0, 1.0)}], 5, cell=4.0, dt=0.1,
                  aoi={"center": (4.0, 1.0, 1.0), "radius": 6.0})
    assert r["migrated"] == [1] and r["shards"] == 2 and sorted(r["aoi"]["ids"]) == [1, 2]
    assert r["digests"][-1] == w1.world_digest(), "run_world diverged from ShardWorld"

    # 10) BUS CONJUNCTION: two BusShardHosts on the existing MessageBus, split cell ownership,
    #     produce the SAME world as one in-process ShardWorld -- entity by entity, exactly.
    from holographic.scene_and_pipeline.holographic_distbus import MessageBus
    def two_node_world():
        bus = MessageBus()
        wa = ShardWorld(cell=4.0, dt=0.1, seed=0)   # node A owns x-cell 0
        wb = ShardWorld(cell=4.0, dt=0.1, seed=0)   # node B owns x-cell 1
        ha = BusShardHost(bus, wa, [(0, 0, 0)], world_id="t")
        hb = BusShardHost(bus, wb, [(1, 0, 0)], world_id="t")
        wa.spawn(1, (3.5, 1.0, 1.0), vel=(2.0, 0.0, 0.0))
        wa.spawn(2, (1.0, 1.0, 1.0))
        return bus, ha, hb
    _, ha, hb = two_node_world()
    ref = ShardWorld(cell=4.0, dt=0.1, seed=0)
    ref.spawn(1, (3.5, 1.0, 1.0), vel=(2.0, 0.0, 0.0))
    ref.spawn(2, (1.0, 1.0, 1.0))
    crossed = False
    for _ in range(6):
        oa, ob = ha.tick(), hb.tick()
        ref.tick()
        if oa["sent"]:
            crossed = True
    assert crossed, "entity never crossed the node boundary over the bus"
    assert 1 in hb.world.owner and hb.world.owner[1] == (1, 0, 0), "node B never received entity 1"
    # exact-state parity with the single-process reference (positions AND velocities):
    k = (1, 0, 0)
    i_ref = ref.shards[k].ids.index(1)
    i_b = hb.world.shards[k].ids.index(1)
    assert np.allclose(ref.shards[k].pos[i_ref], hb.world.shards[k].pos[i_b], atol=1e-12)
    assert np.allclose(ref.shards[k].vel[i_ref], hb.world.shards[k].vel[i_b], atol=1e-12)

    # 11) STREAMER: first event is the full AOI as 'added'; later events carry only changes;
    #     AOI exits appear as 'removed'; a non-advancing streamer never moves the clock.
    w = ShardWorld(cell=8.0, dt=0.1, seed=0)
    # 2 sits OFF entity 1's path (first fixture had them touching at spawn: the collision pass
    # transferred velocity and BOTH moved -- right physics, wrong test; kept as the reminder).
    w.spawn(1, (1.0, 1.0, 1.0), vel=(3.0, 0.0, 0.0))   # will leave the AOI sphere
    w.spawn(2, (1.5, 2.5, 1.0))
    st = WorldStreamer(w, advance_per_event=True)
    e1 = st.next_event("c1", center=(1.5, 1, 1), radius=2.0)
    assert sorted(x["id"] for x in e1["added"]) == [1, 2] and not e1["moved"]
    e2 = st.next_event("c1", center=(1.5, 1, 1), radius=2.0)
    assert [x["id"] for x in e2["moved"]] == [1] and not e2["added"], e2
    removed_seen = False
    for _ in range(10):
        e = st.next_event("c1", center=(1.5, 1, 1), radius=2.0)
        removed_seen = removed_seen or (1 in e["removed"])
    assert removed_seen, "entity 1 left the AOI but never appeared in 'removed'"
    frozen = WorldStreamer(w, advance_per_event=False)
    d0 = w.world_digest()
    frozen.next_event("admin")
    assert w.world_digest() == d0, "non-advancing streamer moved the clock"

    print("holographic_gameshard selftest OK")


if __name__ == "__main__":
    _selftest()
