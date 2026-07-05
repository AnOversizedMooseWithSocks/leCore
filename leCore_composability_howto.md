# leCore Composability — Complete Guide
### what the composability work enabled, and how to use it

*Written as if the Composability backlog is done — every capability below is available now. It's plain usage,
stdlib only, readable. The single idea that ties it all together: **every node speaks the same two endpoints**
(`GET /tools`, `POST /invoke`), **merges by a conflict-free monoid**, and **shares worlds as a seed + sparse
deltas** — so anything composes with anything, from a solo laptop to a public farm.*

---

## The mental model (read this once)

Four rules make everything else follow:
1. **Every node is a tool and can use tools.** leCore serves `/tools` + `/invoke`, and can call another node's
   `/tools`+`/invoke`, an LLM (a callable), or a shell command — all as `orchestrator.Tool`s a planner can chain.
2. **Every actor is a `Principal`.** An agent, a user, a service, or a whole guest leCore instance gets one scoped
   identity (private overlay + namespace + inbox topic + provenance role), so signals and state never cross.
3. **Combine is a monoid; disagreement is detected.** Results from farm buckets, forked worlds, or many solvers
   merge conflict-free where they agree; where they diverge, `opponent` flags it (it never guesses).
4. **A world is a seed + deltas.** The base regenerates deterministically from a seed anywhere; only the sparse
   changes travel. Joining a shared world costs a seed and a delta feed, not a database.

Everything below is these four rules applied at growing scale.

---

## 1. Composability basics — leCore as a tool, and leCore using tools

**What this enabled:** leCore drops into any pipeline in both directions — a harness can drive it, and it can
drive LLMs, programs, and other nodes.

**Serve leCore as a tool.** The standalone service exposes a self-describing tool list and an invoke endpoint:
```python
from holographic_service import serve
serve(host="127.0.0.1", port=8080, token="secret")   # GET /tools  ->  the manifest ; POST /invoke {name,args} -> run one
```
Any client (curl, a harness, another leCore) lists `/tools` and calls `/invoke`. That's leCore as a tool provider.

**Use a tool from leCore.** Three kinds, all registered the same way — as a `Tool` with a callable:
```python
from holographic_unified import UnifiedMind
from holographic_toolclient import remote_tools
mind = UnifiedMind(dim=1024, seed=0)

mind.attach_llm(llm)                                  # an LLM: a callable text->text (agent_bridge; imports no SDK)
for t in remote_tools("http://other-node:8080"):     # another leCore/harness: its /tools become local tools
    mind.orchestrator.register(t)
mind.orchestrator.register_command("ffmpeg", ...)     # a shell program (allowlisted, sandboxed)
```

**The refine loop — the middle of a pipeline.** Produce a result, have a *critic* (any callable — a small model,
a leCore metric, a human) judge it, adjust, retry until good enough:
```python
from holographic_refine import refine
out = refine(produce = lambda: mind.simulate(params),
             critique = lambda r: score_it(r),        # 0..1: a model, measure(), or opponent
             adjust   = lambda r, s: mind.simulate(tweak(params, s)),
             accept=0.9, budget=6)                     # -> {"result","score","accepted","tries"}
```

## 2. Farms — distributed compute

**What this enabled:** the same work runs on one core, on all your cores, or across a farm of machines, without
changing your code — the coordinator picks a backend and reassembles by the monoid.

**On your own machine (offload heavy or blocking work):**
```python
from holographic_coordinator import Coordinator, LocalPool
farm = Coordinator(LocalPool())                       # process pool + shared memory; each worker its own GIL
result = farm.run(buckets, worker, cache, reduce=reduce_sum)   # e.g. mesh work, render tiles, sim bricks
```

**Across machines (a real farm):** start a worker daemon on each node, point the coordinator at them:
```python
# on each farm node:
from holographic_coordinator import serve_worker
serve_worker(host="0.0.0.0", port=9000, token="farm-secret")

# on the client:
from holographic_coordinator import Coordinator, NetworkFarm
farm = Coordinator(NetworkFarm(nodes=["node1:9000","node2:9000","node3:9000"], token="farm-secret"))
result = farm.run(buckets, worker, cache, reduce=reduce_sum)   # partitioned out, reassembled by the monoid
```
The existing tiled faculties (`render`, `octree`, `pathtrace`, `tree` recall) route through this one backend, so
"render on the farm" and "simulate on the farm" are the same farm. `jobs` lets you start/pause/cancel long runs
with checkpoints that survive a restart.

## 3. Multi-user sessions — shared workspaces, fork/merge, and choosing what to share

**What this enabled:** several people work in one workspace, any of them can fork to a private copy, work alone,
and merge back with a policy — and a host can invite guests and share exactly what it chooses.

**A shared (multiplayer) workspace.** Each user is a `Principal`; they share a workspace, coordinate over the bus:
```python
from holographic_principal import Principal
alice = Principal(mind.base, mind.db, "alice", workspace="lab", kind="user")
bob   = Principal(mind.base, mind.db, "bob",   workspace="lab", kind="user")
# each writes only its own namespace; edits to the shared scene flow through the bus + the merge policy
```

**Fork to single-player, then merge back.** Because a world is a seed + deltas, forking is cheap:
```python
mine = mind.workspace.fork("lab")                     # regenerate the base from the seed + my own delta layer
# ... work alone; edits accumulate as deltas ...
from holographic_merge import merge_forks
res = merge_forks([mine.delta, theirs.delta], policy="select")
#   policy: "auto" (only agreements) | "left"/"right" (one side wins) | "select" (surface conflicts) | callable (per-slot)
mind.apply(res["merged"]);  hand_to_user(res["conflicts"])   # auto-merge agreements; resolve real conflicts by choice
```

**Invite guests and choose what to share (access control).** A guest sees nothing until granted:
```python
code = mind.invite(kind="user", grants={"read": ["lab/scene"]})   # an invite token; grants specific namespaces
# the guest connects with `code`; they can READ lab/scene, WRITE only their own namespace, nothing else
mind.grant(alice, read="lab/notes")                    # share more later, selectively
mind.revoke(alice, read="lab/notes")                   # or stop sharing
```
Under the hood this is the symmetric twin of leCore's existing "write only your own" rule: a `_require_readable`
check gated by per-principal grants, so "choose what to share beyond the workspace" is an explicit grant, never a
default.

## 4. Agent connectivity

**What this enabled:** dozens to thousands of agents connect to a host, use leCore as a tool, and coordinate —
without crossed signals, and without flooding the host.

**Many agents, each isolated.** Every agent is a `Principal`; messages are directed and sender-tagged:
```python
agents = [Principal(mind.base, mind.db, f"agent{i}", workspace="lab", kind="agent") for i in range(1000)]
agents[3].send(mind.bus, to="agent7", payload={...})   # only agent7's inbox sees it; sender + seq stamp every message
```

**Who's connected (presence).** The registry tracks live actors (heartbeat), so agents and nodes find each other:
```python
mind.registry.announce(agents[3])                      # "agent3 is online"
online = mind.registry.list(kind="agent")              # discover peers to coordinate with
```

**Agents use leCore as a tool.** They hit the same `/tools` + `/invoke` a harness uses — `route` narrows the
choice so even a small agent isn't drowned. **Backpressure** keeps a burst of agents from flooding a slow
subscriber: the bus uses a bounded per-subscriber queue with a drop/coalesce policy, so the host degrades
gracefully instead of falling over.

## 5. Federation — guest leCore instances as peers

**What this enabled:** a guest isn't limited to a thin client — a whole leCore instance can connect as a peer,
using the host's tools and offering its own.

```python
# a guest node connects to a host and pulls its tools in:
for t in remote_tools("http://host:8080", token=my_invite_code):
    guest_mind.orchestrator.register(t)                # now the guest can call the host's faculties
# and the guest can expose ITS tools back, gated the same way:
serve(host="0.0.0.0", port=8081, token="guest-secret") # the host (or others) can call the guest's /tools too
```
A guest is just a `Principal(kind="peer")` that speaks `/tools`+`/invoke`, under the same invites + read-grants as
any other actor. Composition is recursive: host → guest → guest, each only seeing what it's granted.

## 6. The extreme case — a public, federated science farm

**What this enabled:** the union of everything — a big farm of leCore nodes, public guests connecting their own
instances, and agents joining to coordinate on hard problems. Every simpler setup above is a subset of this.

```python
# 1) stand up the farm (section 2) and a distributed bus so coordination spans machines:
from holographic_coordinator import Coordinator, NetworkFarm
from holographic_bus import DistributedBus
farm = Coordinator(NetworkFarm(nodes=[...], token=SECRET))
bus  = DistributedBus(farm)                            # same publish/subscribe API, routed across nodes

# 2) admit guests + agents as Principals, each invited and granted only what they need (sections 3-5):
code = mind.invite(kind="peer", grants={"read": ["problem/spec"]})   # a guest leCore may read the problem, nothing else

# 3) coordinate the solve: many principals produce candidates on the farm; opponent aggregates them:
from holographic_opponent import opponent_channels
def solve(problem):
    candidates = [p.propose(problem) for p in mind.registry.list()]  # many solvers, farmed out
    agg = opponent_channels(candidates)                # shared = consensus; purple = what no single solver saw
    return refine(produce=lambda: agg["shared"], critique=verify_physics, adjust=nudge, accept=0.95)

# 4) the shared problem is a seed + deltas, so joining costs a seed and a delta feed, not a database.
```

**On a public deployment the guardrails are not optional** (on a trusted LAN they relax):
- **untrusted nodes** → run each bucket redundantly and accept only on **agreement** (`opponent` is the detector);
- **integrity** → `verify`-checked, signed deltas; a corrupted or forged delta is caught;
- **access** → per-principal invites + read-grants (nobody sees more than granted; merges into shared state stay
  policy-gated so no guest overwrites the commons);
- **exposure** → bind behind auth/TLS, never openly; the command backend runs only an allowlist, sandboxed.

None of that is extra machinery — it's the same voting, verify, grant, and merge discipline from the sections
above, switched on.

## Honest notes (operating it)

- **Start small; it all degrades gracefully downward.** Solo is just `UnifiedMind`. Add a `LocalPool` for your own
  cores. Add users/agents as `Principal`s. Add a farm and a distributed bus when you outgrow one machine. Each step
  is additive; nothing you wrote earlier changes.
- **Keep it local until you mean to go public.** Everything binds `127.0.0.1` by default with a token. Public
  exposure is a deliberate step with every guardrail on — invite-gating controls *who's in*, but only the
  voting/verify/sandbox discipline protects an open compute endpoint from a node that returns a plausible-but-wrong
  answer.
- **leCore imports no model library.** LLMs are callables, embeddings are vectors, external tools are commands or
  remote nodes. The engine stays numpy/stdlib; a `NullProvider` runs everything with zero external deps.
- **Determinism holds.** Overlays, monoid merge, message `seq`, role-binding, seed-generation, and the
  refine/opponent math are all seeded/pure; only network *timing* varies, and `sender`+`seq` make it observable and
  orderable. Bit-exactness across machines isn't the invariant — decision determinism is, and the repair layer
  (`cleanup`/`fountain`/`verify`) absorbs float wobble far below the noise floor.
- **Adding a new ability is free everywhere.** Whatever leCore can do is a capability `route` finds and
  `getattr(mind, name)(**args)` calls — so a new faculty is instantly usable by every LLM, harness, agent, guest,
  and farm node at once.
