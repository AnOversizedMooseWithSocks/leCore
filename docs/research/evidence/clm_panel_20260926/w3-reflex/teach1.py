import json
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/9ee53278-d85f-54b0-979e-17c930fe54e2/scratchpad")
from inv import inv

T = [
    ("how accurate is find_capability at routing catalog aliases it has never seen",
     "E0.2 baseline (w3-reflex, 2026-09-26, scratch swarm/w3-reflex/e02_bench.py): 3,954 held-out aliases (hash rule "
     "int(sha256(alias)[:2],16)%2==1 over the 8,090 unique aliases of holographic_skills._catalog()), each alias string "
     "ABLATED from every card, scored on UnifiedMind(dim=256,seed=0)._capability_catalog() with 3,952 cards: top-1 0.402, "
     "top-5 0.638, top-8 0.690, median rank 2, 10.2% share no content word with any card (unrankable). With the aliases "
     "left in the index it is a lookup: top-1 1.000. Same protocol on default_catalog (893 cards): top-1 0.437, top-5 0.659."),
    ("how accurate is route_tiered answer tier on held-out catalog aliases at full catalog size",
     "E0.2 (w3-reflex, 2026-09-26): on 3,954 hash-held-out aliases ablated from a 3,952-card catalog, route_tiered "
     "(z_answer -0.1, z_refuse -0.5, k=5) answers 37.1% at 0.604 accuracy (22.4% of all queries answered right), menus "
     "39.5% holding the truth 61.7%, refuses 23.4%. On default_catalog (893 cards) same protocol: answers 38.5% at 0.694, "
     "menu recall 0.673, refuses 32.9%. The sweep-176 figure 0.833 was 1 alias ablated per card on 3x150 sampled cards "
     "of the small catalog; this half-ablation protocol on that catalog gives 0.694, and growth 893 to 3,952 cards takes "
     "it to 0.604 (-9 points). The answer-tier thresholds do not "
     "transfer with catalog growth -- a loud negative for acting on route_tiered answers unverified."),
    ("where does route_tiered spend its time",
     "Measured (w3-reflex, 2026-09-26, 3,952-card catalog, in-process, warm nulls): route_tiered mean 16.7 ms (p50 13.5, "
     "p95 30.8, max 345 ms when a new query token count builds its 64-query null); of that families() is 15.2 ms per call "
     "because Catalog.route_tiered calls self.families() on every call (holographic_catalog.py:497); route_or_abstain warm "
     "2.2 ms; find_scored 1.85 ms; _score_all 1.55 ms mean; find_capability 1.72 ms (memo miss); one bake 95 ms. On 893 "
     "cards families() is 3.9 ms. Caching families() per catalog hash would cut route_tiered roughly 4x with bit-identical "
     "results. The scorer is already cached (baked haystacks), so a CLM-style candidate cache buys nothing here."),
    ("are family siblings the right hard negatives for capability routing",
     "Measured no (w3-reflex E0.2, 2026-09-26): on 3,954 ablated held-out aliases only 13.5% of top-1 errors on the "
     "3,952-card catalog put a card of the true card's family first; restricted to errors where BOTH families resolve "
     "(618 of 2,366 errors; families() resolves 1,509 of 3,952 cards) it is 50.8% (49.5% on the 893-card catalog). The "
     "card that actually outranked the truth (the nearest wrong prototype, LVQ's rule) is the hard negative by "
     "definition and covers 100% of errors; family siblings are at best a supplement."),
    ("how long does a serve call take on the leCore service",
     "Measured (w3-reflex, 2026-09-26, 20 hash-held-out catalog aliases, service at 127.0.0.1:8080, 516 taught rows, "
     "438 meaning rows, load avg 3.4 on 2 cores from other swarm workers): serve round trip mean 6.1 ms, p50 3.6, p95 "
     "15.2; all 20 escalated (serve never consults the catalog router). No-op verb reflex_stats p50 0.5 ms; "
     "find_capability p50 2.3 ms, mean 3.3."),
    ("how many capabilities are in the leCore catalog",
     "Counted 2026-09-26 (w3-reflex): UnifiedMind._capability_catalog() and the live service hold 3,952 cards (10,759 "
     "alias occurrences, 9,509 unique); holographic_skills._catalog() (default + modules, no mind faculties) holds 1,681 "
     "cards with 8,090 unique aliases (78 aliases sit on more than one card); default_catalog() holds 893. The '887 "
     "capabilities' in AGENTS.md and BACKLOG_contrastive.md E0.2/E3.4 is stale."),
]
for q, a in T:
    r = inv("teach", query=q, answer=a)
    print(q, "->", json.dumps(r)[:200])
