# Swarm audit + above/below sweep

## Resident swarm (per organ group)

- agents_and_reasoning         103 modules, 103 import, 87 with selftests
- caching_and_storage          56 modules, 56 import, 54 with selftests
- io_and_interop               89 modules, 89 import, 86 with selftests
- materials_and_texture        18 modules, 18 import, 18 with selftests
- mesh_and_geometry            111 modules, 111 import, 109 with selftests
- misc                         151 modules, 151 import, 130 with selftests
- rendering                    69 modules, 69 import, 66 with selftests
- sampling_and_signal          58 modules, 58 import, 56 with selftests
- scene_and_pipeline           32 modules, 32 import, 30 with selftests
- semantic_router              9 modules, 9 import, 9 with selftests
- simulation_and_physics       53 modules, 53 import, 47 with selftests
- unified                      21 modules, 21 import, 21 with selftests

## Above/below matrix (L0 engine, L1 facade, L2 hosted, L3 chat, L4 pinned)

- ask/teach/veto                 L0:+ L1:+ L2:+ L3:+ L4:+
- semantic recall                L0:+ L1:+ L2:+ L3:+ L4:+
- saturation estimate            L0:+ L1:+ L2:+ L3:+ L4:+
- drift sentinel / teach_check   L0:+ L1:+ L2:+ L3:+ L4:+
- void explore/mix/propose       L0:+ L1:+ L2:+ L3:+ L4:+
- hypothesis test                L0:+ L1:+ L2:+ L3:+ L4:+
- conjecture record/promote      L0:+ L1:+ L2:+ L3:+ L4:+
- api learn/use                  L0:+ L1:+ L2:+ L3:+ L4:+
- contextual tool find           L0:+ L1:+ L2:+ L3:+ L4:+
- research archive               L0:+ L1:+ L2:+ L3:- L4:+
- scene render                   L0:+ L1:+ L2:- L3:+ L4:+
- procedural texture             L0:+ L1:+ L2:- L3:+ L4:-
- workspaces                     L0:+ L1:+ L2:- L3:+ L4:-
- memory slots/compare           L0:+ L1:+ L2:- L3:+ L4:-
- sessions                       L0:+ L1:+ L2:- L3:+ L4:+
- panel                          L0:+ L1:+ L2:+ L3:- L4:+
- ouroboros selection            L0:+ L1:+ L2:- L3:- L4:-
- grounded answering             L0:+ L1:+ L2:+ L3:+ L4:+
- docs explain                   L0:+ L1:+ L2:- L3:+ L4:+
- memory portfolio               L0:+ L1:+ L2:- L3:+ L4:+
- source attribution             L0:+ L1:+ L2:- L3:+ L4:+

## Deliberate gaps (chosen, with reasons)

- memory portfolio at L2: hosted import/export is an abuse surface; local-runtime feature like memory upload
- memory slots/compare at L2: hosted memory upload is an SSRF/abuse surface; local-runtime feature
- ouroboros selection at L2: selection is inside the reader path, not a callable tool
- ouroboros selection at L3: same: substrate-internal
- panel at L3: panel runs through ask when relevant; no dedicated verb yet (accepted)
- procedural texture at L2: same cost decision as scene render
- research archive at L3: archive building is an operator/dev act; chat consumes via ask
- scene render at L2: hosted raymarching is a cost decision, not a wiring gap
- sessions at L2: hosted sessions are connection-scoped by the server
- source attribution at L2: requires model-directory access; hosted operators enable it server-side on their own runtime
- workspaces at L2: hosted callers are namespaced by the server, not by chat workspaces

## Unintended gaps

- none
