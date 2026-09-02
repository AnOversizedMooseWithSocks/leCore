# Swarm audit + above/below sweep

## Resident swarm (per organ group)

- agents_and_reasoning         108 modules, 108 import, 92 with selftests
- caching_and_storage          59 modules, 59 import, 57 with selftests
- io_and_interop               94 modules, 94 import, 91 with selftests
- materials_and_texture        18 modules, 18 import, 18 with selftests
- mesh_and_geometry            111 modules, 111 import, 109 with selftests
- misc                         151 modules, 151 import, 130 with selftests
- rendering                    70 modules, 70 import, 67 with selftests
- sampling_and_signal          58 modules, 58 import, 56 with selftests
- scene_and_pipeline           32 modules, 32 import, 30 with selftests
- semantic_router              9 modules, 9 import, 9 with selftests
- simulation_and_physics       53 modules, 53 import, 47 with selftests
- unified                      29 modules, 29 import, 29 with selftests

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

## Derived matrix (catalog-wide, sweep 133)

Run it through the mind with `mind.above_below()`; on the command line `python3 tools/swarm_audit.py --gate`.

The hand-written matrix above covers 21 capabilities and has since cp67. This one takes its population from the CATALOG, so it grows whenever a sweep registers a capability instead of when somebody remembers to edit a literal.

| measure | count |
|---|---|
| catalog cards carrying a `method=` | 720 |
| distinct doors behind them | 652 |
| L0 engine floor | 626 |
| L1 reachable on the facade class | 626 |
| L2 reachable (in the /tools manifest, so `lecore_invoke` can call it) | 626 |
| L2 PROMOTED to a dedicated MCP tool | 15 |
| L3 PROMOTED to a chat verb | 3 |
| L4 named under tests/ | 501 |
| **genuine gaps** | **13** |
| not meaningful to ask (object methods) | 13 |

PROMOTION IS A CENSUS, NEVER A GAP. `holographic_mcp` hosts `lecore_invoke(name, args)`, which runs any public faculty, so L2 reachability is universal by construction and a dedicated tool is a curation decision. Scoring 637 unpromoted doors as defects would be the bar nobody clears.

### Genuine gaps

- `add_caustics` (unreachable) -- module-level function: importable in-process, NOT callable over /invoke
- `aniso_render` (unreachable) -- module-level function: importable in-process, NOT callable over /invoke
- `chain_transport` (unreachable) -- module-level function: importable in-process, NOT callable over /invoke
- `compress_arrays` (unreachable) -- module-level function: importable in-process, NOT callable over /invoke
- `evaluate_candidates` (unreachable) -- module-level function: importable in-process, NOT callable over /invoke
- `export_all` (unreachable) -- module-level function: importable in-process, NOT callable over /invoke
- `from_components` (unreachable) -- module-level function: importable in-process, NOT callable over /invoke
- `merge_drift` (unreachable) -- module-level function: importable in-process, NOT callable over /invoke
- `near_duplicates` (unreachable) -- module-level function: importable in-process, NOT callable over /invoke
- `route_question` (unreachable) -- module-level function: importable in-process, NOT callable over /invoke
- `similar_to` (unreachable) -- module-level function: importable in-process, NOT callable over /invoke
- `splat_denoise` (unreachable) -- module-level function: importable in-process, NOT callable over /invoke
- `write_multichannel` (unreachable) -- module-level function: importable in-process, NOT callable over /invoke

### Not meaningful to ask (recorded so the judgement is on the record)

- `active_workspace` -- object method -- reachable via an object the mind returns, which is a deliberate pattern
- `add_velocity` -- object method -- reachable via an object the mind returns, which is a deliberate pattern
- `as_atom` -- object method -- reachable via an object the mind returns, which is a deliberate pattern
- `close_mailbox` -- object method -- reachable via an object the mind returns, which is a deliberate pattern
- `cool_all` -- object method -- reachable via an object the mind returns, which is a deliberate pattern
- `create` -- object method -- reachable via an object the mind returns, which is a deliberate pattern
- `disable_cold_storage` -- object method -- reachable via an object the mind returns, which is a deliberate pattern
- `new_workspace` -- object method -- reachable via an object the mind returns, which is a deliberate pattern
- `not_null` -- object method -- reachable via an object the mind returns, which is a deliberate pattern
- `query_fuzzy` -- object method -- reachable via an object the mind returns, which is a deliberate pattern
- `redo_stack` -- object method -- reachable via an object the mind returns, which is a deliberate pattern
- `resolve_reference` -- object method -- reachable via an object the mind returns, which is a deliberate pattern
- `run_view` -- object method -- reachable via an object the mind returns, which is a deliberate pattern

