# The installed leCore: coverage sweep (what's in, what ports, what's missing)

*The sweep the question deserved: the engine's full surface measured against BOTH
install planes — the weight plane (install.bat → structures IN the weights) and the
resident plane (galvapack → scaffolding IN the forward pass at serve time). The
resident plane is the "automatically port anything, for the most part" machinery:
`maximal_specs()` builds a declarative resident stack from a mind + partition +
corpus, and `_build_residents` constructs 14 kinds. Your current galvatron uses
only the weight plane. This list is the delta.*

## A. Installed today (weight plane — proven on your box)
exit_calibration, memory_index, nullspace_guard, prepend, registers, router,
self_write, state_track — audit TOTAL 0 problems, harden 6/6.
(+ sidecar passages the moment you re-run install.bat from this tree.)

## B. Ports TODAY via the resident plane — built, tested kinds, just not deployed
Each maps to an engine family; deploying = `galvatron.py --imbue` or
`save_pack`/`load_pack`, then serve via the OpenAI-compatible server or
`HFCompatWrapper` (both already in galvapack):

| Resident kind | Engine family it carries |
|---|---|
| `memory` | taught rows + passages injected at a late layer (notes travel in the manifest) |
| `toolbelt` | callable tools in the forward pass (max_calls budget) |
| `ward` | **vetoes as banned emissions** — logit-level, survives a mind-free load |
| `verifier` | the grounding gate's cousin: passage-grounded verified_generate |
| `ouroboros` | selection + durable-note spill back to a leCore partition |
| `cache` | one generation per unique question, served thereafter |
| `screen`, `leap`, `dreamer`, `oracle`, `hrnn`, `carrier`, `capability`, `corpus` | filtering, k-step lookahead, early-layer steering, prediction, recurrent state, payload carriage, capability cards, searchable corpus |

## C. The list, knocked out (cp83) — statuses

1. **Toolbelt↔engine bridge — DONE.** The belt already carried the router (the
   coverage sweep understated it); the true gap was argument discovery:
   `ToolbeltResident.describe(query)` now returns routed candidates WITH
   signatures and doc heads, and a no-args `invoke` failure names the best
   candidate and its signature. Verified: "estimate memory saturation" →
   `saturation_estimate(margins)`.
2. **Serving-door mismatch — DONE.** `chat.py --pack` loads a galvapack and
   serves with residents LIVE (numpy Galvatron), including in-chat `teach:`.
3. **`--forward-keys` — DONE AND VALIDATED ON THE REAL MINI.** factbake's keys
   were already forward-derived; `tools/unicron_install.py` now exposes
   `install_facts_forward` (and its embedding-key sibling says in its docstring
   when not to use it). Measured this pass: fact recall **0/6 → 6/6**, guards
   **40/40 unchanged**, original untouched. The cp70 kept negative is RESOLVED.
4. **Live teach/veto on a running pack — DONE.** `Galvatron.teach` writes the
   memory resident's database while serving; `Galvatron.veto` extends the ward's
   banned set live; both miss honestly when the resident is absent.
5a. **Drift + confidence guards in the serving path — DONE (cp85).**
   `Galvatron.post_check()` runs the POST probe ids against the model's own
   healthy baseline (`galvatron_profile.npz`, written at install) with a
   RELATIVE trigger (the selfheal lesson); measured: healthy ratio 1.0, a
   noised MLP reads 0.0286 → drifted. `Galvatron.ask()` serves memory answers
   only above a confidence margin, REFUSES the model arm outright while
   drifted (with the ratio in the refusal), and escalates honestly otherwise.
   Live pack `teach` runs the drift sentinel and flags contradictions.
5. **Attribution in the serving path — OPEN (the one survivor).** Anchor:
   wrap the pack's generation with `holographic_runtimerung.RuntimeRung` so
   served answers carry source addresses; follow-up item.
6. **Pack manifests as portfolio citizens — DONE.** `pack_notes_to_portfolio`
   and `portfolio_to_memory_spec` in galvapack; verified round-trip.
7. Unchanged: host-compute faculties serve THROUGH the toolbelt, by design.

## C-prior. The original missing list (for the record)

1. **The toolbelt↔engine bridge (the big one).** The `toolbelt` resident exists;
   the engine's 2,200+ faculties exist; NOTHING connects them. Registering engine
   faculties (signal_scan, solve_maze_grid, market_signal_test, api_use, explain,
   tool_find...) as toolbelt tools would put the entire host-compute engine one
   call away from the model's forward pass. Effort: an adapter that wraps
   `mind.<faculty>` as toolbelt entries with argument schemas.
2. **Chat/serving door mismatch.** `chat.py` loads plain transformers — even for
   galvatron, the resident stack is ABSENT at chat time. The pack server
   (`run_galvatron`) and `HFCompatWrapper` are the doors residents actually walk
   through. Missing: `chat.py --pack` (or pointing chat at the running server) so
   efficacy conversations exercise residents.
3. **`--forward-keys` weight-plane facts** (designed cp70, unbuilt): the model
   *speaking* installed content in plain generation, no residents required.
   Acceptance already written in UNICRON_REMAINING.md.
4. **Teach/veto live loop into a running pack.** Ward carries vetoes declared at
   pack time; missing: runtime teach → memory-resident refresh, and veto → ward
   update, while serving (the chat's `teach:`/`wrong` verbs, pack-side).
5. **Attribution in the serving path.** The runtime rung's logit-lens addresses
   and truncation tiers exist engine-side; the HF wrapper exposes no hooks for
   them yet. Missing: per-answer source addresses on served generations.
6. **Portfolio import/export of a pack's memory.** memory_export/import work on
   partitions; a pack's manifest snapshot is not yet a portfolio citizen.
7. **Not installable by design (say it in print):** anything needing host compute
   or I/O beyond the forward pass — rendering artifacts, live HTTP (`api_use`
   executes host-side; the toolbelt bridge is the honest path), jobs, file
   export. These serve THROUGH residents, never inside weights.

## The one-line summary
The weight plane is done and free; the resident plane is built and idle; the
bridge between the engine's faculties and the resident toolbelt is the largest
missing piece of "installed leCore" — and it is an adapter, not a research
problem.
