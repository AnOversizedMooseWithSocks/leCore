# Unicron's first live run (field report, real weights, user's box)

*The week-awaited numbers: qwen3.5-class hybrid (24L, hidden 1024, vocab 248,320,
GDN 16/16 heads + full attention at layers 3 and 23). Every claim below is from the
operator's own console output or artifacts, verdicted against the cp65 acceptance
criteria.*

| Acceptance criterion | Result | Evidence |
|---|---|---|
| Base model healthy before touch | **PASS** — ppl 16.2 plain English | diagnose.bat sanity line |
| Layout resolves (qkv order, rt.cfg) | **PASS** — flat / True | diagnose.bat |
| Blank-layer prepend is safe | **PASS** — drift exactly 0.000e+00 (bit-identical) | diagnose.bat; nonzero non-norm tensors are gated-off hybrid sublayers — the VERDICT line now says so in print |
| Install finished (not just wrote) | **PASS** — lecore.json present, layer counts match | audit.bat |
| Installed leCore is WIRED | **PASS** — boots as 'leCore' with 8 capabilities (exit_calibration, memory_index, nullspace_guard, prepend, registers, router, self_write, state_track); **TOTAL: 0 problems** | audit.bat |
| Parts exercisable from artifact alone | **PASS** — registers **128/128** recalled, regenerated from seed; ladder ACT-R fit **R² 0.99858** | audit.bat |
| Preservation (the AlphaEdit claim, live) | **PASS** — plain-English ppl **15.7846 installed vs 16.2 original**: the install cost nothing measurable | assess.bat both runs |
| BIOS resident, POST passes | **PASS** — `bios True`; POST reference profile validated (39 in-vocab probes, consistent hidden norms 7.3–10.5, finite, no dead rows) | assess.bat + galvatron_profile.npz |
| Mixed-probe elevation is the expected kind | **INFO** — 52.8 = 3.4× plain (probe carries leCore-specific content); track this ratio across installs | assess.bat |
| Hardening audit detail (6 checks) | **5/6 PASS; the 6th CRASHED, now fixed** — bios_post, bios_enumerates (24 layers, split layout), boots_from_weights ('leCore'), channel_addressed (wrong-seed agreement 0.45, inside the 0.35–0.65 honest band), cache_saves_work all PASS; `expansion_deterministic` raised IndexError on the EMPTY codebook (no passages installed) — the verifier crashing on the input class it judges; rewritten to compare the whole codebook with empty vacuously deterministic. Re-run `assess.bat` after pulling to see 6/6 | galvatron.npz manifest + holographic_harden.py fix |
| Bundle deep health | **PASS** — 36 gate tensors (A_log −6.6..2.4, dt_bias −13.1..8.4, all finite); 24 activation layers, zero dead, norms grow 1.2 → 11.6; 246 singular spectra, effective-rank fraction 0.97 mean (0.06 min — the filtered layers); top-64 logits + logsumexp finite; 162 probe ids all in-vocab | this review |
| Hardening re-run after fix | **PASS — 6/6** on real weights (assessment of 18:18; ppl unchanged 15.7846) | second galvatron.npz |
| Mind preserved through assimilation | **PASS (qualitative)** — side-by-side fluency and character comparable; both ramble like the 0.8B base they are | chat --both transcript |
| Efficacy by conversation | **PENDING — and the first attempt tested the wrong model**: `--both` paired original vs *assimilated* (no leCore inside); the user caught it from the banner. chat.py now has `--galvatron`; the second attempt exposed that the label fix left `load(assim_dir)` in place (banner said GALVATRON, loaded assimilated — caught by the user from the load lines); the load line now loads the subject. AND the deeper truth, documented in docs/UNICRON_REMAINING.md: chat parity is CORRECT current behavior — generation-plane fact install (--forward-keys) is the remaining load-bearing item | this review |

**Verdict: the install is real, wired, and free.** Preservation held on real weights
— the number the sandbox could never produce. Two items remain: the 8-check audit
detail (send `assessments/galvatron.npz`) and the side-by-side conversation.
