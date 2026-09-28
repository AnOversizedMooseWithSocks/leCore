# Plugins -- extending leCore without growing the core

A plugin adds verbs to a mind. It is a plain Python module with two things in it: a `PLUGIN`
dict saying who it is, and a `register(mind, config)` function returning the verbs it adds.
Nothing is subclassed, nothing is decorated, nothing is imported from the engine to write one.

Every verb a plugin adds is a first-class faculty on that mind: callable as `mind.verb(...)`,
callable over the wire as `POST /invoke`, advertised by `GET /tools`, listed by `features()`,
and surfaced by `find_capability` through the aliases the plugin declared. That last property is
the engine's definition of existing at all -- a verb that cannot be found does not exist -- and
it is why the loader is strict about descriptions and aliases.

The optional parts of the engine itself are plugins. Everything that needs a dependency outside
the wheel -- the Zig kernels, the SymPy derivations, the Numba JIT renderer, the WGSL runtime,
the CuPy device path, the Lean 4 bridge -- lives in `holographic/plugins/`, binds at construction,
and can be left out.

## Using plugins

```python
import lecore

m = lecore.UnifiedMind()                      # default: every plugin discovery finds
m = lecore.UnifiedMind(plugins=())            # a slim mind: none of them
m = lecore.UnifiedMind(plugins=("zig", "lean4"))   # exactly these, by name

for p in m.plugin_list():                     # the preflight
    print(p["name"], p["available"], p["missing"], p["install"])
```

`plugin_list()` is the honest preflight. A plugin whose dependency is not installed still loads
and its verbs still bind -- each verb fails with a clear message on its own, which is the
fallback that made it a plugin in the first place -- but the record says `available=False`,
names what is `missing`, and gives the `install` command. An app can ask "do I have the Zig
path here" without calling it and catching.

The bundled plugins are named after their pip extras, so the extra and the plugin are one thing
under two names:

| plugin | verbs | needs | install |
|---|---|---|---|
| `jit` | `compiled_sdf_numba`, `render_sdf_fast` | numba, sympy | `pip install leos-core[jit,symbolic]` |
| `symbolic` | `exact_sdf_normal`, `compiled_sdf_normal`, `gradient_cache_symbolic` | sympy | `pip install leos-core[symbolic]` |
| `zig` | `zig_batch_eval`, `zig_regime_map`, `zig_dispatch_policy`, `zig_march_compare`, `validate_kernel` | ziglang | `pip install leos-core[zig]` |
| `wgsl` | `run_wgsl_kernel` | wgpu | `pip install leos-core[wgsl]` |
| `gpu` | `unicron_device` | cupy | `pip install cupy-cuda12x` (match your CUDA) |
| `lean4` | `lean_export`, `lean_verify`, `lean_status`, `lean_fuzz` | a `lean` binary (optional) | `python3 tools/install_lean.py` |
| `clm` | `clm_status`, `clm_systemone`, `clm_rank`, `clm_escalator`, `clm_embedder` | nothing for the http backend; torch, transformers, contrastive-lm for the local one | `pip install contrastive-lm transformers torch` (local only) |

## Where plugins come from

Discovery runs once, at `UnifiedMind` construction, and looks in three places in this order:

1. **Bundled** -- every `*.py` in `holographic/plugins/`. Ships in the wheel.
2. **Folders** -- every `*.py` in each folder named by the `LECORE_PLUGIN_PATH` environment
   variable (separated by `os.pathsep`: `:` on Unix, `;` on Windows). This is the per-app and
   per-user door.
3. **Installed** -- every `lecore.plugins` entry point a pip-installed distribution declares.

Within each source, files load in sorted-name order, so the same install gives the same verb
surface every run. Files whose name starts with `_` are skipped (that is what makes the shipped
template safe to sit in the bundled folder). Two plugins claiming the same name is refused loudly
-- a configuration error you should see, not one the loader quietly resolves.

```sh
export LECORE_PLUGIN_PATH=/opt/myapp/plugins:/home/me/.lecore/plugins
python3 -c "import lecore; print([p['name'] for p in lecore.UnifiedMind().plugin_list()])"
```

## Writing a plugin

Copy the template, drop the underscore, edit:

```sh
cp holographic/plugins/_template.py  /opt/myapp/plugins/voice.py
python3 /opt/myapp/plugins/voice.py          # its own selftest: callable, discoverable, listed
```

The whole contract, from that file:

```python
PLUGIN = {
    "name": "voice",                       # what plugins=("voice",) selects by
    "version": "0.1",
    "does": "Text-to-speech through the host's TTS engine.",
    "requires": ("some_tts_lib",),         # optional: what the verbs need
    "install": "pip install some-tts-lib", # optional: how to get it
}

def voice_say(text, rate=1.0):
    """Speak `text`; returns {'wav_bytes': ...}."""
    import some_tts_lib                    # imported INSIDE the verb, so the plugin loads without it
    ...

def register(mind, config=None):
    cfg = config or {}
    return [
        {"name": "voice_say", "fn": voice_say,
         "does": "Speak text aloud and return the audio.",
         "example": "mind.voice_say('hello')",
         "aliases": ("text to speech", "say this out loud", "read this to me", "tts")},
    ]
```

Things worth knowing when you write one:

- **Verbs are plain functions, not methods.** `fn` is bound to the mind as-is and does not receive
  `self`. A verb that needs the mind closes over the `mind` argument `register` receives. That is
  deliberate: it keeps a plugin testable without a mind.
- **Import your dependency inside the verb**, not at the top of the module. The plugin then loads
  (and is listed, with `available=False`) even when the dependency is absent, and the verb gives
  its own clear error when called.
- **Write aliases the way a stranger would type them**, not the way you named the function.
  `find_capability` matches on them; five phrasings is a good number. Test them:
  `mind.find_capability("read this to me")` should rank your verb in the top three.
- **`config`** is whatever the operator passed to `_plugin_load(ref, config=...)`; discovered
  plugins receive `None`. Read it with `.get()` and defaults.
- **The selftest at the bottom of the template** loads the file explicitly onto a slim mind, so it
  runs from any folder whether or not discovery would have found it.

To publish a plugin on PyPI, declare the entry point and it is discovered with no configuration:

```toml
[project.entry-points."lecore.plugins"]
voice = "lecore_plugin_voice"          # a module (or package) with PLUGIN + register
```

## What the loader refuses, and why

Every refusal is a `PluginError` with a sentence, and nothing binds -- a plugin whose third verb
is bad leaves the mind exactly as it was.

- **A verb name that already exists on the mind** -- a core faculty or another plugin's verb.
  There is no override flag. Measured reason: an instance attribute silently shadows a bound
  method, so a plugin that redefined `version` would have `GET /tools` advertising the real
  signature while `POST /invoke` ran the plugin's body. A manifest that lies is worse than one
  that is incomplete. Rename your verb.
- **A verb with no `does`.** It could never be found.
- **A verb name starting with `_`**, or not a valid identifier. `/invoke` refuses private names,
  so the verb would exist and be uncallable.
- **An unknown key in a verb dict.** A misspelled `aliases` is a verb nobody can find.
- **A plugin already loaded by that name.** Unload it first.

## Adapting a plugin to your workflow

Per-plugin configuration is declared once at construction and reaches that plugin's
`register(mind, config)`:

```python
import lecore
m = lecore.UnifiedMind(plugins=("lean4",), plugin_config={"lean4": {"timeout": 30}})
```

A plugin you left out, or unloaded, can be added back by **bare name** -- discovery resolves it, so
you never need to know where it lives -- and the construction-time config still applies:

```python
import lecore
m = lecore.UnifiedMind(plugins=())          # slim
m._plugin_load("zig")                       # by name, from wherever discovery found it
m._plugin_unload("zig")
```

A host program that defines its plugin **in its own code** hands the module object straight in --
no file, no environment variable, no entry point. This is the embedding case:

```python
import types, lecore
plug = types.ModuleType("myapp_plugin")
plug.PLUGIN = {"name": "myapp", "version": "1", "does": "My app's verbs."}
plug.register = lambda mind, config=None: [
    {"name": "myapp_ping", "fn": lambda: {"pong": True}, "does": "Liveness check."}]
m = lecore.UnifiedMind(plugins=())
m._plugin_load(plug)
print(m.myapp_ping(), m.plugin_list()[0]["ref"])
```

One rule the loader enforces here: **the file (or entry point) is named after the plugin.**
`PLUGIN["name"]` must equal the filename stem, because `plugins=(...)` selects and `plugin_config`
keys by that name before the module can be imported. A mismatch is refused with a sentence.

## The `clm` plugin: a CLM model end, on CPU or GPU

`holographic/plugins/clm.py` adds [Contrastive-LM's CLM-8B](https://github.com/Contrastive-LM/CLM) as one more
tier of the reflex arc (backlog items E6.2 and E6.3). CLM is a two-tower "System One" scorer. It has a state tower
and a candidate tower, trained with InfoNCE. It answers the same three typed questions leCore does (Choice, Noul,
Score) and ranks free-form candidates. It does not generate text.

**What CLM actually is, as read from its source.**

- The encoder is Qwen3-8B, served separately as an OpenAI-compatible `/v1/embeddings` endpoint. The reference
  setup is vLLM with `--runner pooling` on port 8090. Vectors are 4096-d, last-token pooled and L2-normalised.
- The heads are a 75 MB checkpoint loaded by the `contrastive-lm` package (`import clm`). They project 4096-d to
  512-d, one head per tower.
- `clm-serve` (port 8700) wraps both. It serves `POST /v1/systemone`, `POST /v1/rank`, `GET /v1/models` and
  `GET /health`. It is not OpenAI-compatible, and it exposes no state vectors.

**The contract is partly ASSUMED.** The request and response shapes were read through a fetch-and-summarise tool,
not a clone. Where a detail was not visible, the plugin's docstring lists what it assumes. The stub server in
`tests/test_clm_plugin.py` speaks exactly that dialect. If a live server disagrees, strict validation refuses its
replies with the reason. The fix goes in the mapping, never in the validation.

### Two backends, one interface

| backend | how | needs | device |
|---|---|---|---|
| `http` | talks to a running `clm-serve` with stdlib `urllib` | nothing (`LECORE_CLM_URL` or `url=`) | wherever the server runs; a CPU-only client is fine |
| `local` | runs `clm.Engine` in this process | `pip install contrastive-lm transformers torch` | `device="auto"` picks CUDA when torch sees it, else CPU |
| `auto` (default) | `http` when a URL is configured, else `local` | | |

```python
import lecore
m = lecore.UnifiedMind(plugin_config={"clm": {"url": "http://gpu-host:8700", "timeout": 5}})
m.clm_status()           # {backend, device, reachable, why, local_missing, ...} -- never raises
```

`plugin_list()` reports `available=False` when the **local** backend's packages are missing. The http backend
still works then, because it needs nothing. `clm_status()` is the real answer to "can I use CLM here".

**CPU is supported, and it is slow.** These are ESTIMATES, not measurements. The box this was written on has no
torch and no GPU, so no live number is claimed anywhere.

- **The local backend with an embeddings server** (`emb_url=`). Only the 75 MB heads run in-process, so CPU is
  fine. The encoder's cost is the server's.
- **The local backend with no embeddings server.** Qwen3-8B loads in-process with transformers: float32 on CPU,
  bfloat16 on CUDA, or whatever `dtype=` says.
  - Qwen3-8B has about 8.2B parameters, so float32 weights need about 33 GB of RAM (bfloat16 about 16 GB).
  - A 30-token state costs about 4e11 FLOPs. That is roughly 2 to 4 s per new state on a 16-core server running at
    an effective 100 to 200 GFLOP/s.
  - On a CUDA GPU in bfloat16 the forward pass is memory-bound, about 10 ms on an A100-class card.
  - The plugin warns loudly when it loads the encoder on CPU.
  - Candidate embeddings are cached by text, so a revisited candidate set only pays for the new state.
- **Timeouts.** A local call cannot be interrupted, so `timeout` applies to the http backend only. The default is
  10 s; CLM's own client uses 300 s, which is too long for a reflex arc.

Configuration keys, all optional: `backend`, `url`, `emb_url`, `emb_model`, `model` (default `clm-latest`),
`timeout`, `device`, `dtype`, `encoder` (default `Qwen/Qwen3-8B`), `checkpoint`, `temperature`, `max_tokens`,
`cache`, and `api_key`. The environment equivalents are `LECORE_CLM_URL`, `LECORE_CLM_BACKEND`,
`LECORE_CLM_EMB_URL`, `LECORE_CLM_TIMEOUT` and `LECORE_CLM_API_KEY`. CLM's own `CLM_EMB_URL`, `CLM_DEVICE`,
`CLM_CKPT` and `CLM_API_KEY` are read too. The key only goes into the `Authorization` header. It never appears in
status, evidence or errors.

### The reply is one more typed verdict

`clm_systemone(state, questions)` takes leCore's own question schema and validates it strictly before anything is
sent. It returns one record per question, in the same shape every door returns: `value`, `ranked`, `margin`,
`p_correct`, `p_null`, `set`, `via`, `id` and `evidence`.

- **`p_correct` is never CLM's probability.** It stays `None` until leCore's own per-door calibrator
  (`m.door_calibrator("clm:<question>")`) has seen enough reported outcomes. CLM's numbers travel as
  `evidence.model_confidence`.
- **Every record is in the decision ledger** (`via="model_end"`, filed under `clm:<question>`). So
  `m.decision_outcome(id, truth)` calibrates the CLM door. It also runs `reflex_learn`: one confirmed CLM answer
  teaches the reflex to answer the next similar state without CLM.
- **Strict validation refuses, it never repairs.** An answer outside the options is refused. So are probabilities
  over the wrong keys or not summing to 1, a choice that is not the argmax of its own probabilities, and an
  out-of-range number. So is a missing answer, a non-JSON body, or a rank list that drops, duplicates or invents a
  candidate. Each comes back as `via="refused"` with a `why`. None of them raise.

**As the escalation tier of `typed`:**

```python
ex = {"billing": ["charged twice on my card"], "shipping": ["courier lost my parcel"]}
fn = m.clm_escalator(examples=ex)        # pass the SAME examples: the escalation payload carries none
a = m.typed("the courier lost my parcel", ["billing", "shipping"], examples=ex, escalate=fn)
m.decision_outcome(a["id"], truth)       # trains SystemOne + the reflex
m.decision_outcome(fn.last["id"], truth) # calibrates the CLM tier
```

When the substrate abstains, CLM answers under the schema. A CLM that is down or answers off-schema becomes an
honest `via="refused"` answer, not an exception out of `typed`.

The contract tests found two things this way:

- **Without `examples=`, CLM scores the bare option names.** `decide_or_escalate` strips examples from the
  payload.
- **The CLM record and the escalated typed record first hashed to the same ledger id.** They share state,
  question, options, answer and via. The typed record's hook then replaced CLM's calibration hook. Filing CLM
  records under `clm:<question>` fixed that. One id carrying both hooks would need `Ledger.add` to compose hooks.

### CLM's encoder behind `set_embedder` (E6.3)

`clm_embedder(space="raw")` returns a text-to-vector callable that uses the encoder: 4096-d, over http or
in-process. `space="state"` also applies CLM's state head (512-d). That needs torch and contrastive-lm locally,
because `clm-serve` exposes no vectors. Pass the callable to `m.set_embedder(fn)`; the seam's own verify gate
decides.

The honest outcome, pinned by the tests: **the gate refuses CLM for the shipped routing index.** That index lives
in nomic's 128-d space. CLM's 4096-d or 512-d vectors are refused at the dimension check. A 128-d deterministic
hash projection gets past the dimension check and then fails the space probe at chance: about 5/509 self-recall,
against a 0.30 bar. A stub serving the index's own rows passes the gate through the same plugin, so the plugin's
transport is not what fails. CLM's encoder becomes useful to `set_embedder` only with an index built in CLM's
space, which is separate work.

## Building ON the holographic framework

The template shows the contract. `holographic/plugins/_example_tags.py` shows the point: a
capability the core does not have, built from the engine's own algebra in about sixty lines. It is a
one-vector tag memory -- "this item carries these tags; which items carry this tag?" -- and every
primitive it uses is the right one for a reason its docstring gives:

| primitive | what it does here |
|---|---|
| `derived_atom(seed, name, dim)` | every item and tag is a vector that is a pure function of its name -- nothing stored to reproduce it |
| `bind(tag, item)` | one fact as one vector that resembles neither operand |
| a plain sum | the whole memory is one vector; adding a fact is adding a vector |
| `unbind(trace, tag)` | "what carries this tag?" as a noisy estimate |
| `nearest` / `cosine` | snap the estimate to real items with a score, and **abstain** below a floor |

```python
import lecore
m = lecore.UnifiedMind(dim=256, plugins=())
m._plugin_load("holographic.plugins._example_tags")
m.tags_remember("kettle", ["kitchen", "electric"])
m.tags_remember("sofa", ["lounge"])
print(m.tags_query("kitchen")["ranked"][0]["name"])   # kettle
print(m.tags_query("garage")["abstained"])            # True -- refusal is an answer
```

The primitives live in `holographic/agents_and_reasoning/holographic_ai.py` (Plate's HRR, in one
readable file); `find_capability("bind two vectors")` will take you there. Two lessons the example
carries on the record, both found by measurement while writing it: renormalising on every add
(`bundle()` does) makes an exponentially-decaying memory that forgets early facts -- a bug that
looked like a capacity limit until recall failed to improve with dimension; and the real capacity
limit of a one-vector memory at small `dim` is measured in the selftest rather than asserted.

A plugin that needs scale should not build its own index: call `mind.learn` / `mind.recall` and let
the engine's memory (with its calibrated abstention) hold the data.

## Connecting from outside

Every plugin verb is reachable over both wires with no extra work:

- **HTTP** (`lecore-service`): `GET /tools` advertises it, `POST /invoke {"name": ..., "args": {...}}`
  runs it. The loader is refused over the wire.
- **MCP** (`lecore-mcp`): `lecore_map` reports which plugins this node carries and whether each
  dependency is installed; `lecore_find` surfaces plugin verbs by their aliases; `lecore_invoke`
  calls them. Same gate.

Configure the mind in-process, then serve it: `Service(mind=m)` / `MCPServer(mind=m)`.

## Loading and unloading by hand

Discovery covers the normal case. For the rest, two operator calls on the mind:

```python
m._plugin_load("/opt/myapp/plugins/voice.py", config={"rate": 1.2})   # file path or dotted module
m._plugin_unload("voice")
```

**They are underscore-private on purpose.** The HTTP service exposes every public mind method to
`POST /invoke`, blocking only leading underscores. A public loader would let any agent that can
POST execute arbitrary code in the service process; a public unloader would let one agent strip
capabilities another relies on. Configure the mind in-process, then serve it. The read-only half
-- `plugin_list()` and `plugin_manifest()` -- is public, because knowing what is loaded is exactly
what an agent legitimately needs.

The same boundary is why folder scanning is fine: the folders are the bundled directory and
whatever the operator put in `LECORE_PLUGIN_PATH` before the process started, and scanning runs
at construction. An agent cannot change either. What is forbidden is a public faculty that loads
from a caller-supplied path or adds a folder to the scan list at runtime -- that would be handing
`import` to `/invoke`.

## What a plugin is not

- **Not sandboxed.** A plugin runs with full process privilege, like any imported module. The gate
  is who may load, not what a loaded plugin may do. Claiming a sandbox we cannot enforce would be
  worse than stating the limit.
- **Not process-global.** A plugin binds to one mind instance, matching the engine's per-mind
  catalog. A global registry would put a verb in every mind's `/tools` while it was callable in
  only one.
- **Not revocable once referenced.** `_plugin_unload` removes the name, which is what discovery
  and `/invoke` go through; a reference a caller already took (`fn = m.voice_say`) keeps working.
  Python offers no way around that, so it is stated rather than hidden.
- **Not a size win by itself.** Moving the six bundled plugins out took 16 of ~2,410 verbs off the
  default surface -- under 1%. What the architecture buys is the pattern and the switch: the next
  optional capability has a home that is not core, and `plugins=()` is a real slim mind.

## The rule for what belongs in a plugin

*Does this need something outside the wheel?* -- not *does the word appear in it*.

`lean_verify` shells out to a Lean binary: plugin. The 1,300-line pure-stdlib Horn prover it sits
on, which other faculties build on: core. `render_sdf_fast` only produces Numba kernels: plugin.
`signed_distance_field`, which has a Numba fast path and a NumPy fallback: core -- it works without
the dependency, which is the opposite of a plugin. `engine_status`, which reports on which
accelerators are present: core, because it reports *on* plugins.

## Where the code is

| what | where |
|---|---|
| the host: load / list / unload / manifest, the collision gate | `holographic/io_and_interop/holographic_plugin.py` |
| discovery: bundled folder, `LECORE_PLUGIN_PATH`, entry points | `holographic/plugins/__init__.py` |
| the mind's door: `plugins=`, `plugin_list`, `plugin_manifest`, `_plugin_load`, `_plugin_unload` | `holographic/unified/holographic_unified_p26_plugins.py` |
| the template | `holographic/plugins/_template.py` |
| the bundled plugins | `holographic/plugins/{jit,symbolic,zig,wgsl,gpu,lean4,clm}.py` |
| the clm plugin's contract tests (stub server) | `tests/test_clm_plugin.py` |
| the migration tool that moved them | `tools/migrate_to_plugins.py` |
| tests | `tests/test_plugin.py` |
