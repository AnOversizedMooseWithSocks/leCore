"""CLM -- bundled plugin: Contrastive-LM's CLM-8B as a MODEL-END TIER and a STATE ENCODER (backlog E6.2, E6.3).

WHAT CLM IS (read from its repository and model card on 2026-09-26; sources at the bottom)
-----------------------------------------------------------------------------------------
A two-tower "System One" scorer: a STATE tower and an ACTION (candidate) tower, trained with bidirectional InfoNCE,
answering typed questions (Noul / Choice / Score, the same three shapes as JEV and leCore's typed decisions) by
cosine between the projected state and each projected candidate. It does not generate text. Its parts:

  * the ENCODER is Qwen3-8B, served SEPARATELY as an OpenAI-compatible embeddings endpoint (the reference setup is
    `vllm serve Qwen/Qwen3-8B --runner pooling --port 8090`), last-token pooled, L2-normalised, 4096-d;
  * the HEADS are a 75 MB checkpoint (CLM_v0.1-8B.pt: state_head and action_head, 4096 -> 512, plus logit_scale),
    loaded by the `contrastive-lm` package (`import clm`); score = exp(logit_scale) * cos(state_head(s),
    action_head(c));
  * `clm-serve` (FastAPI, default port 8700) wraps both:
        POST /v1/systemone  {state, model, questions: {id: {type, instructions, criteria}}, temperature}
                            -> {model, answers: {id: {...}}, usage}
             choice answer: {choice, confidence, probabilities: {option: p}}
             noul answer:   {noul: p_true}
             score answer:  {score: expected level index, confidence, legend, probabilities}
        POST /v1/rank       {context, question, answers: [...], model, temperature}
                            -> {model, ranked: [{rank, candidate, prob}]}   (best first)
        GET  /v1/models     -> {models: [...]}
        GET  /health        -> {ok, embedder, models, cache}
    It is NOT OpenAI-compatible (only its encoder server is). Auth is an Authorization header (CLM_API_KEY).
    There is NO state-vector endpoint: the projected state lives inside Engine.answer (see E6.3 below).

ASSUMED, NOT VERIFIED BYTE-FOR-BYTE -- stated loudly because the whole contract test rests on it. The shapes above
were read through a fetch-and-summarise tool, not a raw clone (the session could not clone the repository).
Where a detail was not visible, this plugin ASSUMES, and the stub in tests/test_clm_plugin.py speaks exactly this
assumed dialect:
  * probabilities for a Score answer are either a list of L floats or a dict keyed by level text or str(index);
  * probabilities are softmax outputs, so they sum to 1 within 0.01 (JSON rounding allowed);
  * the Authorization header is "Bearer <key>";
  * the in-process Engine accepts any object with `embed(texts) -> (ndarray, tokens)` as its `embedder`
    (Engine(embedder=...) is in its signature; the duck type is our reading of embedder.py).
If a live server disagrees, the strict validation below REFUSES its replies with a named reason -- it cannot
silently mis-read them. Fix the mapping here, never loosen the validation.

WHAT THIS PLUGIN ADDS (verbs on the mind)
-----------------------------------------
  clm_status()                      backend, device, reachable, why not -- the honest preflight, never raises
  clm_systemone(state, questions)   leCore typed questions -> CLM -> one typed RECORD per question
  clm_rank(state, candidates)       free-form candidates -> CLM's scores, validated, as one record
  clm_escalator()                   a callable for m.typed(..., escalate=...) / systemone_decide(escalate=...)
  clm_embedder(space="raw")         a text -> vector callable for m.set_embedder(fn) (E6.3)

THE REPLY IS ONE MORE TYPED VERDICT, NOT AN ORACLE (the backlog's rule for E6.2)
-------------------------------------------------------------------------------
Every record has leCore's shape: value, ranked, margin, p_correct, p_null, set, via, id, evidence. `p_correct` is
NEVER CLM's own probability. It is None until OUR per-door calibrator (m.door_calibrator("clm:<question>")) has
seen enough reported outcomes, then it is our isotonic estimate of P(CLM right | CLM's margin). CLM's own numbers
travel as evidence (evidence.model_confidence), the same way decide_or_escalate refuses to launder a model end's
judgment through our calibrator. Every record is in the mind's decision ledger (via "model_end", meta.tier "clm"),
so m.decision_outcome(id, truth) (1) calibrates the CLM door and (2) runs reflex_learn -- leOS's self-extending
instruction: one confirmed CLM answer teaches the reflex to answer the next similar state without CLM.

STRICT VALIDATION (the seam is where type guarantees die; this is SystemOne._validate_escalated's discipline):
an answer that is not one of the options, probabilities over the wrong keys or not summing to 1, a choice that is
not the argmax of its own probabilities, a non-finite or out-of-range number, a missing answer, a non-JSON body,
a candidate list that drops, duplicates or invents a candidate -- each is REFUSED with the reason, never repaired.

TWO BACKENDS, ONE INTERFACE; THE OWNER'S RULE IS THAT CPU MUST WORK
------------------------------------------------------------------
  "http"  talk to a running clm-serve (config url, or env LECORE_CLM_URL). stdlib urllib only -- nothing to
          install, CPU-only clients are fine, the GPU lives wherever the server runs. Hard per-call timeout
          (config timeout, default 10 s; CLM's own client defaults to 300 s, far too long for a reflex arc).
  "local" run CLM in-process: the `contrastive-lm` package's Engine with its heads on device "cuda" when torch
          sees one, else "cpu" (config device="auto"|"cpu"|"cuda"). The encoder is EITHER an embeddings server
          (config emb_url / env LECORE_CLM_EMB_URL -- the reference setup) OR, with no emb_url, Qwen3-8B loaded
          in-process with transformers (float32 on CPU, bfloat16 on CUDA; config dtype overrides).
          CPU IS SUPPORTED AND IT IS SLOW -- ESTIMATES, NOT MEASUREMENTS (no torch on the box this was written
          on, and no GPU): Qwen3-8B has ~8.2B parameters, so float32 weights are ~33 GB of RAM (bfloat16 ~16 GB);
          one forward pass costs ~2 x 6.95e9 FLOPs per token, ~4e11 FLOPs for a 30-token state, i.e. roughly 2-4 s
          per NEW state on a 16-core AVX-512 server at an effective 100-200 GFLOP/s, and it cannot load at all on
          a 2 GB box. On a CUDA GPU in bfloat16 a forward is memory-bound (~16 GB of weights read) -- order
          10 ms on an A100-class card. Candidate (action) embeddings are cached by text in the in-process
          embedder, so a revisited candidate set costs only the state forward. A local call cannot be
          interrupted, so `timeout` does not apply to it: use the http backend when a hard deadline matters.
  "auto"  (default) http when a URL is configured, else local.

PLUGIN["requires"] names the LOCAL backend's packages (torch, transformers, clm). On a box without them the plugin
still loads, plugin_list() says available=False with the install hint, and the http backend still works -- it
needs nothing. clm_status() is the authoritative "can I use CLM here", per backend.

E6.3 -- CLM'S STATE ENCODER BEHIND set_embedder
------------------------------------------------
clm_embedder(space="raw") returns text -> Qwen3-8B embedding (4096-d, from the embeddings server over http, or
in-process); space="state" additionally applies CLM's state head (512-d; needs torch + contrastive-lm locally,
since clm-serve exposes no vector endpoint). Pass it to m.set_embedder(fn): set_embedder's own verify gate
decides. THE HONEST PREDICTION, pinned by the tests with a stub: CLM's space is not the shipped routing index's
space (nomic, 128-d), so the gate REFUSES it (dimension 4096 or 512 != 128 is refused before the space probe
even runs); a 128-d hash projection -- the stub's embedding -- reaches the space probe and fails it at chance
(chance = 5/509 = 0.0098 against a 0.30 bar). A stub that serves the index's own rows passes, which proves the
plugin's transport is not what the gate is refusing. CLM's encoder becomes useful to set_embedder only with an
index built in CLM's space -- that is new work, not this item.

No secret is ever returned: the API key goes into one request header and nowhere else (not status, not
evidence, not errors). Tests use the fake key "Hunter2-FAKE-9c1d".

Sources: https://github.com/Contrastive-LM/CLM (README; src/clm/{server,engine,embedder,heads,client}.py) ·
https://huggingface.co/Contrastive-LM/CLM-v0.1-8B · https://pypi.org/project/contrastive-lm/
"""
import base64
import importlib.util
import ipaddress
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from holographic.agents_and_reasoning.holographic_systemone import SchemaError, validate_questions

PLUGIN = {
    "name": "clm",
    "version": "0.1",
    "does": ("Contrastive-LM's CLM-8B as a model-end tier: typed Choice/Noul/Score decisions and candidate ranking "
             "over HTTP (clm-serve, no dependencies) or in-process (torch on CUDA or CPU), validated strictly, "
             "recorded for decision_outcome and calibrated by leCore's own gate; plus CLM's encoder for "
             "set_embedder. available=False only means the LOCAL backend's packages are missing."),
    "requires": ("torch", "transformers", "clm"),
    "install": ("http backend: nothing -- run clm-serve and set LECORE_CLM_URL; local backend: "
                "pip install contrastive-lm transformers torch"),
}

# The in-process backend's packages. Checked with find_spec -- NEVER imported to answer "are they here".
LOCAL_REQUIRES = ("torch", "transformers", "clm")

# A reply larger than this is refused before json.loads sees it: a typed answer is a few hundred bytes, and an
# unbounded read is how a misconfigured URL (pointing at a file server) eats the process's memory.
MAX_REPLY_BYTES = 8 * 1024 * 1024

# Probabilities from a softmax sum to 1; JSON rounding (4-6 decimals over up to ~100 options) stays far inside this.
PROB_SUM_TOL = 0.01

# Generic Score levels when a leCore score question has no anchors (CLM's Score needs a list of level texts).
GENERIC_LEVELS = ("very low", "low", "medium", "high", "very high")


class CLMUnavailable(SchemaError):
    """The model end could not be asked (unreachable, timed out, HTTP error, backend packages missing).
    A SchemaError on purpose: decide_or_escalate catches SchemaError, retries once and then REFUSES, so a dead
    CLM server becomes an honest 'refused' typed answer instead of an exception inside m.typed()."""


class CLMReplyError(SchemaError):
    """CLM answered, and the answer broke the schema (not an option, bad probabilities, missing answer, ...)."""


# ---------------------------------------------------------------------------------------------------------------
# configuration: config dict > environment > default. Read once per register() call (one mind).
# ---------------------------------------------------------------------------------------------------------------
def _resolve_config(config):
    """-> (public config dict, api key or None). Precedence: config dict > environment > default.

    AN EXPLICIT CONFIG ERROR RAISES; A BAD ENVIRONMENT VARIABLE DOES NOT. Discovery calls register() inside
    UnifiedMind.__init__ with no try/except, and every default mind discovers this plugin -- so a raise here
    would turn a typo in LECORE_CLM_TIMEOUT into "no mind can be built on this machine". A bad env value falls
    back to the default and is listed in cfg["config_warnings"], which clm_status() shows."""
    cfg = dict(config or {})
    env = os.environ
    warnings_ = []

    def pick(key, *env_names, default=None):
        if cfg.get(key) is not None:
            return cfg[key], "config:" + key
        for e in env_names:
            if env.get(e):
                return env[e], e
        return default, None

    def number(kind, key, *env_names, default):
        v, src = pick(key, *env_names, default=default)
        try:
            return kind(v)
        except (TypeError, ValueError):
            if src and src.startswith("config:"):
                raise ValueError("clm config %s=%r is not a %s" % (key, v, kind.__name__))
            warnings_.append("%s=%r is not a %s; using %r" % (src, v, kind.__name__, default))
            return kind(default)

    def choice(key, allowed, *env_names, default):
        v, src = pick(key, *env_names, default=default)
        if str(v) in allowed:
            return str(v)
        if src and src.startswith("config:"):
            raise ValueError("clm %s must be one of %s, got %r" % (key, allowed, v))
        warnings_.append("%s=%r is not one of %s; using %r" % (src, v, allowed, default))
        return default

    url = pick("url", "LECORE_CLM_URL")[0]
    out = {
        "url": str(url).rstrip("/") if url else None,
        "emb_url": pick("emb_url", "LECORE_CLM_EMB_URL", "CLM_EMB_URL")[0],
        "emb_model": pick("emb_model", "CLM_EMB_MODEL", default="qwen3-8b")[0],
        "model": pick("model", default="clm-latest")[0],
        "timeout": number(float, "timeout", "LECORE_CLM_TIMEOUT", default=10.0),
        "device": choice("device", ("auto", "cpu", "cuda"), "CLM_DEVICE", default="auto"),
        "dtype": pick("dtype")[0],
        "encoder": pick("encoder", default="Qwen/Qwen3-8B")[0],
        "checkpoint": pick("checkpoint", "CLM_CKPT")[0],
        "temperature": number(float, "temperature", default=1.0),
        "max_tokens": number(int, "max_tokens", default=2048),
        "cache": number(int, "cache", default=4096),
    }
    backend = choice("backend", ("auto", "http", "local"), "LECORE_CLM_BACKEND", default="auto")
    out["backend"] = ("http" if out["url"] else "local") if backend == "auto" else backend
    out["config_warnings"] = warnings_
    # the key is kept OUT of the dict anything prints; see _Transport
    key = pick("api_key", "LECORE_CLM_API_KEY", "CLM_API_KEY")[0]
    return out, key


def _missing_local():
    """Which LOCAL_REQUIRES do not import here (find_spec only -- nothing is imported).

    THE NAME TRAP, found by running this file's own selftest: `python3 holographic/plugins/clm.py` puts
    holographic/plugins/ first on sys.path, so find_spec("clm") finds THIS FILE and reported the real
    contrastive-lm package as installed. A spec whose origin is this file is therefore counted as missing.
    (PluginHost's own `missing` uses plain find_spec and can be fooled the same way in that one invocation;
    discovery imports bundled plugins as holographic.plugins.clm, where the trap does not arise.)"""
    here = os.path.abspath(__file__)
    miss = []
    for m in LOCAL_REQUIRES:
        try:
            spec = importlib.util.find_spec(m)
        except (ImportError, ValueError):
            spec = None
        if spec is None or (spec.origin and os.path.abspath(spec.origin) == here):
            miss.append(m)
    return miss


# ---------------------------------------------------------------------------------------------------------------
# HTTP transport (stdlib)
# ---------------------------------------------------------------------------------------------------------------
def _is_loopback(url):
    host = urllib.parse.urlsplit(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class _Transport:
    """JSON over HTTP with a hard socket timeout. Loopback URLs bypass any configured proxy explicitly (a global
    http_proxy with no no_proxy would otherwise send a local CLM server's traffic to the proxy); every other URL
    uses urllib's normal proxy handling, as an operator would expect."""

    def __init__(self, api_key=None):
        self._key = api_key or None               # held here only; never copied into any returned structure
        self._direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._default = urllib.request.build_opener()

    def call(self, method, url, body=None, timeout=10.0):
        """-> (parsed JSON, {"latency_ms", "server_latency_ms"}). Raises CLMUnavailable (could not ask) or
        CLMReplyError (answered with something that is not JSON / too large)."""
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self._key:
            headers["Authorization"] = "Bearer " + self._key
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        opener = self._direct if _is_loopback(url) else self._default
        path = urllib.parse.urlsplit(url).path or "/"
        t0 = time.perf_counter()
        try:
            with opener.open(req, timeout=float(timeout)) as r:
                raw = r.read(MAX_REPLY_BYTES + 1)
                server_ms = r.headers.get("X-CLM-Latency-Ms")
        except urllib.error.HTTPError as e:
            raise CLMUnavailable("CLM answered HTTP %d on %s" % (e.code, path))
        except (socket.timeout, TimeoutError):
            raise CLMUnavailable("CLM timed out after %.1fs on %s" % (float(timeout), path))
        except urllib.error.URLError as e:
            if isinstance(e.reason, (socket.timeout, TimeoutError)):
                raise CLMUnavailable("CLM timed out after %.1fs on %s" % (float(timeout), path))
            raise CLMUnavailable("CLM unreachable at %s: %s" % (path, e.reason))
        except (ConnectionError, OSError) as e:
            raise CLMUnavailable("CLM unreachable at %s: %s" % (path, e))
        if len(raw) > MAX_REPLY_BYTES:
            raise CLMReplyError("CLM reply on %s is larger than %d bytes -- refused unread" % (path, MAX_REPLY_BYTES))
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise CLMReplyError("CLM reply on %s is not JSON" % path)
        lat = {"latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)}
        try:
            lat["server_latency_ms"] = float(server_ms) if server_ms is not None else None
        except ValueError:
            lat["server_latency_ms"] = None
        return obj, lat


# ---------------------------------------------------------------------------------------------------------------
# the in-process backend (torch / transformers / contrastive-lm, all imported lazily INSIDE methods)
# ---------------------------------------------------------------------------------------------------------------
class _TransformersEmbedder:
    """Qwen3-8B loaded in-process, duck-typing contrastive-lm's Embedder for Engine(embedder=...):
    embed(texts) -> (float32 array (n, 4096), L2-normalised, tokens spent).

    Last-token pooling + L2 normalisation, which is what the reference vLLM `--runner pooling` setup produces for
    a causal LM (CLM's embedder.py comments name last-token pooling). EQUIVALENCE WITH vLLM IS AN ASSUMPTION: if
    CLM's answers from this path disagree with the server's, compare one embedding against the server's
    /v1/embeddings before trusting either. An LRU cache by text keeps a revisited candidate set cheap (CLM's own
    embedder caches the same way)."""

    def __init__(self, model_id, device, dtype, max_tokens=2048, cache=4096):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from collections import OrderedDict
        self.torch = torch
        self.device = device
        self.max_tokens = int(max_tokens)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:                  # batching needs one; the pooled token ignores padding anyway
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        self._cache = OrderedDict()
        self._cache_max = int(cache)

    def healthy(self):
        return True

    def embed(self, texts):
        torch = self.torch
        texts = [str(t) for t in texts]
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        got = {t: self._cache[t] for t in dict.fromkeys(texts) if t in self._cache}
        spent = 0
        for a in range(0, len(missing), 8):
            blk = missing[a:a + 8]
            enc = self.tok(blk, padding=True, truncation=True, max_length=self.max_tokens, return_tensors="pt").to(self.device)
            with torch.no_grad():
                hid = self.model(**enc).last_hidden_state
            last = enc["attention_mask"].sum(dim=1) - 1                 # the last REAL token of each row
            pooled = hid[torch.arange(hid.shape[0], device=hid.device), last].float().cpu().numpy()
            pooled /= np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12
            spent += int(enc["attention_mask"].sum().item())
            for t, v in zip(blk, pooled):
                got[t] = v
                self._cache[t] = v
                if len(self._cache) > self._cache_max:
                    self._cache.popitem(last=False)
        # answer from THIS call's vectors: a call with more new texts than the cache holds would otherwise have
        # evicted some of its own results before reading them back
        return np.stack([got[t] for t in texts]).astype(np.float32), spent


class _LocalBackend:
    """contrastive-lm's Engine in this process. Nothing is imported until the first real call; a missing package
    is reported by status() and raised as CLMUnavailable by every call, with the install line."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.engine = None
        self.device = None
        self.dtype = None
        self.encoder_kind = None

    def status(self):
        miss = _missing_local()
        if miss:
            return {"reachable": False, "device": None,
                    "why": "local backend needs %s (not installed; nothing was imported). Install: "
                           "pip install contrastive-lm transformers torch -- or use the http backend "
                           "(set LECORE_CLM_URL to a running clm-serve)." % ", ".join(miss)}
        dev = self._resolve_device()
        return {"reachable": self.engine is not None, "device": dev,
                "why": None if self.engine is not None else "packages present; the model loads on first call",
                "cpu_note": self._cpu_note(dev)}

    def _resolve_device(self):
        import torch
        want = self.cfg["device"]
        if want == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if want == "cuda" and not torch.cuda.is_available():
            raise CLMUnavailable("clm device='cuda' was requested but torch sees no CUDA device")
        return want

    @staticmethod
    def _cpu_note(dev):
        if dev != "cpu":
            return None
        return ("CPU: an in-process Qwen3-8B encoder in float32 needs ~33 GB RAM and roughly 2-4 s per new state "
                "on a 16-core server (ESTIMATE from ~4e11 FLOPs per 30-token forward, not measured here); "
                "with emb_url set, only the 75 MB heads run on CPU and the encoder cost is the server's.")

    def _load(self):
        if self.engine is not None:
            return self.engine
        miss = _missing_local()
        if miss:
            raise CLMUnavailable(self.status()["why"])
        import warnings
        import torch
        from clm import Engine
        self.device = self._resolve_device()
        dt = self.cfg["dtype"]
        self.dtype = getattr(torch, dt) if isinstance(dt, str) else (dt or (torch.float32 if self.device == "cpu"
                                                                           else torch.bfloat16))
        if self.cfg["emb_url"]:
            # the reference setup: the encoder is a server, only the heads are local
            self.encoder_kind = "server:" + self.cfg["emb_url"]
            self.engine = Engine(emb_url=self.cfg["emb_url"], emb_model=self.cfg["emb_model"], device=self.device,
                                 checkpoint=self.cfg["checkpoint"])
        else:
            if self.device == "cpu":
                warnings.warn("clm local backend: loading %s IN-PROCESS ON CPU. %s" % (
                    self.cfg["encoder"], self._cpu_note("cpu")), RuntimeWarning, stacklevel=3)
            self.encoder_kind = "in-process:" + self.cfg["encoder"]
            emb = _TransformersEmbedder(self.cfg["encoder"], self.device, self.dtype,
                                        max_tokens=self.cfg["max_tokens"], cache=self.cfg["cache"])
            self.engine = Engine(embedder=emb, device=self.device, checkpoint=self.cfg["checkpoint"])
        return self.engine

    @staticmethod
    def _plain(obj):
        """Engine may hand back answer objects rather than dicts; normalise to plain JSON-like data."""
        if isinstance(obj, dict):
            return {k: _LocalBackend._plain(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_LocalBackend._plain(v) for v in obj]
        if hasattr(obj, "to_dict"):
            return _LocalBackend._plain(obj.to_dict())
        if hasattr(obj, "__dict__") and not isinstance(obj, type):
            return _LocalBackend._plain(vars(obj))
        return obj

    def systemone(self, state, questions):
        t0 = time.perf_counter()
        out = self._load().answer(state, questions, self.cfg["model"], self.cfg["temperature"])
        return self._plain(out), {"latency_ms": round((time.perf_counter() - t0) * 1000.0, 2), "server_latency_ms": None}

    def rank(self, state, candidates, instructions):
        t0 = time.perf_counter()
        ranked = self._load().rank(state, list(candidates), instructions, self.cfg["model"], self.cfg["temperature"])
        return ({"model": self.cfg["model"], "ranked": self._plain(ranked)},
                {"latency_ms": round((time.perf_counter() - t0) * 1000.0, 2), "server_latency_ms": None})

    def embed_raw(self, text):
        eng = self._load()
        vecs, _ = eng.embedder.embed([str(text)])
        return np.asarray(vecs, dtype=np.float64)[0]

    def project_state(self, raw):
        """Apply CLM's state head (4096 -> 512). Engine.heads[model].ensure().project_states is the path Engine
        itself uses (engine.py); a tensor result is brought back to numpy."""
        eng = self._load()
        head = eng.heads[self.cfg["model"]].ensure() if self.cfg["model"] in eng.heads else \
            next(iter(eng.heads.values())).ensure()
        z = head.project_states(np.asarray(raw, dtype=np.float32)[None, :])
        if hasattr(z, "detach"):
            z = z.detach().float().cpu().numpy()
        return np.asarray(z, dtype=np.float64).reshape(-1)


# ---------------------------------------------------------------------------------------------------------------
# schema mapping: leCore question spec <-> CLM question, and strict reply validation
# ---------------------------------------------------------------------------------------------------------------
def _instruction(name, instructions):
    if isinstance(instructions, dict) and instructions.get(name):
        return str(instructions[name])
    if isinstance(instructions, str) and instructions:
        return instructions
    return str(name).replace("_", " ")


def to_clm_question(name, spec, instructions=None):
    """One VALIDATED leCore spec -> (CLM question dict, mapping info for the reply).
    choice: criteria {option: description}; the description is the option's first examples when given, else the
            option name with underscores as spaces (CLM scores against the criteria TEXT, so a bare token loses
            information the examples carry).
    noul:   criteria {"true": ..., "false": ...} from the yes/no examples when given, else omitted.
    score:  criteria = level texts: the anchors sorted by value when given (each level maps back to its anchor
            value), else five generic levels spread evenly over [min, max]."""
    instr = _instruction(name, instructions)
    ex = spec.get("examples") or {}
    if spec["type"] == "choice":
        crit = {o: ("; ".join(ex[o][:3]) if ex.get(o) else o.replace("_", " ")) for o in spec["options"]}
        return {"type": "choice", "instructions": instr, "criteria": crit}, {"options": list(spec["options"])}
    if spec["type"] == "noul":
        q = {"type": "noul", "instructions": instr}
        if ex.get("yes") or ex.get("no"):
            q["criteria"] = {"true": "; ".join(ex.get("yes", [])[:3]) or "yes",
                             "false": "; ".join(ex.get("no", [])[:3]) or "no"}
        return q, {}
    anchors = sorted(spec.get("anchors") or [], key=lambda a: (a[1], a[0]))
    if anchors:
        levels = [a[0] for a in anchors]
        values = [float(a[1]) for a in anchors]
    else:
        levels = list(GENERIC_LEVELS)
        values = list(np.linspace(spec["min"], spec["max"], len(levels)))
    return ({"type": "score", "instructions": instr, "criteria": levels},
            {"levels": levels, "values": values, "min": spec["min"], "max": spec["max"]})


def _finite_prob(x, what):
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise CLMReplyError("%s is not a number: %r" % (what, x))
    if not np.isfinite(v) or v < -1e-9 or v > 1 + 1e-9:
        raise CLMReplyError("%s is not a probability in [0, 1]: %r" % (what, x))
    return min(max(v, 0.0), 1.0)


def _ranked(labels, probs):
    """[(label, p)] best first; ties keep the given (insertion) order -- the engine's one tie rule."""
    order = sorted(range(len(labels)), key=lambda i: -probs[i])
    return [(labels[i], float(probs[i])) for i in order]


def validate_answer(spec, info, raw):
    """Hold one CLM answer to the schema. -> (value, ranked, margin, model_confidence). Raises CLMReplyError with
    the reason on anything malformed; never repairs a reply."""
    if not isinstance(raw, dict):
        raise CLMReplyError("answer is not an object: %r" % (raw,))
    if spec["type"] == "choice":
        opts = info["options"]
        choice = raw.get("choice")
        v = choice.strip() if isinstance(choice, str) else choice
        if v not in opts:
            raise CLMReplyError("CLM choice %r is not one of %s" % (choice, opts))
        pr = raw.get("probabilities")
        if not isinstance(pr, dict) or set(pr) != set(opts):
            raise CLMReplyError("probabilities must be keyed by exactly the options %s, got %r"
                                % (opts, sorted(pr) if isinstance(pr, dict) else pr))
        probs = [_finite_prob(pr[o], "probability of %r" % o) for o in opts]
        if abs(sum(probs) - 1.0) > PROB_SUM_TOL:
            raise CLMReplyError("probabilities sum to %.4f, not 1" % sum(probs))
        if probs[opts.index(v)] < max(probs) - 1e-9:
            raise CLMReplyError("CLM chose %r but gave %r a higher probability -- an inconsistent reply"
                                % (v, opts[int(np.argmax(probs))]))
        rk = _ranked(opts, probs)
        conf = raw.get("confidence")
        conf = _finite_prob(conf, "confidence") if conf is not None else None
        margin = rk[0][1] - rk[1][1]
        return v, rk, float(margin), conf
    if spec["type"] == "noul":
        p = _finite_prob(raw.get("noul"), "noul (P(true))")
        # our noul options are ["yes", "no"] in that order, so an exact 0.5 goes to "yes" (insertion-order tie)
        rk = _ranked(["yes", "no"], [p, 1.0 - p])
        return bool(p >= 0.5), rk, float(abs(2.0 * p - 1.0)), p
    # score
    levels, values = info["levels"], info["values"]
    L = len(levels)
    try:
        e = float(raw.get("score"))
    except (TypeError, ValueError):
        raise CLMReplyError("score %r is not numeric" % (raw.get("score"),))
    if not np.isfinite(e) or e < -1e-9 or e > (L - 1) + 1e-9:
        raise CLMReplyError("score %r is outside the level range [0, %d]" % (raw.get("score"), L - 1))
    pr = raw.get("probabilities")
    if isinstance(pr, (list, tuple)):
        if len(pr) != L:
            raise CLMReplyError("score probabilities have %d entries for %d levels" % (len(pr), L))
        probs = [_finite_prob(x, "level probability") for x in pr]
    elif isinstance(pr, dict):
        by = {}
        for k, x in pr.items():
            if k in levels:
                by[levels.index(k)] = x
            elif str(k).isdigit() and int(k) < L:
                by[int(k)] = x
            else:
                raise CLMReplyError("score probability key %r is not a level" % (k,))
        if len(by) != L:
            raise CLMReplyError("score probabilities cover %d of %d levels" % (len(by), L))
        probs = [_finite_prob(by[i], "level probability") for i in range(L)]
    else:
        raise CLMReplyError("score answer has no probabilities")
    if abs(sum(probs) - 1.0) > PROB_SUM_TOL:
        raise CLMReplyError("score probabilities sum to %.4f, not 1" % sum(probs))
    value = float(np.clip(np.interp(e, np.arange(L), values), info["min"], info["max"]))
    rk = _ranked(levels, probs)
    conf = raw.get("confidence")
    conf = _finite_prob(conf, "confidence") if conf is not None else None
    return value, rk, float(rk[0][1] - rk[1][1]) if L > 1 else 1.0, conf


def validate_rank(candidates, reply):
    """Hold a /v1/rank reply to the request: every sent candidate exactly once, nothing invented, probabilities in
    [0, 1] summing to 1, ranks 1..n best first. -> ranked [(candidate, prob)]."""
    rows = reply.get("ranked") if isinstance(reply, dict) else None
    if not isinstance(rows, list):
        raise CLMReplyError("rank reply has no 'ranked' list")
    if len(rows) != len(candidates):
        raise CLMReplyError("rank reply has %d rows for %d candidates" % (len(rows), len(candidates)))
    seen, out = set(), []
    for j, r in enumerate(rows):
        if not isinstance(r, dict):
            raise CLMReplyError("rank row %d is not an object" % j)
        c = r.get("candidate")
        if c not in candidates:
            raise CLMReplyError("rank reply names %r, which was not a candidate" % (c,))
        if c in seen:
            raise CLMReplyError("rank reply lists %r twice" % (c,))
        seen.add(c)
        if r.get("rank") is not None and r.get("rank") != j + 1:
            raise CLMReplyError("rank row %d carries rank %r" % (j, r.get("rank")))
        out.append((c, _finite_prob(r.get("prob"), "probability of %r" % c)))
    if any(out[j][1] < out[j + 1][1] - 1e-9 for j in range(len(out) - 1)):
        raise CLMReplyError("rank reply is not ordered best first")
    if abs(sum(p for _, p in out) - 1.0) > PROB_SUM_TOL:
        raise CLMReplyError("rank probabilities sum to %.4f, not 1" % sum(p for _, p in out))
    return out


def _decode_embedding(e):
    """An OpenAI-style embedding: a list of floats, or base64 float32 when encoding_format='base64'."""
    if isinstance(e, str):
        try:
            return np.frombuffer(base64.b64decode(e), dtype=np.float32).astype(np.float64)
        except (ValueError, TypeError):
            raise CLMReplyError("embedding is a string but not base64 float32")
    if isinstance(e, list):
        try:
            return np.asarray(e, dtype=np.float64)
        except (TypeError, ValueError):
            raise CLMReplyError("embedding list is not numeric")
    raise CLMReplyError("embedding has an unknown shape: %r" % type(e))


# ---------------------------------------------------------------------------------------------------------------
# the plugin object: one per mind (register), verbs close over it
# ---------------------------------------------------------------------------------------------------------------
class _CLM:
    """One per mind: the resolved config, the two backends, and the verbs' bodies.

    PICKLABLE, BECAUSE A MIND WITH PLUGINS MUST BE (tests/test_plugin.py pins it: the unified app caches a taught
    mind by pickling it). What is dropped on the way out, and why:
      * the API key -- a pickled mind is a file on disk, and a secret is never persisted. A key that came from the
        ENVIRONMENT is re-read from the environment on load; a key passed in config must be passed again
        (status() says so under auth_note);
      * the urllib openers (rebuilt) and any loaded local Engine (reloads lazily on the next call)."""

    def __init__(self, mind, config):
        self.mind = mind
        self._config_had_key = bool((config or {}).get("api_key"))
        self.cfg, key = _resolve_config(config)
        self.http = _Transport(key)
        self.local = _LocalBackend(self.cfg)
        self._key_dropped = False

    def __getstate__(self):
        return {"mind": self.mind, "cfg": dict(self.cfg), "config_had_key": self._config_had_key}

    def __setstate__(self, st):
        self.mind = st["mind"]
        self.cfg = st["cfg"]
        self._config_had_key = st["config_had_key"]
        key = None if self._config_had_key else _resolve_config({})[1]
        self._key_dropped = self._config_had_key
        self.http = _Transport(key)
        self.local = _LocalBackend(self.cfg)

    # ---- the two backends behind one call shape -------------------------------------------------------------
    def _ask_systemone(self, state, clm_questions, timeout):
        if self.cfg["backend"] == "http":
            if not self.cfg["url"]:
                raise CLMUnavailable("http backend has no URL: pass config url or set LECORE_CLM_URL")
            body = {"state": state, "model": self.cfg["model"], "questions": clm_questions,
                    "temperature": self.cfg["temperature"]}
            return self.http.call("POST", self.cfg["url"] + "/v1/systemone", body, timeout)
        return self.local.systemone(state, clm_questions)

    def _ask_rank(self, state, candidates, instructions, timeout):
        if self.cfg["backend"] == "http":
            if not self.cfg["url"]:
                raise CLMUnavailable("http backend has no URL: pass config url or set LECORE_CLM_URL")
            body = {"context": state, "question": instructions, "answers": list(candidates),
                    "model": self.cfg["model"], "temperature": self.cfg["temperature"]}
            return self.http.call("POST", self.cfg["url"] + "/v1/rank", body, timeout)
        return self.local.rank(state, candidates, instructions)

    # ---- records --------------------------------------------------------------------------------------------
    def _record(self, state, question, options, value, ranked, margin, conf, lat, calibrate=True):
        """One typed record in leCore's shape, added to the mind's decision ledger (via model_end, tier clm) with
        a hook that feeds THIS question's CLM calibrator when the outcome is reported."""
        door = "clm:" + str(question)
        cal = self.mind.door_calibrator(door)
        p_correct = cal.p_correct(margin) if calibrate else None
        evidence = {"backend": self.cfg["backend"], "model": self.cfg["model"], "model_confidence": conf,
                    "latency_ms": lat.get("latency_ms"), "server_latency_ms": lat.get("server_latency_ms")}
        from holographic.agents_and_reasoning.holographic_decisionrecord import DecisionRecord
        # The ledger question is the DOOR ("clm:<question>"), not the bare question name. MEASURED COLLISION, found
        # by the escalation contract test: a record id is sha256(state, question, options, answer, via), and an
        # escalated typed answer is filed by systemone_decide with the SAME five fields (via model_end, the same
        # options, CLM's answer) -- so both records got one id and the typed record, added second, REPLACED this
        # record's calibration hook. Filing under the door name keeps the two records (and their hooks) apart.
        meta = {"tier": "clm", "door": door, "question": question, "evidence": evidence}
        if calibrate:
            # THE OUTCOME HOOK IS A RESTART-PROOF SPEC (learning-loop audit, 2026-09-26). It was a live closure, so an
            # outcome reported after a restart calibrated nothing -- MEASURED: clm:<question> calibrator pairs 0 of 1
            # (tools/audit_learning_loop.py). The mind's 'calibrate' kind (p29 _decision_hook) is the same act from
            # the record: door_calibrator(door).observe(margin, outcome_matches(answer, outcome)) -- outcome_matches
            # maps a yes/no answer onto "yes"/"no" and, stricter than the old closure, never counts an outcome that
            # is neither (e.g. "maybe") as confirming a False answer. Score questions are not calibrated (as in
            # SystemOne), so they carry no hook.
            meta["hook"] = {"kind": "calibrate", "door": door, "score": None if margin is None else float(margin),
                            "answer": value}
        rec = DecisionRecord(state, door, list(options), value, "model_end", margin=margin, p=p_correct, meta=meta)
        self.mind.decision_ledger().add(rec)
        return {"value": value, "ranked": ranked, "margin": margin, "margin_gap": margin,
                "p_correct": p_correct, "p_null": None, "set": None, "calibrated": p_correct is not None,
                "via": "model_end", "tier": "clm", "id": rec.id, "evidence": evidence, "verify": None,
                "abstained": False, "door": door}

    @staticmethod
    def _refused(qtype, err):
        """A refusal record. `refusal` says which kind: 'unavailable' (CLM could not be asked) or 'reply' (it
        answered off-schema) -- the escalator re-raises the matching exception class from it."""
        kind = "unavailable" if isinstance(err, CLMUnavailable) else "reply"
        return {"type": qtype, "value": None, "ranked": [], "margin": None, "margin_gap": None, "p_correct": None,
                "p_null": None, "set": None, "calibrated": False, "via": "refused", "tier": "clm", "id": None,
                "evidence": None, "verify": None, "abstained": True, "why": str(err), "refusal": kind}

    # ---- verbs ----------------------------------------------------------------------------------------------
    def systemone(self, state, questions, instructions=None, timeout=None):
        qs = validate_questions(questions)                  # OUR schema, strict: a caller error raises here
        clm_q, info = {}, {}
        for name, spec in qs.items():
            clm_q[name], info[name] = to_clm_question(name, spec, instructions)
        try:
            reply, lat = self._ask_systemone(state, clm_q, self.cfg["timeout"] if timeout is None else timeout)
        except (CLMUnavailable, CLMReplyError) as e:
            return {n: self._refused(qs[n]["type"], e) for n in qs}
        answers = reply.get("answers") if isinstance(reply, dict) else None
        if not isinstance(answers, dict):
            return {n: self._refused(qs[n]["type"], CLMReplyError("CLM reply has no 'answers' object")) for n in qs}
        out = {}
        for name, spec in qs.items():
            if name not in answers:
                out[name] = self._refused(spec["type"], CLMReplyError("CLM reply has no answer for %r" % name))
                continue
            try:
                value, rk, margin, conf = validate_answer(spec, info[name], answers[name])
            except CLMReplyError as e:
                out[name] = self._refused(spec["type"], e)
                continue
            opts = [n for n, _ in rk]
            rec = self._record(state, name, opts, value, rk, margin, conf, lat, calibrate=spec["type"] != "score")
            rec["type"] = spec["type"]
            out[name] = rec
        return out

    def rank(self, state, candidates, instructions=None, timeout=None):
        cands = list(candidates)
        if len(cands) < 2 or len(set(cands)) != len(cands) or not all(isinstance(c, str) and c for c in cands):
            raise ValueError("clm_rank needs >= 2 unique non-empty candidate strings")
        try:
            reply, lat = self._ask_rank(state, cands, instructions, self.cfg["timeout"] if timeout is None else timeout)
            rk = validate_rank(cands, reply)
        except (CLMUnavailable, CLMReplyError) as e:
            return self._refused("rank", e)
        rec = self._record(state, "rank", cands, rk[0][0], rk, float(rk[0][1] - rk[1][1]), None, lat)
        rec["type"] = "rank"
        return rec

    def status(self):
        out = {"backend": self.cfg["backend"], "model": self.cfg["model"], "url": self.cfg["url"],
               "config_warnings": list(self.cfg.get("config_warnings", [])),
               "emb_url": self.cfg["emb_url"], "timeout": self.cfg["timeout"], "device": None,
               "reachable": False, "why": None, "local_missing": _missing_local(), "auth": bool(self.http._key)}
        if self._key_dropped:
            out["auth_note"] = ("the api_key passed in config was not persisted when this mind was pickled (secrets "
                                "are never written to disk); pass it again or set LECORE_CLM_API_KEY")
        if self.cfg["backend"] == "http":
            out["device"] = "remote (the server's; a CPU-only client is fine)"
            if not self.cfg["url"]:
                out["why"] = "http backend has no URL: pass config url or set LECORE_CLM_URL"
                return out
            try:
                health, lat = self.http.call("GET", self.cfg["url"] + "/health", None, self.cfg["timeout"])
                out["reachable"] = bool(isinstance(health, dict) and health.get("ok", True))
                out["health"] = health if isinstance(health, dict) else None
                out["latency_ms"] = lat["latency_ms"]
                if not out["reachable"]:
                    out["why"] = "clm-serve /health says not ok"
            except (CLMUnavailable, CLMReplyError) as e:
                out["why"] = str(e)
            return out
        try:
            st = self.local.status()
        except CLMUnavailable as e:
            st = {"reachable": False, "device": None, "why": str(e)}
        out.update(st)
        return out

    def escalator(self, instructions=None, examples=None):
        return _Escalator(self, instructions, examples)

    def embedder(self, space="raw", normalize=True):
        if space not in ("raw", "state"):
            raise ValueError("clm_embedder space must be 'raw' (the encoder's 4096-d) or 'state' (CLM's state head)")
        if space == "state" or self.cfg["backend"] == "local":
            miss = _missing_local()
            if miss:
                raise CLMUnavailable(
                    "clm_embedder(space=%r, backend=%r) needs %s in this process (clm-serve has no vector "
                    "endpoint, so the state head must run locally). Install: pip install contrastive-lm "
                    "transformers torch" % (space, self.cfg["backend"], ", ".join(miss)))
        if self.cfg["backend"] == "http" and not self.cfg["emb_url"]:
            raise CLMUnavailable("clm_embedder over http needs the ENCODER's embeddings URL (config emb_url or "
                                 "LECORE_CLM_EMB_URL, e.g. http://127.0.0.1:8090/v1/embeddings) -- clm-serve "
                                 "itself exposes no vectors")
        return _Embedder(self, space, normalize)


class _Escalator:
    """A callable for m.typed(..., escalate=...) / systemone_decide(escalate=...). decide_or_escalate hands it a
    payload {question, spec, state, evidence, why, prompt[, error]} and validates what it returns against the same
    schema (_validate_escalated), retrying once and then refusing. This asks CLM the one question, keeps the full
    CLM record (with its OWN ledger id) in .last, and returns the raw value.

    TWO IDS, ON PURPOSE: the typed answer's id (a["id"]) trains SystemOne and the reflex when you report it; the
    CLM record's id (escalator.last["id"]) calibrates the CLM tier. Report both with the same truth to get both.
    (They were ONE id at first -- same state, question, options, answer and via hash to the same record -- and the
    typed record's hook silently replaced the CLM calibration hook; the CLM record is now filed under its door
    name "clm:<question>". Ledger.add now combines hooks per id (wave 2), but the door-named question stays: it keeps
    the two records -- and their ids -- apart, and the CLM record's hook is a restart-proof 'calibrate' spec, so an
    outcome reported after a restart still calibrates the CLM door.)
    A CLM that is down or answers off-schema raises a SchemaError subclass, which decide_or_escalate turns into
    an honest via='refused' answer -- never an exception out of m.typed()."""

    def __init__(self, clm, instructions=None, examples=None):
        self.clm = clm
        self.instructions = instructions
        self.examples = {k: list(v) for k, v in (examples or {}).items() if v}
        self.last = None
        self.calls = 0

    def __call__(self, payload):
        self.calls += 1
        name = payload["question"]
        spec = payload["spec"]
        # rebuild a spec our validator accepts: the payload's spec is SystemOne's normalised form (noul carries
        # options, examples are stripped), which validate_questions would reject for noul
        # EXAMPLES: decide_or_escalate strips them from the payload, so without the escalator's own `examples`
        # CLM scores the bare option NAMES -- found by the contract test, where the stub tied 0.5/0.5 on
        # 'billing' vs 'shipping'. Pass the same examples dict you pass to m.typed.
        if spec["type"] == "choice":
            clean = {"type": "choice", "options": list(spec["options"])}
            ex = {o: self.examples[o] for o in clean["options"] if o in self.examples}
            if ex:
                clean["examples"] = ex
        elif spec["type"] == "noul":
            clean = {"type": "noul"}
            ex = {k: self.examples[k] for k in ("yes", "no") if k in self.examples}
            if ex:
                clean["examples"] = ex
        else:
            clean = {"type": "score", "min": spec["min"], "max": spec["max"], "anchors": spec.get("anchors", [])}
        rec = self.clm.systemone(payload["state"], {name: clean}, instructions=self.instructions)[name]
        self.last = rec
        if rec["via"] == "refused":
            raise (CLMUnavailable if rec["refusal"] == "unavailable" else CLMReplyError)(rec["why"])
        return rec["value"]


class _Embedder:
    """text -> vector for m.set_embedder(fn). Validates every vector (finite, non-empty, one fixed dimension) and
    caches by text. Raises on failure -- set_embedder's probe counts a raise as a failed probe and route_semantic
    counts it as a miss, which are exactly the honest outcomes."""

    def __init__(self, clm, space, normalize):
        from collections import OrderedDict
        self.clm = clm
        self.space = space
        self.normalize = bool(normalize)
        self.dim = None
        self._cache = OrderedDict()

    def _raw(self, text):
        cfg = self.clm.cfg
        if cfg["backend"] == "http":
            body = {"model": cfg["emb_model"], "input": [str(text)], "encoding_format": "float"}
            if cfg["max_tokens"]:
                body["truncate_prompt_tokens"] = cfg["max_tokens"]
            reply, _ = self.clm.http.call("POST", cfg["emb_url"], body, cfg["timeout"])
            data = reply.get("data") if isinstance(reply, dict) else None
            if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
                raise CLMReplyError("embeddings reply must carry exactly one data row for one input")
            return _decode_embedding(data[0].get("embedding"))
        return self.clm.local.embed_raw(text)

    def __call__(self, text):
        key = str(text)
        v = self._cache.get(key)
        if v is not None:
            self._cache.move_to_end(key)
            return v
        v = self._raw(key)
        if self.space == "state":
            v = self.clm.local.project_state(v)
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        if v.size == 0 or not np.all(np.isfinite(v)):
            raise CLMReplyError("embedding is empty or not finite")
        if self.dim is None:
            self.dim = int(v.size)
        elif v.size != self.dim:
            raise CLMReplyError("embedding dimension changed from %d to %d between calls" % (self.dim, v.size))
        if self.normalize:
            n = float(np.linalg.norm(v))
            v = v / n if n > 0 else v
        self._cache[key] = v
        if len(self._cache) > self.clm.cfg["cache"]:
            self._cache.popitem(last=False)
        return v


# ---------------------------------------------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------------------------------------------
def _v_status(clm):
    """Report whether CLM can be asked here: {backend, device, reachable, why, local_missing, ...}. Never raises."""
    return clm.status()


def _v_systemone(clm, state, questions, instructions=None, timeout=None):
    """Ask CLM typed questions (leCore schema: choice / noul / score); returns {question: record} with value,
    ranked, margin, p_correct (OUR calibration, None until calibrated), id, evidence -- or via='refused' + why
    when CLM is unreachable or its answer breaks the schema. Report outcomes with decision_outcome(id)."""
    return clm.systemone(state, questions, instructions=instructions, timeout=timeout)


def _v_rank(clm, state, candidates, instructions=None, timeout=None):
    """Score free-form candidates for a state with CLM; returns one record (value = the top candidate, ranked =
    every candidate with CLM's probability, margin, p_correct from our calibrator, id) or via='refused'."""
    return clm.rank(state, candidates, instructions=instructions, timeout=timeout)


def _v_escalator(clm, instructions=None, examples=None):
    """Return a callable to pass as m.typed(..., escalate=fn): when the substrate abstains, CLM answers under the
    schema; the full CLM record (with its own id) is kept in fn.last. Pass the same `examples` dict you give
    typed -- the escalation payload carries none, and CLM scores against the criteria text."""
    return clm.escalator(instructions=instructions, examples=examples)


def _v_embedder(clm, space="raw", normalize=True):
    """Return a text -> vector callable from CLM's encoder (space='raw', 4096-d) or state head ('state', 512-d)
    to pass to m.set_embedder(fn); set_embedder's own verify gate decides whether it is admitted."""
    return clm.embedder(space=space, normalize=normalize)


def _verb(fn, clm):
    """A picklable verb: functools.partial over a MODULE-LEVEL function (pickled by reference) with this mind's
    _CLM bound first. A closure would read the same but cannot be pickled -- and a mind carrying an unpicklable
    verb broke tests/test_plugin.py::test_mind_with_plugins_pickles, found on the first run. The partial keeps
    the real signature (inspect drops the bound argument) for GET /tools, and carries the function's docstring."""
    import functools
    v = functools.partial(fn, clm)
    v.__doc__ = fn.__doc__
    return v


def register(mind, config=None):
    """Bind the CLM verbs to one mind. Cheap on purpose (every default mind discovers this plugin): it reads the
    config and environment, builds no connection, imports nothing optional."""
    clm = _CLM(mind, config)
    clm_status = _verb(_v_status, clm)
    clm_systemone = _verb(_v_systemone, clm)
    clm_rank = _verb(_v_rank, clm)
    clm_escalator = _verb(_v_escalator, clm)
    clm_embedder = _verb(_v_embedder, clm)

    return [
        {"name": "clm_status", "fn": clm_status,
         "does": "Report whether the CLM model end can be asked here (backend, device, reachable, why not).",
         "example": "mind.clm_status()",
         "aliases": ("is the clm server up", "clm model end status", "check contrastive lm backend",
                     "can i use clm here", "clm cpu or gpu")},
        {"name": "clm_systemone", "fn": clm_systemone,
         "does": ("Ask Contrastive-LM's CLM typed Choice/Noul/Score questions; returns validated typed records "
                  "calibrated by leCore's own gate and learnable through decision_outcome."),
         "example": "mind.clm_systemone('courier lost my parcel', {'topic': {'type': 'choice', 'options': ['billing', 'shipping']}})",
         "aliases": ("ask clm a typed question", "contrastive language model decision", "system one model end",
                     "score actions with clm", "clm choice question", "external system one scorer")},
        {"name": "clm_rank", "fn": clm_rank,
         "does": "Rank free-form candidate actions for a state with CLM's two-tower scorer; validated, recorded.",
         "example": "mind.clm_rank('user asks to undo the last edit', ['revert commit', 'open settings', 'run tests'])",
         "aliases": ("rank candidate actions with clm", "score candidates with a contrastive model",
                     "clm rank answers", "which action fits this state clm", "two tower ranking")},
        {"name": "clm_escalator", "fn": clm_escalator,
         "does": "Build the escalate= callable that makes CLM the model-end tier behind typed decisions.",
         "example": "fn = mind.clm_escalator(examples={'shipping': ['parcel lost']}); mind.typed('courier lost my parcel', ['billing', 'shipping'], escalate=fn)",
         "aliases": ("use clm as the model end", "escalate typed decisions to clm", "clm escalation tier",
                     "clm fallback when the substrate abstains")},
        {"name": "clm_embedder", "fn": clm_embedder,
         "does": "Build a text-to-vector callable from CLM's encoder for set_embedder (its verify gate decides).",
         "example": "fn = mind.clm_embedder(); mind.set_embedder(fn)",
         "aliases": ("use clm as an embedder", "clm state encoder", "qwen3 embeddings for routing",
                     "plug clm into set_embedder")},
    ]


def _selftest():
    """Contract without a server and without torch: the plugin binds on a slim mind, reports the local backend
    unavailable cleanly when its packages are missing, imports none of them, and validates replies strictly."""
    import sys
    import lecore
    before = set(sys.modules)
    m = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    m._plugin_load(os.path.abspath(__file__), config={"backend": "local"})
    st = m.clm_status()
    assert st["reachable"] is False or not st["local_missing"]
    assert not ({"torch", "transformers", "clm"} & (set(sys.modules) - before)), "an optional package was imported"
    spec = validate_questions({"q": {"type": "choice", "options": ["a", "b"]}})["q"]
    info = {"options": ["a", "b"]}
    v, rk, mg, _ = validate_answer(spec, info, {"choice": "a", "probabilities": {"a": 0.7, "b": 0.3}})
    assert v == "a" and abs(mg - 0.4) < 1e-12 and rk[0] == ("a", 0.7)
    for bad in ({"choice": "c", "probabilities": {"a": 0.7, "b": 0.3}},
                {"choice": "a", "probabilities": {"a": 0.2, "b": 0.8}},
                {"choice": "a", "probabilities": {"a": 0.7, "b": 0.7}}):
        try:
            validate_answer(spec, info, bad)
            raise AssertionError("accepted a malformed reply %r" % (bad,))
        except CLMReplyError:
            pass
    print("holographic.plugins.clm selftest OK -- status: %s" % (st.get("why") or "local backend ready"))


if __name__ == "__main__":
    _selftest()
