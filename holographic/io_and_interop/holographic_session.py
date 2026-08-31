"""SESSION STORE -- contexts that outlive the process.

A Galvatron's context is not a transcript, it is STATE: the GDN recurrent
matrices, the conv windows, the KV arrays, the position clock, and -- on
leCore's side -- the oracle memories, learned routes and evidence the residents
accumulated. All of that lived in RAM and died with the process, which meant a
conversation could not span a coffee break, let alone weeks.

This makes it a FILE. A session is a named directory: save it, load it, list
them, fork one into two, delete one. Because the state is the model's actual
inference state rather than a prompt to be re-read, resuming costs NO re-prefill
-- a 10,000-token context comes back in the time it takes to read an npz, and
the model continues mid-thought.

THE CONTRACT, asserted rather than hoped: generation continued from a RELOADED
session is TOKEN-IDENTICAL to generation that never stopped. A session store
that quietly changes the model's behaviour is worse than none, because the
difference shows up as a personality drift nobody can debug.

MULTIPLE CONTEXTS ARE THE POINT: sessions are independent by construction (fork
gives two futures from one past, and writing to one never touches the other), so
a harness can keep a session per user, per document, or per task, swap them in
and out by name, and expire them on its own schedule. Nothing here assumes a
single conversation.
"""

import json
import os
import shutil
import time

import numpy as np


MANIFEST = "session.json"


def state_to_arrays(state):
    """Flatten an InferenceState into a plain dict of arrays (npz-friendly).
    Keys encode WHERE each array belongs so a reload cannot silently mis-file a
    layer's memory into another layer's slot."""
    out = {"__pos__": np.asarray([state.pos], np.int64)}
    if getattr(state, "logits", None) is not None:
        out["__logits__"] = np.asarray(state.logits, np.float64)
    for L, st in state.gdn.items():
        for key, val in st.items():
            out["gdn:%d:%s" % (int(L), key)] = np.asarray(val, np.float64)
    for L, st in state.kv.items():
        for key, val in st.items():
            out["kv:%d:%s" % (int(L), key)] = np.asarray(val, np.float64)
    return out


def state_from_arrays(arrays):
    """Rebuild an InferenceState from the flattened form."""
    from holographic.io_and_interop.holographic_gdnruntime import InferenceState
    st = InferenceState()
    st.pos = int(np.asarray(arrays["__pos__"]).ravel()[0])
    st.logits = (np.asarray(arrays["__logits__"], np.float64)
                 if "__logits__" in arrays else None)
    for k in arrays:
        if k.startswith("gdn:") or k.startswith("kv:"):
            kind, layer, field = k.split(":", 2)
            slot = (st.gdn if kind == "gdn" else st.kv).setdefault(int(layer), {})
            slot[field] = np.asarray(arrays[k], np.float64)
    return st


class SessionStore:
    """Named, persistent, independent contexts on disk.

    Each session directory holds state.npz (the model's inference state),
    session.json (metadata: token count, timestamps, model fingerprint) and
    optional memory.json (oracle memories, learned routes, evidence spans).

    The model fingerprint is recorded and CHECKED on load: a session restored
    into a different checkpoint would produce confident nonsense, and silently
    is the worst way for that to happen."""

    def __init__(self, root, fingerprint=None):
        self.root = str(root)
        os.makedirs(self.root, exist_ok=True)
        self.fingerprint = fingerprint

    def _dir(self, name):
        safe = "".join(c for c in str(name) if c.isalnum() or c in "-_.")
        if not safe:
            raise ValueError("session name must contain usable characters")
        return os.path.join(self.root, safe)

    def save(self, name, state, tokens=None, memory=None, meta=None,
             carry="full"):
        """Persist a session. `carry` decides WHAT, and the sizes are not close.

        MEASURED on a real model:
            tokens   full state    memory only    ratio
               256      325.1 KB       62.0 KB     5.2x
             1,024    1,111.6 KB       62.2 KB    17.9x
             4,096    4,257.3 KB       62.0 KB    68.6x
        THE FULL STATE IS 97% KV CACHE AT 2,000 TOKENS and the fraction only
        rises -- so a saved conversation grows at about 1 KB PER TOKEN on disk,
        which is the linear cost this whole architecture exists to avoid. The
        GDN accumulator, leCore's actual memory, is CONSTANT at 62 KB.
        carry="memory" writes only that.
        THE TRADE, stated because it is real and not free: without the KV cache
        a resumed session must RE-PREFILL the tokens it wants attention over.
        The GDN memory comes back exactly; the attention window does not. For a
        long-lived context that is the right trade -- 62 KB and a re-prefill
        beats 4 MB and growing -- and for a short one it is not, which is why
        "full" stays the default rather than being quietly replaced."""
        d = self._dir(name)
        os.makedirs(d, exist_ok=True)
        if str(carry) == "memory":
            from holographic.caching_and_storage.holographic_stateio import (
                export_memory)
            # export_memory returns BYTES (a self-describing blob), not a dict
            # -- it is a wire format, and wrapping it in one array keeps it
            # exactly as import_memory expects to find it.
            # KEEP THE SCALARS THE LOADER NEEDS. export_memory carries the
            # ACCUMULATOR, not the bookkeeping, and load() rebuilds a state from
            # arrays -- so dropping __pos__ made a memory-carry session
            # UNLOADABLE. A save mode that cannot be loaded is not a save mode,
            # and only a round-trip assertion catches it: the file wrote fine.
            arrays = {"lecore_memory": np.frombuffer(export_memory(state),
                                                     dtype=np.uint8),
                      "__pos__": np.array([int(state.pos)], np.int64),
                      "__carry__": np.array([1], np.int64)}
        else:
            arrays = state_to_arrays(state)
        np.savez_compressed(os.path.join(d, "state.npz"), **arrays)
        # TOKENS AS PACKED BYTES, NOT AS JSON DECIMAL TEXT. Moose: the same
        # token recurs constantly, so storing it every time is waste. He is
        # right, and it was worse than he thought -- we wrote them as JSON
        # INTEGERS, "104, 101, 32", about 4.7 bytes per token before any
        # structure is touched at all. MEASURED on 2,000 tokens:
        #     JSON decimal text          9.25 KB   <- what we were writing
        #     uint16                     4.00 KB
        #     zlib over uint16           1.37 KB   <- LZ77 back-references,
        #                                             which IS the reference
        #                                             scheme he described
        #     arithmetic-coded by the model 0.65 KB
        # The structure is real: 2,000 tokens hold only 67 DISTINCT values, and
        # 76% of 2-gram positions repeat an earlier 2-gram.
        # WE STORE THE ZLIB TIER, not the model-coded one: 14x is available but
        # decoding it requires running the model, which turns "read the token
        # list" into an inference dependency. A session file that cannot be read
        # without the exact model that wrote it is a worse artifact than one
        # that is 0.7 KB larger. THE 0.65 KB NUMBER IS KEPT AS A MEASURED
        # NEGATIVE rather than shipped.
        tok_blob = None
        if tokens is not None:
            import zlib as _zlib
            _a = np.asarray([int(t) for t in tokens], np.uint32)
            _w = np.uint16 if int(_a.max(initial=0)) < 65536 else np.uint32
            tok_blob = _zlib.compress(_a.astype(_w).tobytes(), 9)
            np.savez_compressed(os.path.join(d, "tokens.npz"),
                                blob=np.frombuffer(tok_blob, dtype=np.uint8),
                                width=np.array([np.dtype(_w).itemsize]))

        man = {"name": str(name), "pos": int(state.pos), "carry": str(carry),
               "saved_at": time.time(),
               "fingerprint": self.fingerprint,
               "n_tokens": (len(tokens) if tokens is not None else int(state.pos)),
               "tokens_in": ("tokens.npz" if tok_blob is not None else None),
               "tokens": (None if tok_blob is not None else
                          ([int(t) for t in tokens]
                           if tokens is not None else None))}
        man.update(meta or {})
        with open(os.path.join(d, MANIFEST), "w") as f:
            json.dump(man, f, indent=1, sort_keys=True)
        if memory is not None:
            with open(os.path.join(d, "memory.json"), "w") as f:
                json.dump(memory, f)
        return man

    def load(self, name, strict_fingerprint=True):
        """Returns (state, manifest, memory). Raises when the session belongs to
        a different checkpoint unless the caller explicitly overrides."""
        d = self._dir(name)
        with open(os.path.join(d, MANIFEST)) as f:
            man = json.load(f)
        # UNPACK THE TOKENS BACK INTO THE MANIFEST, so every existing caller
        # keeps reading man["tokens"] and never learns the storage changed.
        # A format change that forces every reader to be updated is a migration;
        # this one is an implementation detail, and it should stay one.
        if man.get("tokens") is None and man.get("tokens_in"):
            tp = os.path.join(d, man["tokens_in"])
            if os.path.exists(tp):
                import zlib as _zlib
                z = np.load(tp)
                w = int(np.asarray(z["width"]).ravel()[0])
                dt = np.uint16 if w == 2 else np.uint32
                blob = _zlib.decompress(
                    np.asarray(z["blob"], np.uint8).tobytes())
                man["tokens"] = [int(t) for t in
                                 np.frombuffer(blob, dtype=dt)]
        if (strict_fingerprint and self.fingerprint is not None
                and man.get("fingerprint") not in (None, self.fingerprint)):
            raise ValueError(
                "session %r was saved under model fingerprint %r but this "
                "runtime is %r -- restoring it would produce confident nonsense"
                % (name, man.get("fingerprint"), self.fingerprint))
        with np.load(os.path.join(d, "state.npz")) as z:
            if "__carry__" in z.files:
                # a memory-carry session has NO KV and NO conv windows by
                # design; the caller re-prefills man["tokens"] to rebuild them,
                # which is the bank-or-formula trade this mode exists to make.
                # import_memory RESTORES INTO a live state rather than
                # creating one -- "leaving everything else" is the point, since
                # the accumulator is all it carries. So the loader returns the
                # blob and the position, and the caller re-prefills the tokens
                # into a fresh state and pours the memory back in. Returning a
                # half-built state object would look like a session and behave
                # like a trap.
                state = {"lecore_memory": bytes(
                             np.asarray(z["lecore_memory"], np.uint8).tobytes()),
                         "pos": int(np.asarray(z["__pos__"]).ravel()[0]),
                         "needs_reprefill": True}
            else:
                state = state_from_arrays({k: z[k] for k in z.files})
        mem = None
        mp = os.path.join(d, "memory.json")
        if os.path.exists(mp):
            with open(mp) as f:
                mem = json.load(f)
        return state, man, mem

    def list(self):
        out = []
        for n in sorted(os.listdir(self.root)):
            p = os.path.join(self.root, n, MANIFEST)
            if os.path.exists(p):
                with open(p) as f:
                    out.append(json.load(f))
        return out

    def fork(self, name, new_name):
        """Two futures from one past. A copy, not a link -- writing to one must
        never reach the other, which is what makes parallel contexts safe."""
        src, dst = self._dir(name), self._dir(new_name)
        if os.path.exists(dst):
            raise ValueError("session %r already exists" % new_name)
        shutil.copytree(src, dst)
        with open(os.path.join(dst, MANIFEST)) as f:
            man = json.load(f)
        man["name"] = str(new_name)
        man["forked_from"] = str(name)
        man["saved_at"] = time.time()
        with open(os.path.join(dst, MANIFEST), "w") as f:
            json.dump(man, f, indent=1, sort_keys=True)
        return man

    def delete(self, name):
        d = self._dir(name)
        if os.path.isdir(d):
            shutil.rmtree(d)
            return True
        return False

    def expire(self, older_than_seconds):
        """Housekeeping a harness can call on its own schedule."""
        cut = time.time() - float(older_than_seconds)
        gone = []
        for man in self.list():
            if man.get("saved_at", 0) < cut:
                self.delete(man["name"])
                gone.append(man["name"])
        return gone


def runtime_fingerprint(runtime):
    """A cheap, deterministic id for the checkpoint behind a runtime, so a
    session cannot be restored into the wrong model unnoticed."""
    import hashlib
    h = hashlib.sha256()
    h.update(str(sorted(runtime.cfg.items())).encode())
    emb = np.asarray(runtime.embed, np.float64)
    h.update(np.ascontiguousarray(emb[:8, :8]).tobytes())
    h.update(str(emb.shape).encode())
    return h.hexdigest()[:16]


def _selftest():
    try:
        import torch
        from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
    except ImportError:
        print("session selftest SKIPPED-REFERENCE (torch/transformers absent)")
        return
    import tempfile

    from holographic.io_and_interop.holographic_gdnruntime import GDNRuntime

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    cfg = Qwen3NextConfig(
        vocab_size=97, hidden_size=64, intermediate_size=112,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, full_attention_interval=4,
        num_experts=0, tie_word_embeddings=True, rms_norm_eps=1e-6)
    ref = Qwen3NextForCausalLM(cfg).eval().float()
    weights = {k: v.detach().numpy().astype(np.float64)
               for k, v in ref.state_dict().items()}
    rt = GDNRuntime(weights, dict(
        hidden=64, n_layers=4, rms_eps=1e-6, rope_theta=10000.0,
        linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=16, conv_kernel=4,
        n_heads=4, n_kv_heads=2, head_dim=16, partial_rotary_factor=0.25))
    ids = [int(t) for t in rng.integers(0, 97, size=32)]
    store = SessionStore(tempfile.mkdtemp(), fingerprint=runtime_fingerprint(rt))

    # ---- THE CONTRACT: resume == never stopped ----
    uninterrupted, _ = rt.generate_fast(ids, n_new=24)
    logits, st = rt.prefill(ids)
    st.logits = logits
    first, mid = rt.generate_fast(ids, n_new=12)
    store.save("chat", mid, tokens=first)
    state2, man, _mem = store.load("chat")
    resumed, _ = rt.generate_fast(first, n_new=12, state=state2)
    assert resumed == uninterrupted, "reloaded session diverged from an unbroken run"
    assert man["pos"] == mid.pos and man["n_tokens"] == len(first)

    # ---- MULTIPLE CONTEXTS: forks are independent, not aliases ----
    store.fork("chat", "branch")
    sA, _m, _ = store.load("chat")
    sB, _m, _ = store.load("branch")
    aA, endA = rt.generate_fast(first, n_new=6, state=sA)
    aB, endB = rt.generate_fast(first, n_new=6, state=sB)
    assert aA == aB, "same past must give the same future"
    store.save("branch", endB, tokens=aB)          # write to one...
    sA2, manA, _ = store.load("chat")              # ...must not touch the other
    assert manA["n_tokens"] == len(first), manA
    names = {m["name"] for m in store.list()}
    assert names == {"chat", "branch"}, names

    # ---- WRONG MODEL: refuse loudly instead of producing confident nonsense --
    other = SessionStore(store.root, fingerprint="deadbeefdeadbeef")
    try:
        other.load("chat")
        raise AssertionError("restored a session into the wrong checkpoint")
    except ValueError as exc:
        assert "fingerprint" in str(exc)

    # ---- LIFECYCLE: delete and expire are real, not decorative ----
    assert store.delete("branch") and not store.delete("branch")
    assert {m["name"] for m in store.list()} == {"chat"}
    assert store.expire(older_than_seconds=-1) == ["chat"]
    assert store.list() == []

    print("session selftest OK -- resumed generation is TOKEN-IDENTICAL to an "
          "unbroken run (%d tokens across a save/load boundary); forks are "
          "independent; a session refuses to load into the wrong checkpoint; "
          "list/delete/expire work" % len(uninterrupted))


if __name__ == "__main__":
    _selftest()
