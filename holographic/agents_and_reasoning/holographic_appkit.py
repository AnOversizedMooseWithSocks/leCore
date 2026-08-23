"""APPKIT -- one object an app built on leCore hands to each of its users.

leStudio (an image editor on this engine) had to hand-roll a capability preflight, a
singleton mind, and its own container format before it could start. Its author wrote the
proposals down; this is the layer underneath them. An app should not have to think about
partitions, salting, provenance or chain mining to get a memory that grows with the person
using it.

    from holographic.agents_and_reasoning.holographic_appkit import App
    app = App("lestudio", user="ana", root="~/.lecore")

    app.remember("what brush do I use for skin", "the soft round at 12% flow")
    app.recall("what brush do I use for skin")      # -> {answer, provenance, tier}
    app.observe("retouch a portrait",                # a sequence that WORKED
                ["duplicate layer", "frequency separation", "soft round 12%", "curves lift"])
    app.suggest("retouch a portrait")                # -> the steps that worked before
    app.habits()                                     # -> subsequences this user repeats

ISOLATION IS PHYSICAL, NOT POLITE. Each (app, user) gets its OWN partition directory, so
one user's memory cannot appear in another's by any code path -- not by a similarity hit,
not by a shared fallback, not by a bug in a salt. openzoo tenants, two people on one
laptop, and two apps on one machine are all the same case, handled the same way. Session
salting still exists INSIDE a user's own space for topics they want kept apart.

WHY APPS LEARN PROCEDURES BETTER THAN CHAT DOES. plan_warm keys on the goal's content
words, so prose paraphrase misses ("roll back a bad deploy" warms, "what to do when a
deployment goes bad" does not). An app's goals are STRUCTURED -- tool names, node kinds,
menu paths -- and structured goals repeat verbatim. The weakest part of the chain path for
conversation is the strongest part for an app.

WHAT AN APP GETS FOR FREE: provenance on every answer (`taught` vs `model-cached`, so a
guess never masquerades as the user's own preference), the veto (`forget`), a capability
preflight that reports what THIS build actually has, and a save/load that keeps unknown
sections untouched so two apps can share one file.
"""
import json
import os

import numpy as np


class App:
    """A leCore substrate scoped to one app and one user."""

    def __init__(self, name, user="default", root=None, llm=None, doctrine=False,
                 dim=512, seed=0):
        import lecore
        self.name = str(name)
        self.user = str(user)
        base = os.path.expanduser(root or os.path.join("~", ".lecore"))
        # PHYSICAL isolation: app/user is a directory, not a prefix on a key.
        self.partition = os.path.join(base, "apps", _safe(self.name), _safe(self.user))
        os.makedirs(self.partition, exist_ok=True)
        self.mind = lecore.UnifiedMind(dim=dim, seed=seed)
        if llm is not None:
            self.mind.zoo_attach(llm)
        try:
            self.mind.learning_load(self.partition)
        except Exception:
            pass                                    # a first run has nothing to load
        if doctrine:
            try:
                self.mind.doctrine_load()
            except Exception:
                pass
        self._dirty = False

    # ---------------------------------------------------------------- memory
    def remember(self, question, answer, topic=None):
        """Establish something for THIS user. Marked `taught`: it is their answer, not a
        model's guess, and `recall` will say so."""
        if topic:
            self.mind.session_open("%s:%s" % (self.name, topic))
        self.mind.teach(str(question), str(answer))
        self._dirty = True
        return {"remembered": True, "topic": topic}

    def recall(self, question, established_only=False):
        """Answer from this user's memory. `established_only` refuses a cached model guess
        and tells you to ask the model instead -- the same contract openzoo exposes as
        taught_only, because a preference the user never stated should never be served as
        though they did."""
        r = self.mind.ask(str(question))
        prov = r.get("provenance", "model-cached")
        if established_only and r.get("tier") == "T0" and prov != "taught":
            return {"tier": "escalate", "answer": None, "provenance": prov,
                    "why": "only a cached guess exists -- ask the user or the model, "
                           "then remember() the real answer"}
        return {"tier": r.get("tier"), "answer": r.get("answer"),
                "provenance": prov, "why": r.get("why")}

    def forget(self, question):
        """Veto an answer. It stops serving in the same breath (it is not deleted from the
        audit record -- leCore keeps what happened)."""
        self.mind.answer_feedback(str(question), ok=False)
        self._dirty = True
        return {"forgotten": True}

    # ---------------------------------------------------------------- habits
    def observe(self, goal, steps, worked=True):
        """Log a sequence that WORKED. This is how the app grows into the user: nobody
        writes a rule, the app just notices what they actually do."""
        steps = [str(s)[:60] for s in list(steps) if str(s).strip()]
        if len(steps) < 2:
            return {"noted": False, "why": "a procedure needs at least two steps"}
        vec = self.mind.semantic_key(str(goal))["vec"][:64]
        self.mind.chain_note(np.asarray(vec, float), [(s, bool(worked)) for s in steps])
        self._dirty = True
        return {"noted": True, "steps": len(steps)}

    def suggest(self, goal, gate=0.7):
        """What this user did last time they were here, or None. Structured goals (tool
        names, node kinds) repeat verbatim, which is why this warms for apps where it
        would miss on prose."""
        vec = self.mind.semantic_key(str(goal))["vec"][:64]
        got = self.mind.plan_warm(np.asarray(vec, float), gate=gate)
        if not got:
            return None
        steps = got.get("steps") if isinstance(got, dict) else got
        return {"steps": list(steps), "source": "this user's own history"}

    def habits(self, min_support=2):
        """Subsequences this user repeats across DIFFERENT goals -- their working style,
        mined rather than configured. Proposals with support counts; promoting one to a
        default is the app's decision, on the app's evidence."""
        out = self.mind.skeleton_mine(min_support=min_support)
        rows = out if isinstance(out, list) else out.get("skeletons", [])
        return [{"steps": list(r["steps"]), "support": int(r["support"])} for r in rows]

    # ---------------------------------------------------------------- plumbing
    def capabilities(self, wanted=()):
        """What THIS build actually has -- the preflight leStudio hand-rolled. Ask before
        you call, so a missing faculty is a disabled button and not a traceback."""
        got = {}
        for n in wanted:
            got[n] = hasattr(self.mind, n)
        return got

    def save(self):
        """Persist this user's memory. Cheap and idempotent; call it on quit."""
        rep = self.mind.learning_save(self.partition)
        self._dirty = False
        return {"partition": self.partition, "bytes": rep.get("bytes"),
                "sections": rep.get("sections")}

    def stats(self):
        lad = self.mind.zoo["ladder"]
        taught = [t for t in getattr(lad, "taught_log", [])]
        return {"app": self.name, "user": self.user, "partition": self.partition,
                "taught": len(taught), "dirty": self._dirty}


def _safe(s):
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s))[:64] or "x"


def _selftest():
    import shutil
    import tempfile
    root = tempfile.mkdtemp(prefix="appkit_")
    try:
        ana = App("lestudio", user="ana", root=root)
        bo = App("lestudio", user="bo", root=root)
        ana.remember("what brush for skin", "soft round at 12% flow")
        got = ana.recall("what brush for skin")
        assert got["answer"] and got["provenance"] == "taught", got

        # ISOLATION IS PHYSICAL: bo must not see ana's answer by any path.
        assert "12%" not in str(bo.recall("what brush for skin").get("answer") or ""), \
            "one user's memory must never surface for another"
        assert ana.partition != bo.partition

        # HABITS: two different goals that share a working subsequence.
        ana.observe("retouch a portrait",
                    ["duplicate layer", "frequency separation", "soft round 12%"])
        ana.observe("retouch a group photo",
                    ["duplicate layer", "frequency separation", "curves lift"])
        sug = ana.suggest("retouch a portrait")
        assert sug and "frequency separation" in sug["steps"], sug
        hab = ana.habits(min_support=2)
        assert any(h["support"] >= 2 for h in hab), hab

        # ESTABLISHED_ONLY refuses a guess rather than dressing it as the user's answer.
        ana.mind.zoo_attach(lambda p: "a model guess")
        ana.mind.ask("what canvas size do I use")           # caches a guess
        strict = ana.recall("what canvas size do I use", established_only=True)
        assert strict["tier"] == "escalate", strict

        # FORGET stops it serving.
        ana.remember("what export format", "png")
        ana.forget("what export format")
        assert "png" not in str(ana.recall("what export format").get("answer") or "")

        rep = ana.save()
        assert rep["bytes"] > 0
        again = App("lestudio", user="ana", root=root)
        assert "12%" in str(again.recall("what brush for skin").get("answer") or ""), \
            "a user's memory must survive a restart"
        return ("OK: App pins passed (per-user physical isolation; taught provenance; "
                "habits mined from observed sequences; established_only refuses a guess; "
                "veto stops serving; memory survives a restart)")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    print(_selftest())
