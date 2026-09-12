"""THE CARD CONTRACT: does a catalog card's `method=` name a door the card itself can reach?

WHY THIS FILE EXISTS. `tools/skill_lint.py` validated the `mind.X(` references INSIDE a card's
example and never the card's own `method=` field. That is exactly how four cards promised doors that
had never existed while every gate stayed green -- `bios_boot`, `doctrine_seedpack`, `panel_realm`,
`phase_randomized_null`, each named after a CONCEPT with `method=` defaulting to that name while the
door its own example called had another (`bios_boot` -> `boot`). It took an audit outside skill_lint
to notice.

AND THE FIXTURE RULE THIS FILE OBEYS, because the swarm has now been bitten by it four times: A TEST
WHOSE FIXTURE IS A REAL BUG DIES THE DAY SOMEBODY FIXES IT. My own round-5 test asserted the
`no_floor` list was non-empty and went red the moment those four cards were repaired. So every
assertion here runs against a SYNTHETIC card set. The live catalog is asserted to be CLEAN -- zero is
the goal state, and a green run here means the check is working, not that it has nothing to say.

OWNERSHIP, stated so a future sweep does not add a second gate: this check asks whether a CARD can
reach the door it names, from the card alone, at 0.07 s. `swarm_audit --gate` owns reachability
ACROSS the layers -- it AST-walks the repo to split an unresolved `method=` into an object method
(fine), a module function (its `unreachable` floor) or nothing at all, at 10.5 s. Two gates saying
the same thing in different words is how one of them gets ignored.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import skill_lint                                                          # noqa: E402
import holographic.caching_and_storage.holographic_catalog as cat_mod      # noqa: E402


class _Card:
    """The three fields the contract check reads. Deliberately not a real Capability: the point is
    to exercise the classifier on inputs the live catalog does not contain."""

    def __init__(self, name, method, example):
        self.name, self.method, self.example = name, method, example


def _with_cards(monkeypatch, cards):
    class _Cat:
        def all(self):
            return list(cards)
    monkeypatch.setattr(cat_mod, "default_catalog", lambda *a, **k: _Cat())
    return skill_lint.audit_card_contract()


# --------------------------------------------------------------------------------------
# 1. The live catalog is CLEAN, and that is the goal state.
# --------------------------------------------------------------------------------------

def test_the_live_catalog_has_no_broken_card_contract():
    r = skill_lint.audit_card_contract()
    assert r["broken"] == [], r["broken"]
    assert r["checked"] > 700, "the check must actually be looking at the whole catalog"


# --------------------------------------------------------------------------------------
# 2. It fires on the shape that slipped through -- proven synthetically, never on a real bug.
# --------------------------------------------------------------------------------------

def test_it_catches_a_card_named_after_a_concept_whose_example_calls_another_door(monkeypatch):
    # THE EXACT SHAPE OF ALL FOUR no_floor CARDS: the card is named for a concept, method= defaults
    # to that name, and the example calls the real door under a different one.
    r = _with_cards(monkeypatch, [
        _Card("BIOS boot", "bios_boot", "import lecore; m=lecore.UnifiedMind(); m.boot()")])
    assert [b[0] for b in r["broken"]] == ["bios_boot"]
    assert r["checked"] == 1


def test_it_does_not_fire_on_a_card_whose_method_is_a_real_verb(monkeypatch):
    r = _with_cards(monkeypatch, [
        _Card("Boot leCore", "boot", "import lecore; m=lecore.UnifiedMind(); m.boot()")])
    assert r["broken"] == []


def test_it_does_not_fire_on_a_module_function_card(monkeypatch):
    # THE DELIBERATE NON-OVERLAP with swarm_audit. These 15 cards name a module function an agent
    # cannot /invoke -- a real limitation, and swarm_audit's `unreachable` floor. But the card DOES
    # show a way in (by import), so it is not lying, and re-reporting it here would be the second
    # gate in different words.
    r = _with_cards(monkeypatch, [
        _Card("aces_tonemap", "aces_tonemap",
              "from holographic.rendering.holographic_gbuffer import aces_tonemap")])
    assert r["broken"] == [], "a module-function card must be swarm_audit's business, not this one's"


def test_a_card_with_no_example_at_all_and_no_door_is_broken(monkeypatch):
    r = _with_cards(monkeypatch, [_Card("ghost", "a_door_that_never_existed", "")])
    assert [b[0] for b in r["broken"]] == ["a_door_that_never_existed"]


def test_a_card_with_no_method_field_is_not_checked(monkeypatch):
    r = _with_cards(monkeypatch, [_Card("prose only", None, "some example")])
    assert r["broken"] == [] and r["checked"] == 0


# --------------------------------------------------------------------------------------
# 3. It gates, and it is cheap enough to sit on the critical path.
# --------------------------------------------------------------------------------------

def test_the_check_is_counted_into_the_exit_code():
    src = open(os.path.join(REPO, "tools", "skill_lint.py"), encoding="utf-8").read()
    assert 'total = gaps + example_gaps + len(al["inert"]) + len(cc["broken"])' in src, \
        "the card contract must gate -- its goal state is 0 and it is 0 today"
    assert "broken card contract" in src


def test_the_check_costs_no_repo_walk():
    # The lesson from sweep 133: 651 regex searches over 4.85 MB cost 43.6 s of a 52.8 s sweep, and
    # an audit that slow is one nobody runs. This one must stay a catalog read.
    import inspect
    src = inspect.getsource(skill_lint.audit_card_contract)
    assert "os.walk" not in src and "_repo_defs" not in src
    assert "ast" not in src.split('"""')[-1], "no AST walk on the critical path"


def test_the_two_instruments_state_who_owns_what():
    # A future sweep adding the same check to both is the failure this comment exists to prevent.
    doc = skill_lint.audit_card_contract.__doc__
    assert "swarm_audit" in doc and "unreachable" in doc
    sa = open(os.path.join(REPO, "tools", "swarm_audit.py"), encoding="utf-8").read()
    assert "unreachable" in sa
