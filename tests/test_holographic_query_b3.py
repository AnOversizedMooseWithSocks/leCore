"""Query BUILD B3: multi-predicate WHERE (AND/OR/parentheses) via a predicate tree."""
import pytest
from holographic.agents_and_reasoning.holographic_query import UserTable, run_sql, parse_where, QueryError


def _t():
    t = UserTable("critters", ["name", "legs", "habitat"], dim=1024, seed=0)
    for r in [{"name": "cat", "legs": 4, "habitat": "land"}, {"name": "duck", "legs": 2, "habitat": "water"},
              {"name": "crab", "legs": 8, "habitat": "water"}, {"name": "ant", "legs": 6, "habitat": "land"}]:
        t.insert(r)
    return t


def test_and():
    r = run_sql("SELECT name FROM critters WHERE legs > 3 AND habitat = 'land'", _t())
    assert {x["name"] for x in r} == {"cat", "ant"}


def test_or():
    r = run_sql("SELECT name FROM critters WHERE legs > 6 OR habitat = 'land'", _t())
    assert {x["name"] for x in r} == {"cat", "crab", "ant"}


def test_parentheses_override_precedence():
    r = run_sql("SELECT name FROM critters WHERE (habitat = 'water' AND legs > 4) OR name = 'cat'", _t())
    assert {x["name"] for x in r} == {"crab", "cat"}


def test_and_binds_tighter_than_or():
    # a = ... OR b AND c  ==  a OR (b AND c)
    r = run_sql("SELECT name FROM critters WHERE name = 'duck' OR habitat = 'land' AND legs > 5", _t())
    assert {x["name"] for x in r} == {"duck", "ant"}         # duck, plus (land AND legs>5)=ant; cat excluded


def test_ge_le_ne_operators():
    assert {x["name"] for x in run_sql("SELECT name FROM critters WHERE legs >= 6", _t())} == {"crab", "ant"}
    assert {x["name"] for x in run_sql("SELECT name FROM critters WHERE legs <= 4", _t())} == {"cat", "duck"}
    assert {x["name"] for x in run_sql("SELECT name FROM critters WHERE habitat != 'water'", _t())} == {"cat", "ant"}


def test_single_predicate_still_works():
    assert {x["name"] for x in run_sql("SELECT name FROM critters WHERE legs = 8", _t())} == {"crab"}


def test_parse_where_tree_shape():
    tree = parse_where("a = 1 AND (b = 2 OR c = 3)")
    assert tree[0] == "and" and tree[1] == ("pred", "a", "=", 1.0) and tree[2][0] == "or"


def test_unknown_column_in_multi_predicate_errors():
    with pytest.raises(QueryError):
        run_sql("SELECT name FROM critters WHERE legs > 1 AND nope = 2", _t())


def test_malformed_where_raises():
    with pytest.raises(ValueError):
        run_sql("SELECT name FROM critters WHERE (legs > 1", _t())   # unbalanced parens
