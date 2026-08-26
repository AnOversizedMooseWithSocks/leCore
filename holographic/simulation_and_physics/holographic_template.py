"""Parameterized recipe templates (ISA-6): a StructureRecipe with named HOLES filled at instantiation, plus a
small named-template library. A template is the macro layer over the assembly -- "bind a value under this role",
"make a two-field record", "make an ordered pair" -- written once and instantiated with different arguments.

Because a StructureRecipe replays BIT-EXACT (atoms are regenerated from the seed by name), instantiating a
template with different arguments produces the correct DISTINCT structures, deterministically.

THE HYGIENE DISCIPLINE (the kept negative, designed out): atoms are derived from NAMES, so two atoms with the
same name are the SAME vector. A template that creates an internal role atom named "role" would COLLIDE with a
caller who fills a hole with an atom also named "role" -- the role and the value become one vector (capture), and
the binding degenerates. So template-INTERNAL atoms are namespaced under a reserved prefix ("@tmpl:<name>:") that
a caller's bare names cannot hit. The witness of capture is cosine(internal_role, caller_value): ~0 with the
discipline, 1.0 without it. `RecipeTemplate` is hygienic by construction; `_UnhygienicTemplate` (tests only)
shows what the discipline prevents.
"""

from holographic.misc.holographic_recipe import StructureRecipe

# The reserved namespace for template-internal atoms. A caller naming an atom is expected never to use this
# prefix; that convention is the whole fresh-atom discipline (a gensym keyed by template name instead of a
# global counter, so the same template is still deterministic across instantiations).
TMPL_NS = "@tmpl:"


class _Builder:
    """Handed to a template's build function. Wraps a StructureRecipe so that:
      * `hole(name)` resolves to the caller's filled argument (a handle already in the recipe), and
      * `atom(name)` creates a template-INTERNAL atom, AUTO-NAMESPACED for hygiene (cannot collide with caller
        atoms). bind/bundle/permute pass straight through to the recipe."""

    def __init__(self, recipe, ns, holes, hygienic=True):
        self.r = recipe
        self._ns = ns
        self._holes = holes
        self._hygienic = hygienic

    def hole(self, name):
        return self._holes[name]

    def atom(self, name, unitary=False):
        # HYGIENE: a template-internal atom is namespaced by the template name so a caller's bare `name` can
        # never derive the same vector. The unhygienic path (tests only) omits the prefix to exhibit capture.
        full = (TMPL_NS + self._ns + ":" + name) if self._hygienic else name
        return self.r.atom(full, unitary=unitary)

    def bind(self, a, b):
        return self.r.bind(a, b)

    def bundle(self, members):
        return self.r.bundle(members)

    def permute(self, a, shift):
        return self.r.permute(a, shift)


class RecipeTemplate:
    """A named, parameterized recipe template. `params` are the hole names; `build(t)` emits the structure using
    `t.hole(...)` for arguments and `t.atom(...)` for (namespaced) internal atoms, returning the output handle."""

    _hygienic = True

    def __init__(self, name, params, build):
        self.name = str(name)
        self.params = list(params)
        self._build = build

    def instantiate(self, dim, seed, **args):
        """Fill the holes and return a concrete StructureRecipe (replayable, bit-exact). A string argument is a
        named atom; anything array-like is stored as a literal `raw` vector."""
        missing = set(self.params) - set(args)
        extra = set(args) - set(self.params)
        if missing or extra:
            raise ValueError(f"template {self.name!r} expects {self.params}; missing={sorted(missing)} "
                             f"extra={sorted(extra)}")
        r = StructureRecipe(dim, seed)
        holes = {}
        for p in self.params:
            v = args[p]
            holes[p] = r.atom(str(v)) if isinstance(v, str) else r.raw(v)   # caller atom (by name) or literal
        t = _Builder(r, self.name, holes, hygienic=self._hygienic)
        out = self._build(t)
        r.mark_output(out)
        return r

    def build_vector(self, dim, seed, **args):
        """Instantiate and materialize the output vector (the common case)."""
        r = self.instantiate(dim, seed, **args)
        return r.get(r._outputs[-1])

    def role_atom(self, dim, seed, name, unitary=True):
        """The vector of a named INTERNAL atom -- exposed so callers/tests can verify hygiene (that it lives in
        the reserved namespace, disjoint from any bare caller name)."""
        from holographic.agents_and_reasoning.holographic_ai import derived_atom
        full = (TMPL_NS + self.name + ":" + name) if self._hygienic else name
        return derived_atom(int(seed), full, int(dim), unitary=unitary)


class _UnhygienicTemplate(RecipeTemplate):
    """A template WITHOUT the fresh-atom discipline -- internal atoms keep their bare names. Used only to show
    the capture the discipline prevents; never put in the public library."""
    _hygienic = False


# ---- the starter library -------------------------------------------------------------------------------------
# Each template's internal role atoms are created via t.atom(...) and so are automatically namespaced (hygienic).

def _pair(t):
    # tag a value under a fixed role: bind(role, x). Recover the value by unbinding the role.
    return t.bind(t.atom("role", unitary=True), t.hole("x"))


def _record(t):
    # a two-field record: bundle of (KEY-role bound to key) and (VAL-role bound to val). Read a field by
    # unbinding its role. The two roles are distinct internal atoms, so the fields don't cross-contaminate.
    return t.bundle([t.bind(t.atom("KEY", unitary=True), t.hole("key")),
                     t.bind(t.atom("VAL", unitary=True), t.hole("val"))])


def _ordered_pair(t):
    # an order-bearing pair: bundle(a, permute(b)). Position is encoded by the permutation, so (a,b) != (b,a).
    return t.bundle([t.hole("a"), t.permute(t.hole("b"), 1)])


STARTER_LIBRARY = {
    "pair": RecipeTemplate("pair", ["x"], _pair),
    "record": RecipeTemplate("record", ["key", "val"], _record),
    "ordered_pair": RecipeTemplate("ordered_pair", ["a", "b"], _ordered_pair),
}


def _selftest():
    import numpy as np
    from holographic.agents_and_reasoning.holographic_ai import unbind, cosine, derived_atom
    dim, seed = 1024, 0

    pair = STARTER_LIBRARY["pair"]
    # bit-exact determinism: same args -> identical vector
    v1 = pair.build_vector(dim, seed, x="a")
    v2 = pair.build_vector(dim, seed, x="a")
    assert np.array_equal(v1, v2), "instantiation is not deterministic"
    # distinct args -> distinct structures, each recovering its value via the role
    vb = pair.build_vector(dim, seed, x="b")
    assert cosine(v1, vb) < 0.5, "pair(a) and pair(b) should be distinct"
    role = pair.role_atom(dim, seed, "role")
    assert cosine(unbind(v1, role), derived_atom(seed, "a", dim)) > 0.99, "pair(a) must recover 'a' under role"
    assert cosine(unbind(vb, role), derived_atom(seed, "b", dim)) > 0.99, "pair(b) must recover 'b' under role"

    # HYGIENE: even when the caller's value name equals the internal role's BARE name ("role"), the namespaced
    # internal atom is disjoint from the caller's atom -> no capture. (Compare like-for-like: collision is a
    # NAME question, so we hold the unitary flag fixed.)
    captured = cosine(pair.role_atom(dim, seed, "role"), derived_atom(seed, "role", dim, unitary=True))
    assert captured < 0.1, f"hygienic role collided with caller atom (cos {captured})"
    # the unhygienic version DOES capture: its bare role atom IS the caller's "role" atom of the same kind
    bad = _UnhygienicTemplate("pair", ["x"], _pair)
    assert cosine(bad.role_atom(dim, seed, "role"), derived_atom(seed, "role", dim, unitary=True)) > 0.99

    print("holographic_template selftest: ok (parameterized templates, bit-exact, hygienic)")


if __name__ == "__main__":
    _selftest()
