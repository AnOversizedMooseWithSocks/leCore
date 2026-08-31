"""Part 24 of UnifiedMind's faculty surface -- THE DARK DOORS (sweep 123, release audit).

Five modules that were 'findable via the catalog but not a mind faculty' -- the reachability
audit's declared import-only list -- and that a power user would reach for under a verb:
the reasoning kit (split-conformal intervals, the epistemic map, the semantic compass), the
node-to-node tool client, versioned table history, the determinism contract (plus a twins
report composed from it), and the microfacet BRDF terms. Reachable by import was never
reachable: every capability must be callable over /invoke. Each method here DELEGATES; the
logic stays in its family module.

NOT A STANDALONE MODULE. One slice of the single `UnifiedMind` class, assembled by
holographic/misc/holographic_unified.py, which is still the only import path anyone uses.
Carries no `__init__`; assumes the state UnifiedMind.__init__ sets up.
"""
import numpy as np

from holographic.unified import check_part


class _UnifiedPart24:

    # ---- reasoning kit (holographic_reasoning) ---------------------------------------------

    def conformal_interval(self, residuals, prediction, alpha=0.1):
        """SPLIT-CONFORMAL INTERVAL in one call: calibrate on held-out |residuals| then wrap a
        point prediction in the (1-alpha) band. Distribution-free coverage under
        exchangeability -- the honest error bar for any predictor. Distinct from
        adaptive_conformal (an ONLINE quantile tracker): this is the batch form.
        See holographic_reasoning.ConformalPredictor."""
        from holographic.agents_and_reasoning.holographic_reasoning import ConformalPredictor
        cp = ConformalPredictor(alpha=float(alpha))
        cp.calibrate(np.asarray(residuals, dtype=np.float64))
        return cp.interval(prediction)

    def epistemic_map(self, density_a, density_b, disagreement,
                      density_threshold=2, disagree_threshold=0.15):
        """CLASSIFY WHAT KIND OF NOT-KNOWING this is: sparse evidence, contested evidence, or
        settled -- from two evidence densities and a disagreement score. The map that tells a
        model whether to gather more, arbitrate, or answer. See holographic_reasoning.EpistemicMap."""
        from holographic.agents_and_reasoning.holographic_reasoning import EpistemicMap
        return EpistemicMap(density_threshold=int(density_threshold),
                            disagree_threshold=float(disagree_threshold)).classify(
            density_a, density_b, disagreement)

    def vector_disagreement(self, vec_a, vec_b):
        """How much two evidence vectors DISAGREE (0 = same direction). The disagreement input
        epistemic_map consumes. See holographic_reasoning.vector_disagreement."""
        from holographic.agents_and_reasoning.holographic_reasoning import vector_disagreement
        return vector_disagreement(np.asarray(vec_a, dtype=np.float64),
                                   np.asarray(vec_b, dtype=np.float64))

    def semantic_compass(self):
        """A SEMANTIC COMPASS handle: record(vec, success) what worked, direction() the learned
        heading, steer(query, step, k) a query toward past success. Returned as a handle
        because it accumulates. See holographic_reasoning.SemanticCompass."""
        from holographic.agents_and_reasoning.holographic_reasoning import SemanticCompass
        return SemanticCompass()

    # ---- node-to-node tool client (holographic_toolclient) ---------------------------------

    def remote_tools(self, base_url, token=None, timeout=30.0):
        """LIST another leCore node's tools the same way leCore is served (GET /tools) -- the
        multi-node door: a zoo node calls a zoo node. See holographic_toolclient.list_tools."""
        from holographic.io_and_interop.holographic_toolclient import list_tools
        return list_tools(str(base_url), token=token, timeout=float(timeout))

    def remote_call(self, base_url, name, args=None, token=None, timeout=30.0):
        """CALL one tool on another leCore node (POST /invoke). Strict JSON in, strict JSON out;
        the receipt discipline is the remote node's. See holographic_toolclient.call."""
        from holographic.io_and_interop.holographic_toolclient import call
        return call(str(base_url), str(name), args=dict(args or {}), token=token,
                    timeout=float(timeout))

    # ---- versioned table history (holographic_querytime) ------------------------------------

    def table_history(self, table):
        """A GIT-LIKE TIMELINE for one query table (make_table builds one): commit() snapshots,
        checkout(version) restores. The handle the other table_* history verbs read. See
        holographic_querytime.TableHistory."""
        from holographic.agents_and_reasoning.holographic_querytime import TableHistory
        return TableHistory(table)

    def table_commit(self, history, table, note=""):
        """Snapshot the table's current rows into the history with a note; returns the version."""
        return history.commit(table, note=str(note))

    def table_as_of(self, history, version, sql):
        """Run a query AGAINST A PAST VERSION -- 'what did this table look like at v3'. See
        holographic_querytime.select_as_of."""
        from holographic.agents_and_reasoning.holographic_querytime import select_as_of
        return select_as_of(history, version, sql)

    def table_diff(self, history, version_a, version_b, pk_col=None):
        """What changed between two versions (added / removed / changed rows, keyed by pk_col
        when given). See holographic_querytime.diff_versions."""
        from holographic.agents_and_reasoning.holographic_querytime import diff_versions
        return diff_versions(history, version_a, version_b, pk_col=pk_col)

    def table_revert(self, history, version):
        """Restore a past version as the current table (the history keeps every step). See
        holographic_querytime.revert_to."""
        from holographic.agents_and_reasoning.holographic_querytime import revert_to
        return revert_to(history, version)

    def table_versions(self, history, pk_col, key):
        """One row's timeline: every version of the record with primary key `key`. See
        holographic_querytime.history_of."""
        from holographic.agents_and_reasoning.holographic_querytime import history_of
        return history_of(history, pk_col, key)

    # ---- the determinism contract (holographic_determinism) ---------------------------------

    def deterministic_topk(self, scores, k):
        """TOP-K WITH A FIXED TIE RULE: identical scores return identical indices on every
        machine (ISA-1). Use it anywhere a sort decides a trajectory. See
        holographic_determinism.topk_det."""
        from holographic.misc.holographic_determinism import topk_det
        return topk_det(np.asarray(scores, dtype=np.float64), int(k))

    def hash_unit(self, *keys):
        """A DETERMINISTIC UNIT FLOAT in [0, 1) from any keys -- hashlib, never hash(): the
        same keys give the same number in every process and every seed. See
        holographic_determinism.hash_unit."""
        from holographic.misc.holographic_determinism import hash_unit
        return hash_unit(*keys)

    def determinism_report(self, n_facts=120):
        """THE TWINS PROBE AS A VERB (sweep 107's gauntlet made routine): two independent
        minds, identical teach streams, byte-compare the saved learning partitions. The
        constitutional property, measurable on demand before a release. Returns
        {byte_identical, sha256, n_facts}; a False here is a stop-the-release finding."""
        import hashlib, os, tempfile
        shas = []
        for _ in range(2):
            m = self.__class__(dim=self.dim, seed=0)
            for i in range(int(n_facts)):
                m.teach("twin fact %d topic %d" % (i, i % 53), "payload %d" % (i * 13))
            root = tempfile.mkdtemp()
            m.learning_save(root)
            shas.append(hashlib.sha256(open(os.path.join(root, "learning", "state.lecore"),
                                            "rb").read()).hexdigest())
        return {"byte_identical": shas[0] == shas[1], "sha256": shas[0], "n_facts": int(n_facts)}

    # ---- microfacet BRDF terms (holographic_brdf) -------------------------------------------

    def brdf_terms(self, n_dot_h, n_dot_v, n_dot_l, roughness, base_color=None, metallic=0.0,
                   f0=None):
        """THE COOK-TORRANCE / GGX TERMS in one call: D (GGX normal distribution), G (Smith
        masking-shadowing), F0 (from base colour + metallic when f0 is not given) and F
        (Schlick Fresnel at n.v). Scalars or arrays. The pieces a shader or a material fit
        needs, without hand-rolling a BRDF. See holographic_brdf."""
        from holographic.rendering.holographic_brdf import (fresnel_schlick, d_ggx, g_smith,
                                                             metallic_f0)
        if f0 is None:
            f0 = metallic_f0(np.asarray(base_color if base_color is not None else [0.04, 0.04, 0.04],
                                        dtype=np.float64), float(metallic))
        return {"D": d_ggx(n_dot_h, float(roughness)),
                "G": g_smith(n_dot_v, n_dot_l, float(roughness)),
                "F0": f0,
                "F": fresnel_schlick(n_dot_v, f0)}


def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p24_wired", "_UnifiedPart24")
    print("holographic_unified_p24_wired selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()
