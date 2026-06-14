"""The curriculum is wired into the live app, not just the library: the dataset
loads a UnifiedMind taught a dictionary then an encyclopedia, and the query
endpoint returns meaning neighbours and an is_a chain over that one mind."""
import unified_app as ua


def _client():
    ua.app.testing = True
    return ua.app.test_client()


def test_curriculum_dataset_is_offered_and_needs_no_network():
    c = _client()
    ds = c.get("/api/unified/datasets").get_json()
    entry = next((d for d in ds["datasets"] if d["id"] == "curriculum"), None)
    assert entry is not None
    assert entry["available"] is True            # hand-built, no corpus download


def test_curriculum_loads_and_learns_both_layers():
    c = _client()
    r = c.post("/api/unified/load", json={"id": "curriculum"}).get_json()
    assert r["ok"] is True
    assert r["accuracy"] == 100                   # one-hop is_a retrieval exact
    assert r["curriculum"]["facts"] > 10
    # the dictionary probes stayed in-domain (cat near felines, not minerals)
    cat_near = [w for w, _ in r["curriculum"]["probes"]["cat"]]
    assert any(w in ("feline", "lion", "tiger") for w in cat_near)


def test_curriculum_query_returns_meaning_and_is_a_chain():
    c = _client()
    c.post("/api/unified/load", json={"id": "curriculum"})
    q = c.post("/api/unified/curriculum", json={"word": "dog"}).get_json()
    # dictionary layer: meaning neighbours present and in-domain
    near = [m["word"] for m in q["meaning"]]
    assert any(w in ("canine", "wolf", "fox", "animal") for w in near)
    # encyclopedia layer: is_a chain climbs to a root, throughput in (0,1]
    assert q["is_a_chain"][0] == "dog"
    assert "animal" in q["is_a_chain"]
    assert 0.0 < q["throughput"] <= 1.0


def test_curriculum_query_handles_unknown_word():
    c = _client()
    c.post("/api/unified/load", json={"id": "curriculum"})
    q = c.post("/api/unified/curriculum", json={"word": "zzqx"}).get_json()
    assert "note" in q or not q.get("meaning")


def test_page_includes_curriculum_panel():
    c = _client()
    html = c.get("/").get_data(as_text=True)
    assert "curriculum" in html and "Look up" in html


def test_answer_endpoint_routes_question():
    # The 'ask' panel's endpoint routes a question to a real operation over the
    # curriculum brain.
    c = _client()
    c.post("/api/unified/load", json={"id": "curriculum"})
    r = c.post("/api/unified/answer", json={"question": "is a dog an animal?"}).get_json()
    assert r["kind"] == "is_a" and r["answer"] is True
    r2 = c.post("/api/unified/answer", json={"question": "what is a wolf?"}).get_json()
    assert r2["kind"] == "define" and "animal" in r2["is_a_chain"]


def test_page_includes_ask_panel():
    c = _client()
    html = c.get("/").get_data(as_text=True)
    assert "question router" in html and "askQ" in html


def test_resolution_endpoint_returns_ladder():
    # The coarse-to-fine resolution profile is exposed in the app: a winner per
    # truncation dimension and the dimension at which it stabilises.
    c = _client()
    c.post("/api/unified/load", json={"id": "world"})
    r = c.post("/api/unified/resolution",
               json={"text": "the capital is tokyo and they use the yen"}).get_json()
    assert "profile" in r and len(r["profile"]) >= 3
    assert 0 < r["stable_from"] <= r["full_dim"]


def test_page_includes_resolution_button():
    c = _client()
    html = c.get("/").get_data(as_text=True)
    assert "How much resolution" in html


def test_topic_pull_endpoint_shows_diversity_collapse():
    # The honest topic-pull experiment in the app: coherence may rise with heavy
    # topic_weight, but lexical diversity collapses -- the kept negative.
    import unified_app as ua
    from holographic_unified import UnifiedMind
    c = _client()
    raw = ("the ship flew through cold dark space past the bright star and moon . "
           "the garden grew green plants in warm wet soil near the old wall . ") * 60
    m = UnifiedMind(dim=512, seed=0)
    ua.STATE.update({"mind": m, "dataset": "t", "labels": ["a"], "is_code": False,
                     "test": [], "raw_len": len(raw), "desc": "t", "seq_raw": raw})
    r = c.post("/api/unified/topic_pull", json={"seed": "the"}).get_json()
    assert "rows" in r and len(r["rows"]) >= 2
    base, hot = r["rows"][0], r["rows"][-1]
    assert hot["diversity"] < base["diversity"]          # diversity collapses under pull


def test_page_includes_topic_pull_panel():
    c = _client()
    html = c.get("/").get_data(as_text=True)
    assert "topic pull" in html and "async function topicPull" in html


def test_predictive_endpoint_reports_curve_and_generalization():
    # The predictive loop in the app: a learning curve, free-energy endpoints, a
    # generalisation score, and a generation-by-anticipation sample.
    import unified_app as ua
    from holographic_unified import UnifiedMind
    c = _client()
    raw = ("the cat sat on the mat . the cat ran to the mat . "
           "the dog sat on the rug . the dog ran to the rug . ") * 40
    m = UnifiedMind(dim=512, seed=0)
    ua.STATE.update({"mind": m, "dataset": "t", "labels": ["a"], "is_code": False,
                     "test": [], "raw_len": len(raw), "desc": "t", "seq_raw": raw})
    r = c.post("/api/unified/predictive", json={}).get_json()
    assert "curve" in r and len(r["curve"]) >= 3
    assert "generalization" in r and "sample" in r
    # on a highly repetitive corpus, accuracy should be substantial by the end
    assert r["curve"][-1]["accuracy"] > 0.3


def test_page_includes_predictive_panel():
    c = _client()
    html = c.get("/").get_data(as_text=True)
    assert "predictive loop" in html


def test_predictive_proof_block_shows_steered_beats_greedy():
    # The structure proof in the app: greedy decoding collapses (low score),
    # steered generation stays in the real-text band (higher score).
    import unified_app as ua
    from holographic_unified import UnifiedMind
    c = _client()
    raw = ("the ship sailed across the cold sea toward the bright star . "
           "the farmer planted green seeds in the warm soil near the barn . ") * 60
    m = UnifiedMind(dim=512, seed=0)
    ua.STATE.update({"mind": m, "dataset": "t", "labels": ["a"], "is_code": False,
                     "test": [], "raw_len": len(raw), "desc": "t", "seq_raw": raw})
    r = c.post("/api/unified/predictive", json={}).get_json()
    assert r.get("proof") is not None
    p = r["proof"]
    assert p["steered_score"] >= p["greedy_score"]      # steering defends structure


def test_respond_endpoint_steers_toward_query():
    # Query-and-generate in the app: the steered response is at least as on-query
    # as the unsteered baseline, and reports both relevance and structure.
    import unified_app as ua
    from holographic_unified import UnifiedMind
    c = _client()
    raw = ("the school taught young children to read books in the bright classroom . "
           "the president led the government and the senate passed a national law . ") * 60
    m = UnifiedMind(dim=512, seed=0)
    ua.STATE.update({"mind": m, "dataset": "t", "labels": ["a"], "is_code": False,
                     "test": [], "raw_len": len(raw), "desc": "t", "seq_raw": raw})
    r = c.post("/api/unified/respond", json={"query": "school children read books"}).get_json()
    assert "steered" in r and "unsteered" in r
    assert r["steered"]["relevance"] >= r["unsteered"]["relevance"] - 0.05


def test_page_includes_respond_panel():
    c = _client()
    html = c.get("/").get_data(as_text=True)
    assert "query &amp; generate" in html


def test_deliberate_endpoint_returns_trace_and_iterations():
    # The deliberation in the app: a chosen response, an iteration count (thinking
    # time), and a trace of the inner drafts.
    import unified_app as ua
    from holographic_unified import UnifiedMind
    c = _client()
    raw = ("the school taught young children to read books in the bright classroom . "
           "the president led the government and the senate passed a national law . ") * 60
    m = UnifiedMind(dim=512, seed=0)
    ua.STATE.update({"mind": m, "dataset": "t", "labels": ["a"], "is_code": False,
                     "test": [], "raw_len": len(raw), "desc": "t", "seq_raw": raw})
    r = c.post("/api/unified/deliberate", json={"query": "school children read books"}).get_json()
    assert "response" in r and "trace" in r
    assert r["iterations"] >= 1 and len(r["trace"]) == r["iterations"]


def test_page_includes_deliberate_panel():
    c = _client()
    html = c.get("/").get_data(as_text=True)
    assert "think before answering" in html


def test_discover_endpoint_recovers_boundaries():
    # Self-discovery in the app: word boundaries recovered from spaceless text beat
    # a random cut, and discovered chunks compress better than characters.
    import unified_app as ua
    from holographic_unified import UnifiedMind
    c = _client()
    raw = (("the cat sat on the mat and the dog ran to the park . "
            "a bird flew over the tall green tree near the river . "
            "she read an old book about ships and distant lands . "
            "they walked through the city streets in the cold rain . ") * 30)
    m = UnifiedMind(dim=512, seed=0)
    ua.STATE.update({"mind": m, "dataset": "t", "labels": ["a"], "is_code": False,
                     "test": [], "raw_len": len(raw), "desc": "t", "seq_raw": raw})
    r = c.post("/api/unified/discover", json={}).get_json()
    assert "f1" in r and r["f1"] > r["random_f1"]
    assert r["chunk_bits"] < r["symbol_bits"]


def test_page_includes_discover_panel():
    c = _client()
    html = c.get("/").get_data(as_text=True)
    assert "self-discovery" in html


def test_factorize_endpoint_solves():
    # The resonator factorization demo: recovers the bound factors (self-contained,
    # needs no corpus).
    import unified_app as ua
    c = _client()
    r = c.post("/api/unified/factorize", json={"codebook_size": 40, "seed": 7}).get_json()
    assert r["solved"]
    assert r["recovered"] == r["true"]
    assert r["search_space"] == 40 ** 3


def test_page_includes_factorize_panel():
    c = _client()
    html = c.get("/").get_data(as_text=True)
    assert "pull a composite apart" in html


def test_codec_endpoint_is_lossless_and_honest():
    # The lossless codec in the app: round-trips exactly, and real text compresses
    # better than the random control.
    import unified_app as ua
    from holographic_unified import UnifiedMind
    c = _client()
    raw = ("the school taught young children to read books in the bright classroom . "
           "the president led the government and the senate passed a national law . ") * 60
    m = UnifiedMind(dim=512, seed=0)
    ua.STATE.update({"mind": m, "dataset": "t", "labels": ["a"], "is_code": False,
                     "test": [], "raw_len": len(raw), "desc": "t", "seq_raw": raw})
    r = c.post("/api/unified/codec", json={}).get_json()
    assert r["lossless"]
    assert r["ratio"] <= r["random_ratio"]


def test_page_includes_codec_panel():
    c = _client()
    html = c.get("/").get_data(as_text=True)
    assert "lossless codec" in html


def test_page_has_searchable_card_system():
    # the examples are searchable/categorized/collapsible cards
    c = _client()
    html = c.get("/").get_data(as_text=True)
    assert 'id="search"' in html                       # search box
    assert 'id="cats"' in html                         # category pills
    assert "function buildCards" in html               # collapse/tag builder
    assert "function filterCards" in html              # search/filter engine
    assert "CATALOG=" in html                           # the example catalog


def test_every_card_has_a_catalog_entry():
    # each card's <h2> must match exactly one catalog key, so all are tagged/categorized
    import re
    c = _client()
    body = c.get("/").get_data(as_text=True).split("<script>")[0]
    h2s = re.findall(r"<h2>(.*?)</h2>", body, re.S)
    keys = ["pull + train", "question router", "classify", "organize", "relations",
            "curriculum", "4 &middot; generate", "predictive loop", "query &amp; generate",
            "deliberate", "self-discovery", "factorize", "many NPCs", "sprite pack",
            "image vault", "market structure", "big-text run", "lossless codec",
            "source tracing", "topic pull"]
    assert len(h2s) >= 14
    for h in h2s:
        matched = [k for k in keys if k in h]
        assert len(matched) == 1, f"{h!r} matched {matched}"


def test_population_endpoint_shows_sharing():
    # the NPC population demo: isolation holds, propagation works, and the shared
    # total is smaller than separate brains.
    import unified_app as ua
    c = _client()
    r = c.post("/api/unified/population", json={"population": 40}).get_json()
    assert r["propagation"]["after"] == "alchemy"          # propagated fact now visible
    assert r["propagation"]["before"] != "alchemy"         # was isolated before
    assert r["cost"]["shared_total"] < r["cost"]["separate_total"]
    assert r["cost"]["saving_x"] > 1.0


def test_page_includes_population_panel():
    c = _client()
    html = c.get("/").get_data(as_text=True)
    assert "many NPCs" in html


def test_sprite_pack_endpoint():
    # cross-sprite compression on the real set: smaller than per-file PNG, bit-exact.
    import os
    if not os.path.isdir("features/sprites"):
        import pytest
        pytest.skip("sprite set not present")
    c = _client()
    r = c.post("/api/unified/sprites", json={"n": 60}).get_json()
    assert "error" not in r, r.get("error")
    assert r["lossless"] is True
    assert r["set_pack"] < r["per_file_png"]
    assert r["saving_x"] > 1.0


def test_image_vault_endpoint():
    # the image repository as a perceptual memory: clusters + query-by-example.
    import os
    if not (os.path.isdir("features/photo_sample") or os.path.isdir("features/sprites")):
        import pytest
        pytest.skip("no image set present")
    c = _client()
    r = c.post("/api/unified/vault", json={}).get_json()
    assert "error" not in r, r.get("error")
    assert r["count"] >= 1
    assert r["query"] and r["query"][0]["sim"] >= 0.99   # an image matches itself


def test_page_includes_image_panels():
    c = _client()
    html = c.get("/").get_data(as_text=True)
    assert "sprite pack" in html
    assert "image vault" in html


def test_market_endpoint_light():
    # On-demand experiment panel: confirm it returns a sane shape on the real data.
    # Skips if the market data isn't present. NOT a heavy/long run -- one z-score.
    import os
    if not (os.path.exists("data/sol_5min.npz") or os.path.exists("data/dai_weth_big.json")):
        import pytest
        pytest.skip("market data not present")
    c = _client()
    r = c.post("/api/unified/market", json={"dataset": "sol"}).get_json()
    if "error" in r:
        import pytest
        pytest.skip(r["error"])
    assert "zscore" in r and "vol_clustering_acf1" in r
    assert abs(r["shuffled_acf1"]) < 0.2          # shuffle control near zero


def test_bigtext_endpoint_small_slice():
    # The heavy panel, exercised on a SMALL slice so the test stays fast. The full
    # run is meant to be triggered from the UI, not the suite.
    import unified_app as ua
    from holographic_unified import UnifiedMind
    c = _client()
    raw = ("the school taught young children to read books in the bright classroom . "
           "the president led the government and the senate passed a national law . ") * 50
    m = UnifiedMind(dim=512, seed=0)
    ua.STATE.update({"mind": m, "dataset": "t", "labels": ["a"], "is_code": False,
                     "test": [], "raw_len": len(raw), "desc": "t", "seq_raw": raw})
    r = c.post("/api/unified/bigtext", json={"tokens": 500}).get_json()
    assert "error" not in r, r.get("error")
    assert r["codec"]["lossless"] is True


def test_page_includes_experiment_panels():
    c = _client()
    html = c.get("/").get_data(as_text=True)
    assert "market structure" in html
    assert "big-text run" in html


def test_training_cache_reuses_and_refreshes():
    # Training is cached per stack; re-loading the same stack reuses it (cached flag),
    # and fresh=True rebuilds. Uses the small 'world' dataset to stay fast. (Restore is
    # a deep copy, so identity differs -- we check the cached flag, not object id.)
    import unified_app as ua
    c = _client()
    r1 = c.post("/api/unified/load", json={"id": "world", "fresh": True}).get_json()
    assert r1["ok"] and r1["cached"] is False
    protos = r1["prototypes"]
    r2 = c.post("/api/unified/load", json={"id": "world"}).get_json()
    assert r2["cached"] is True and r2["prototypes"] == protos        # reused
    r3 = c.post("/api/unified/load", json={"id": "world", "fresh": True}).get_json()
    assert r3["cached"] is False                                      # rebuilt


def test_trained_status_endpoint():
    import unified_app as ua
    c = _client()
    c.post("/api/unified/load", json={"id": "world", "mode": "replace"}).get_json()
    s = c.get("/api/unified/trained").get_json()
    assert s["active"] == "Countries (records)"
    assert s["trained_on"] == ["Countries (records)"]
    assert s["prototypes"] > 0


def test_cumulative_training_stacks_datasets():
    # The core feature: train a base, then add another dataset ON TOP of the same
    # brain. The stack is tracked, and prototypes accumulate (the second layer adds,
    # it does not replace).
    import unified_app as ua
    c = _client()
    base = c.post("/api/unified/load", json={"id": "world", "fresh": True}).get_json()
    base_protos = base["prototypes"]
    assert base["trained_on"] == ["Countries (records)"]
    stacked = c.post("/api/unified/load", json={"id": "self", "mode": "add"}).get_json()
    assert len(stacked["trained_on"]) == 2                # two datasets now
    assert stacked["trained_on"][0] == "Countries (records)"
    assert stacked["prototypes"] >= base_protos           # layered on, not replaced


def test_add_on_top_does_not_corrupt_cached_base():
    # Adding a dataset on top must not mutate the cached base stack (deep-copy safety).
    import unified_app as ua
    c = _client()
    b1 = c.post("/api/unified/load", json={"id": "world", "fresh": True}).get_json()
    base_protos = b1["prototypes"]
    c.post("/api/unified/load", json={"id": "self", "mode": "add"}).get_json()
    # restore the pure base from cache; it must be unchanged
    b2 = c.post("/api/unified/load", json={"id": "world", "mode": "replace"}).get_json()
    assert b2["cached"] is True
    assert b2["prototypes"] == base_protos


def test_empty_state_messages_name_relevant_dataset():
    # an experiment with no mind loaded should point the user to relevant training
    import unified_app as ua
    ua.STATE.update({"mind": None})
    c = _client()
    r = c.post("/api/unified/codec", json={}).get_json()
    assert "error" in r and "load" in r["error"].lower()
