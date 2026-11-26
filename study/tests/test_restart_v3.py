"""Research-integrity checks for the independent-topic restart protocol."""
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from owe.metamath import Statement
from owe.retrieval import metrics
from owe.restart_support import (
    GRAPH_COLUMNS, TEXT_COLUMNS, ProjectedEstimator, context_from_edges, local_context,
)
from owe.sequential_retriever import HybridEngine


@pytest.fixture(scope="module")
def restart_data():
    root = Path("data/processed/restart_v3")
    if not (root / "dataset.pkl").exists():
        pytest.skip("Build the independently verified restart data first")
    data = pickle.loads((root / "dataset.pkl").read_bytes())
    raw = pickle.loads((root / "parsed.pkl").read_bytes())
    return data, raw


@pytest.fixture(scope="module")
def restart_engine(restart_data):
    data, _ = restart_data
    config = json.loads(Path("configs/restart_v3.json").read_text())
    hidden = {n["signature"] for n in data["nodes"] if n["fresh_topic_core"]}
    hidden.update(q["signature"] for q in data["queries"] if q["split"] in ("policy_train", "validation"))
    texts = [n["text"] for n in data["nodes"] if n["signature"] not in hidden]
    # Fit independently here so checks do not depend on fitted v3 model artifacts.
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\S+", ngram_range=(1, 2), dtype=np.float32).fit(texts)
    engine = HybridEngine(data, vectorizer, config)
    clean = context_from_edges(data["nodes"], data["edges"], "test-restart-clean")
    return engine, clean, config["candidate_variants"][1]


def test_fresh_holdout_has_no_old_index_or_split_signature_overlap(restart_data):
    data, _ = restart_data
    old = pickle.loads(Path("data/processed/scoped/dataset.pkl").read_bytes())
    old_signatures = {n["signature"] for n in old["nodes"]}
    test = [q for q in data["queries"] if q["split"] == "test"]
    assert len(test) > 500
    assert {q["signature"] for q in test}.isdisjoint(old_signatures)
    assert len({q["signature"] for q in data["queries"]}) == len(data["queries"])
    splits = ["rank_train", "policy_train", "validation", "test"]
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            assert {q["group"] for q in data["queries"] if q["split"] == left}.isdisjoint(
                {q["group"] for q in data["queries"] if q["split"] == right})


def test_boundary_reconstructs_from_public_proofs_without_heldout_sources(restart_data):
    data, parsed = restart_data
    statements = parsed["statements"]
    assertions = set(parsed["assertions"])
    hidden = {n["signature"] for n in data["nodes"] if n["fresh_topic_core"]}
    hidden.update(q["signature"] for q in data["queries"] if q["split"] in ("policy_train", "validation"))
    core = {n["original_index"] for n in data["nodes"] if n["core"]}
    public = [n for n in data["nodes"] if n["core"] and n["signature"] not in hidden]
    # Reconstruct from checked raw proof dependencies, not saved boundary metadata.
    expected = {j for n in public for j in statements[n["original_index"]].deps if j in assertions and j not in core}
    actual = {n["original_index"] for n in data["nodes"] if not n["core"]}
    assert actual == expected
    assert all(data["nodes"][u]["signature"] not in hidden for u, _ in data["edges"])
    assert all(not data["nodes"][u]["fresh_topic_core"] for u, _ in data["edges"])
    local = {n["original_index"]: n["id"] for n in data["nodes"]}
    reconstructed = {(n["id"], local[j]) for n in public for j in statements[n["original_index"]].deps if j in assertions}
    assert reconstructed == set(map(tuple, data["edges"]))


def test_outside_index_gold_stays_in_full_recall_denominator(restart_data):
    data, parsed = restart_data
    statements = parsed["statements"]
    assertions = set(parsed["assertions"])
    local = {n["original_index"]: n["id"] for n in data["nodes"]}
    total = len(data["nodes"])
    for q in data["queries"]:
        original = sorted(j for j in statements[q["original_index"]].deps if j in assertions)
        assert sorted(q["gold_original_indices"]) == original
        expected = {local[j] if j in local else total + j for j in original}
        assert set(q["gold"]) == expected
        assert set(q["gold_in_index"]) == {local[j] for j in original if j in local}
        assert q["out_of_index_count"] == sum(j not in local for j in original)
    missing = next(q for q in data["queries"] if q["split"] == "test" and q["out_of_index_count"] > 0)
    all_available = metrics(missing["gold_in_index"], set(missing["gold"]), 10000)["recall"]
    assert all_available == len(missing["gold_in_index"]) / len(missing["gold"])
    assert all_available < 1


def test_projected_graph_free_estimator_ignores_graph_feature_changes():
    rng = np.random.default_rng(20261125)
    x = rng.normal(size=(120, 42))
    labels = (x[:, 0] + 0.8 * x[:, 1] > 0).astype(int)
    estimator = LogisticRegression(random_state=7).fit(x[:, TEXT_COLUMNS], labels)
    projected = ProjectedEstimator(estimator, TEXT_COLUMNS)
    changed = x.copy()
    changed[:, GRAPH_COLUMNS] = rng.normal(500, 300, size=(len(x), len(GRAPH_COLUMNS)))
    np.testing.assert_array_equal(projected.predict_proba(x), projected.predict_proba(changed))
    changed[:, 0] += 10
    assert not np.allclose(projected.predict_proba(x), projected.predict_proba(changed))


def test_local_stress_selection_is_gold_blind_and_respects_source_order(restart_engine):
    engine, clean, variant = restart_engine
    query = next(q for q in engine.data["queries"] if q["split"] == "rank_train" and q["topic"] == "graph")
    baseline = engine.retrieve(query, clean, variant)
    actual = local_context(engine, query, clean, variant, count=8)
    poisoned = {**query, "gold": [10**9], "gold_in_index": [], "gold_original_indices": [-1],
                "removal_benefits": np.array([np.nan]), "posthoc_oracle_effect": -999999}
    blind = local_context(engine, poisoned, clean, variant, count=8)
    assert actual["edges"] == blind["edges"]
    assert actual["audit_changes"] == blind["audit_changes"]
    additions = actual["edges"] - clean["edges"]
    assert 0 < len(additions) <= 8
    visited_sources = {u for u, _ in baseline.edges}
    for u, v in additions:
        assert u in visited_sources
        assert v < u < query["node"]
        assert (u, v) not in clean["edges"]
        assert engine.nodes[v]["core"] == engine.nodes[u]["core"]
        assert engine.nodes[v]["signature"] != query["signature"]


def test_restart_suppression_matches_a_materialized_graph_edit(restart_engine):
    engine, clean, variant = restart_engine
    queries = [q for q in engine.data["queries"] if q["split"] == "rank_train" and q["topic"] == "graph"]
    for q in queries[::max(1, len(queries)//3)][:3]:
        context = local_context(engine, q, clean, variant, count=8)
        before = engine.retrieve(q, context, variant)
        removed = list(before.edges[:3])
        assert removed
        suppressed = engine.retrieve(q, context, variant, removed)
        edited_context = context_from_edges(engine.nodes, context["edges"] - set(removed), "materialized-v3-" + q["qid"])
        materialized = engine.retrieve(q, edited_context, variant)
        np.testing.assert_array_equal(suppressed.ids, materialized.ids)
        np.testing.assert_allclose(suppressed.features, materialized.features, rtol=0, atol=1e-7)
        assert suppressed.edges == materialized.edges
        assert suppressed.transitions == materialized.transitions
