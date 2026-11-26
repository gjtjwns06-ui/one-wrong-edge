"""Cross-check the diagnostic decomposition using SciPy shortest paths."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from .diagnose_retriever import COMPONENTS
from .experiment import Study, write_json
from .metamath import Statement
from .retrieval import ranked


def main():
    root = Path("runs/retriever_diagnostic_v1")
    original = Path("runs/scoped_v1")
    data = pickle.loads(Path("data/processed/scoped/dataset.pkl").read_bytes())
    contexts = pickle.loads((original / "graphs.pkl").read_bytes())
    vectorizer = pickle.loads((original / "vectorizer.pkl").read_bytes())
    matrix = vectorizer.transform([n["text"] for n in data["nodes"]]).tocsr()
    study = Study(data, matrix, contexts, root)
    queries = {q["qid"]: q for q in data["queries"]}
    decomposition = pd.read_csv(root / "stage_decomposition.csv")
    results = pd.read_csv(original / "results.csv")
    baseline = results[(results.method == "no_op") & (results.budget == 20)]
    merged = decomposition.merge(baseline[["qid", "condition", "recall"]], on=["qid", "condition"], validate="one_to_one")
    assert len(merged) == len(decomposition) == len(baseline)
    np.testing.assert_allclose(merged.recalled, merged.recall, atol=1e-14, rtol=0)
    np.testing.assert_allclose(decomposition[COMPONENTS].sum(axis=1), 1, atol=1e-14, rtol=0)
    assert decomposition[COMPONENTS].ge(-1e-14).all().all()
    sampled = decomposition.sample(n=40, random_state=4281)
    for row in sampled.itertuples():
        query = queries[row.qid]
        context = contexts[row.condition]
        scores = study.query_scores(query)
        eligible = np.isfinite(scores)
        edge_list = [(u, v) for u, v in context["edges"] if eligible[u] and eligible[v]]
        rr, cc = zip(*edge_list)
        graph = csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(len(data["nodes"]), len(data["nodes"])))
        seeds = ranked(np.flatnonzero(eligible), scores)[:4]
        distances = shortest_path(graph, directed=True, indices=seeds, unweighted=True).min(axis=0)
        gold = np.asarray(query["gold"])
        np.testing.assert_allclose(np.mean(distances[gold] <= 2), row.two_hop_gold_coverage, atol=1e-14, rtol=0)
        np.testing.assert_allclose(np.mean(np.isfinite(distances[gold]) & (distances[gold] > 2)), row.beyond_two_hops, atol=1e-14, rtol=0)
        np.testing.assert_allclose(np.mean(~np.isfinite(distances[gold])), row.no_directed_path, atol=1e-14, rtol=0)
    full = pickle.loads(Path("data/processed/set.pkl").read_bytes())
    assertions = set(full["assertions"])
    seed_log = pd.read_csv(root / "seed_types.csv")
    hidden = seed_log[seed_log.category == "heldout_proof_hidden"]
    for u in hidden.seed_id.unique():
        original_statement = full["statements"][data["nodes"][int(u)]["original_index"]]
        assert any(v in assertions for v in original_statement.deps)
    # Type-weighted losses: each question receives equal total weight.
    ranks = pd.read_csv(root / "gold_global_ranks.csv")
    ranks["weight"] = ranks.qid.map({q: 1 / len(v["gold"]) for q, v in queries.items()}) / len(baseline[baseline.condition == "clean"])
    ranks["not_reached_in_two"] = ranks.distance_from_seeds.isna() | (ranks.distance_from_seeds > 2)
    by_type = []
    for core, frame in ranks.groupby("core"):
        by_type.append({"type": "core" if core else "boundary", "gold_share_macro": frame.weight.sum(),
                       "retrieved_share_macro": frame.loc[frame.returned20 == 1, "weight"].sum(),
                       "outside_two_hops_share_macro": frame.loc[frame.not_reached_in_two, "weight"].sum()})
    pd.DataFrame(by_type).to_csv(root / "gold_type_contributions.csv", index=False)
    write_json(root / "diagnostic_checks.json", {"all_passed": True,
        "original_baseline_rows_matched": len(merged), "decomposition_rows_sum_to_one": True,
        "independent_scipy_shortest_path_cases": len(sampled),
        "hidden_seed_positions_with_actual_source_proof_edges": len(hidden),
        "hidden_unique_seed_nodes_checked": int(hidden.seed_id.nunique()),
        "original_run_results_sha256": hashlib.sha256((original / "results.csv").read_bytes()).hexdigest(),
        "diagnostic_code_sha256": hashlib.sha256(Path("owe/diagnose_retriever.py").read_bytes()).hexdigest()})
    print((root / "diagnostic_checks.json").read_text())
    print(pd.DataFrame(by_type).to_string(index=False))


if __name__ == "__main__":
    main()
