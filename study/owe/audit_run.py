"""Independent replays of saved interventions, data boundaries and provenance."""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .experiment import Study, choose_edges, predict, write_json
from .retrieval import metrics, ranked, traverse


def main():
    root = Path("runs/scoped_v1")
    data = pickle.loads(Path("data/processed/scoped/dataset.pkl").read_bytes())
    contexts = pickle.loads((root / "graphs.pkl").read_bytes())
    vectorizer = pickle.loads((root / "vectorizer.pkl").read_bytes())
    matrix = vectorizer.transform([n["text"] for n in data["nodes"]]).tocsr()
    study = Study(data, matrix, contexts, root)
    queries = {q["qid"]: q for q in data["queries"]}
    cases = pickle.loads((root / "test_cases.pkl").read_bytes())
    models = pickle.loads((root / "models.pkl").read_bytes())
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["code_sha256"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
    for filename, expected in manifest["protocol_lock"]["hashes"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
    reference = set(map(tuple, data["edges"]))
    hidden = {q["signature"] for q in data["queries"] if q["split"] != "train"}
    for name, context in contexts.items():
        assert all(u > v for u, v in context["edges"])
        assert all(data["nodes"][u]["signature"] not in hidden for u, _ in context["edges"])
        for action, u, v in context["audit_changes"]:
            assert ((u, v) in reference) == (action == "removed")
            assert ((u, v) in context["edges"]) == (action == "added")
    saved_retrievals = {}
    with gzip.open(root / "retrievals.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            saved_retrievals[record["qid"], record["condition"]] = record
    rng = np.random.default_rng(9981)
    replayed_labels = 0
    replayed_policies = 0
    sampled = rng.choice(len(cases), size=min(250, len(cases)), replace=False)
    for index in sampled:
        case = cases[int(index)]
        query = queries[case["qid"]]
        context = contexts[case["condition"]]
        neighborhood = study.neighborhood(query, context)
        scores = study.query_scores(query)
        gold = set(query["gold"])
        np.testing.assert_array_equal(case["features"], study.edge_features(query, context, neighborhood, case["edges"]))
        assert set(case["base_ids"]) <= traverse(neighborhood.adjacency, neighborhood.seeds)
        for j, edge in enumerate(case["edges"]):
            actual = traverse(neighborhood.adjacency, neighborhood.seeds, deleted=frozenset([edge]))
            ids = ranked(actual, scores)[:40]
            for k_index, budget in enumerate(data["config"]["output_budgets"]):
                effect = metrics(ids, gold, budget)["recall"] - metrics(case["base_ids"], gold, budget)["recall"]
                assert abs(effect - float(case["benefits"][j, k_index])) < 5e-8
            replayed_labels += 1
        rec = saved_retrievals[query["qid"], context["name"]]
        for name, policy in selection.items():
            prediction = predict(models[policy["model"]], case["features"])
            actions = choose_edges(case, prediction, policy["budget"], policy["threshold"])
            assert [list(e) for e in actions] == rec["actions"][name]
            actual = traverse(neighborhood.adjacency, neighborhood.seeds, deleted=frozenset(actions))
            assert list(map(int, ranked(actual, scores)[:40])) == rec["ranks"][name]
            replayed_policies += 1
    results = pd.read_csv(root / "results.csv")
    assert not results.duplicated(["qid", "condition", "method", "budget"]).any()
    assert results.qid.nunique() == data["stats"]["query_splits"]["test"]
    assert len(results) == 567 * 7 * 13 * 3
    write_json(root / "replay_audit.json", {"all_checks_passed": True,
        "sampled_graph_query_cases": len(sampled), "single_edge_interventions_replayed_with_BFS": replayed_labels,
        "learned_policy_outputs_replayed_with_BFS": replayed_policies,
        "protocol_and_original_code_hashes_match": True, "all_graph_provenance_and_hidden_proof_checks_pass": True,
        "result_rows": len(results), "random_seed": 9981})
    print((root / "replay_audit.json").read_text())


if __name__ == "__main__":
    main()
