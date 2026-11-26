"""Post-hoc output-count and formal-token matched lexical diagnostics."""
from __future__ import annotations
import gzip
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .analyze import cluster_interval
from .retrieval import metrics, ranked


def main():
    root = Path("runs/scoped_v1")
    data = pickle.loads(Path("data/processed/scoped/dataset.pkl").read_bytes())
    vectorizer = pickle.loads((root / "vectorizer.pkl").read_bytes())
    matrix = vectorizer.transform([n["text"] for n in data["nodes"]]).tocsr()
    lengths = np.asarray([len(n["text"].split()) for n in data["nodes"]])
    queries = {q["qid"]: q for q in data["queries"]}
    original = pd.read_csv(root / "results.csv")
    originals = {(r.qid, r.condition, r.method, r.budget): r.recall
                 for r in original.itertuples() if r.method in ["no_op", "learned3"]}
    rows = []
    cache = {}
    with gzip.open(root / "retrievals.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            query = queries[rec["qid"]]
            if query["qid"] not in cache:
                scores = (matrix @ matrix[query["node"]].T).toarray().ravel()
                eligible = [n["id"] for n in data["nodes"] if n["id"] < query["node"] and n["signature"] != query["signature"]]
                cache[query["qid"]] = ranked(eligible, scores)
            lexical = cache[query["qid"]]
            for budget in data["config"]["output_budgets"]:
                baseline = rec["ranks"]["no_op"][:budget]
                count_matched = lexical[:len(baseline)]
                tokens = int(lengths[baseline].sum())
                remaining = tokens
                token_matched = []
                for node in lexical:
                    if len(token_matched) >= budget:
                        break
                    if lengths[node] <= remaining:
                        token_matched.append(int(node))
                        remaining -= int(lengths[node])
                for method, ids in [("lexical_count_matched", count_matched), ("lexical_token_matched", token_matched)]:
                    measured = metrics(np.asarray(ids, dtype=int), set(query["gold"]), budget, lengths)
                    rows.append({"qid": query["qid"], "group": query["group"], "topic": query["topic"],
                        "condition": rec["condition"], "method": method, "budget": budget,
                        **measured, "delta_vs_graph": measured["recall"] - originals[query["qid"], rec["condition"], "no_op", budget],
                        "delta_vs_learned3": measured["recall"] - originals[query["qid"], rec["condition"], "learned3", budget],
                        "graph_returned": len(baseline), "graph_formal_tokens": tokens})
                    if method.endswith("count_matched"):
                        assert measured["returned"] == len(baseline)
                    else:
                        assert measured["formal_tokens"] <= tokens
    results = pd.DataFrame(rows)
    results.to_csv(root / "matched_budgets.csv", index=False)
    frames = {name: frame for name, frame in results.groupby("condition")}
    for name, conditions in [("random20_mean", ["random20_s17", "random20_s29", "random20_s43"]),
                             ("random20_unseen_mean", ["random20_s29", "random20_s43"])]:
        subset = results[results.condition.isin(conditions)]
        keys = ["qid", "group", "topic", "method", "budget"]
        numerical = list(subset.select_dtypes(include="number").columns.difference(["budget"]))
        frames[name] = subset.groupby(keys, as_index=False)[numerical].mean()
    summary = []
    for condition, frame in frames.items():
        for (method, budget), sub in frame.groupby(["method", "budget"]):
            low, high = cluster_interval(sub, "delta_vs_graph")
            summary.append({"condition": condition, "method": method, "budget": int(budget),
                "queries": len(sub), "recall": sub.recall.mean(),
                "delta_vs_graph": sub.delta_vs_graph.mean(), "ci_low": low, "ci_high": high,
                "mean_returned": sub.returned.mean(), "mean_formal_tokens": sub.formal_tokens.mean(),
                "graph_mean_returned": sub.graph_returned.mean(), "graph_mean_formal_tokens": sub.graph_formal_tokens.mean()})
    summary = pd.DataFrame(summary)
    summary.to_csv(root / "matched_budget_summary.csv", index=False)
    print(summary[(summary.budget == 20) & summary.condition.isin(["clean", "random20_mean", "hard20_s17", "missing20_s17"])].to_string(index=False))


if __name__ == "__main__":
    main()
