"""Post-hoc stage decomposition and mechanism checks for the frozen retriever."""
from __future__ import annotations
from collections import Counter, deque
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import cluster_interval
from .experiment import Study, write_json
from .metamath import Statement
from .retrieval import metrics, ranked, traverse


COMPONENTS = ["recalled", "ranking_loss", "capacity_loss", "beyond_two_hops", "no_directed_path", "excluded_by_rules"]


def distances(adjacency, seeds):
    distance = {int(u): 0 for u in seeds}
    queue = deque(distance)
    edge_visits = 0
    while queue:
        u = queue.popleft()
        neighbors = adjacency[u]
        edge_visits += len(neighbors)
        for v in neighbors:
            if v not in distance:
                distance[v] = distance[u] + 1
                queue.append(v)
    return distance, edge_visits


class UndirectedPriorAdjacency:
    def __init__(self, context, cutoff, forbidden):
        self.context = context
        self.cutoff = cutoff
        self.forbidden = forbidden

    def __getitem__(self, u):
        return sorted({v for v in self.context["adjacency"][u] + self.context["incoming"][u]
                       if v < self.cutoff and v not in self.forbidden})


def stage_row(query, context, study):
    neighborhood = study.neighborhood(query, context)
    scores = study.query_scores(query)
    gold = set(query["gold"])
    eligible = {v for v in gold if np.isfinite(scores[v])}
    distance, _ = distances(neighborhood.adjacency, neighborhood.seeds)
    reached_gold = gold & neighborhood.reached
    actual_ids = neighborhood.ranking[:20]
    found = gold & set(actual_ids)
    maximum = min(20, len(reached_gold))
    m = len(gold)
    counts = {"recalled": len(found), "ranking_loss": maximum - len(found),
              "capacity_loss": len(reached_gold) - maximum,
              "beyond_two_hops": sum(v in distance and distance[v] > 2 for v in eligible),
              "no_directed_path": sum(v not in distance for v in eligible),
              "excluded_by_rules": len(gold - eligible)}
    assert sum(counts.values()) == m
    assert all(value >= 0 for value in counts.values())
    assert set(distance) >= neighborhood.reached
    assert list(map(int, actual_ids[:4])) == list(map(int, ranked(np.flatnonzero(np.isfinite(scores)), scores)[:4]))
    if len(neighborhood.reached) <= 20:
        for edge in neighborhood.edges:
            assert metrics(neighborhood.remove_one(edge), gold, 20)["recall"] <= len(found) / m + 1e-12
    row = {"qid": query["qid"], "group": query["group"], "topic": query["topic"],
           "section": query["section"], "condition": context["name"], "gold_count": m,
           **{key: value / m for key, value in counts.items()},
           "two_hop_gold_coverage": len(reached_gold) / m,
           "perfect_reranker_recall20": maximum / m,
           "seed_recall": len(gold & set(neighborhood.seeds)) / m,
           "seed_gold_count": len(gold & set(neighborhood.seeds)),
           "zero_gold_seeds": int(not gold.intersection(neighborhood.seeds)),
           "zero_hits": int(not found), "complete": int(len(found) == m),
           "zero_traversal": int(not neighborhood.edges),
           "no_more_than_20_candidates": int(len(neighborhood.reached) <= 20),
           "reached_nodes": len(neighborhood.reached), "full_reached_nodes": len(distance),
           "visited_edges": len(neighborhood.edges), "returned": len(actual_ids)}
    return row, neighborhood, distance


def main():
    start = time.perf_counter()
    root = Path("runs/retriever_diagnostic_v1")
    root.mkdir(parents=True, exist_ok=True)
    source = Path("runs/scoped_v1")
    data = pickle.loads(Path("data/processed/scoped/dataset.pkl").read_bytes())
    contexts = pickle.loads((source / "graphs.pkl").read_bytes())
    vectorizer = pickle.loads((source / "vectorizer.pkl").read_bytes())
    matrix = vectorizer.transform([n["text"] for n in data["nodes"]]).tocsr()
    study = Study(data, matrix, contexts, root)
    queries = [q for q in data["queries"] if q["split"] == "test"]
    hidden_signatures = {q["signature"] for q in data["queries"] if q["split"] != "train"}
    original_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for path, digest in original_manifest["code_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    write_json(root / "manifest.json", {"analysis": "post-hoc mechanism diagnosis; no model tuning",
        "source_run": str(source), "source_commit": data["config"]["source_commit"],
        "diagnostic_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(Path("RETRIEVER_DIAGNOSTIC_PROTOCOL.md").read_bytes()).hexdigest(),
        "input_dataset_sha256": hashlib.sha256(Path("data/processed/scoped/dataset.pkl").read_bytes()).hexdigest(),
        "vectorizer": {"analyzer": "word tokens split by whitespace", "ngrams": [1, 2],
                       "dimensions": matrix.shape[1], "lowercase": False},
        "seed_grid": [1, 4, 8, 16, 20], "hop_grid": [0, 1, 2, 3, 4, "all"]})
    saved_cases = pickle.loads((source / "test_cases.pkl").read_bytes())
    saved = {(c["qid"], c["condition"]): c for c in saved_cases}
    stage_rows = []
    ablations = []
    seeds_log = []
    rank_records = []
    actual_records = []
    for context_name, context in contexts.items():
        for query in queries:
            row, neighborhood, distance = stage_row(query, context, study)
            stage_rows.append(row)
            assert list(map(int, neighborhood.ranking[:40])) == list(map(int, saved[query["qid"], context_name]["base_ids"]))
            if context_name != "clean":
                continue
            scores = study.query_scores(query)
            lexical = ranked(np.flatnonzero(np.isfinite(scores)), scores)
            global_rank = {int(v): j + 1 for j, v in enumerate(lexical)}
            gold = set(query["gold"])
            categories = []
            for position, node in enumerate(lexical[:4], start=1):
                record = data["nodes"][int(node)]
                if not record["core"]:
                    category = "boundary_terminal"
                elif record["signature"] in hidden_signatures:
                    category = "heldout_proof_hidden"
                elif not context["adjacency"][int(node)]:
                    category = "intrinsic_terminal"
                else:
                    category = "expandable"
                categories.append(category)
                seeds_log.append({"qid": query["qid"], "group": query["group"], "position": position,
                    "seed_id": int(node), "seed_label": record["label"], "category": category,
                    "is_gold": int(node in gold), "outdegree": len(context["adjacency"][int(node)]),
                    "query_zero_traversal": row["zero_traversal"]})
            for v in gold:
                rank_records.append({"qid": query["qid"], "group": query["group"], "gold_id": v,
                    "gold_label": data["nodes"][v]["label"], "core": data["nodes"][v]["core"],
                    "global_tfidf_rank": global_rank.get(v), "query_similarity": float(scores[v]) if np.isfinite(scores[v]) else None,
                    "distance_from_seeds": distance.get(v), "returned20": int(v in set(neighborhood.ranking[:20]))})
            variants = [(f"seeds4_hops{hop}", 4, hop, neighborhood.adjacency) for hop in [0, 1, 2, 3, 4, "all"]]
            variants += [(f"seeds{n}_hops2", n, 2, neighborhood.adjacency) for n in [1, 8, 16, 20]]
            variants += [("seeds4_undirected_hops2", 4, 2,
                          UndirectedPriorAdjacency(context, query["node"], study.signature_nodes[query["signature"]]))]
            for name, seed_count, hops, adjacency in variants:
                seeds = lexical[:seed_count]
                if hops == "all":
                    reached, edge_visits = set(distance), sum(len(adjacency[u]) for u in distance)
                else:
                    reached = traverse(adjacency, seeds, hops)
                    # Count expanded adjacency entries, with each shortest-depth source expanded once.
                    earlier = traverse(adjacency, seeds, max(0, hops - 1)) if hops else set()
                    edge_visits = sum(len(adjacency[u]) for u in earlier)
                ids = ranked(reached, scores)[:20]
                assert all(int(v) < query["node"] for v in ids)
                assert all(data["nodes"][int(v)]["signature"] != query["signature"] for v in reached)
                if seed_count == 20:
                    assert list(map(int, ids)) == list(map(int, lexical[:20]))
                coverage = len(gold & reached) / len(gold)
                ablations.append({"qid": query["qid"], "group": query["group"], "topic": query["topic"],
                    "variant": name, **metrics(ids, gold, 20, study.lengths),
                    "coverage": coverage, "perfect_reranker_recall20": min(20, len(gold & reached)) / len(gold),
                    "reached_nodes": len(reached), "expanded_edge_entries": edge_visits})
            if "Eulerian" in query["chapter"] or query["qid"] in ["usgrfs", "pthontrlon"]:
                actual_records.append({"qid": query["qid"], "source": data["nodes"][query["node"]]["source"],
                    "query_text": data["nodes"][query["node"]]["text"], "gold_count": len(gold),
                    "recall20": row["recalled"], "coverage": row["two_hop_gold_coverage"],
                    "perfect_reranker_recall20": row["perfect_reranker_recall20"],
                    "seed_categories": categories,
                    "seeds": [{"label": data["nodes"][int(u)]["label"], "is_gold": int(u in gold),
                               "similarity": float(scores[u]), "outdegree": len(context["adjacency"][int(u)])} for u in lexical[:4]],
                    "premises": [{"label": data["nodes"][v]["label"], "global_rank": global_rank.get(v),
                        "distance": distance.get(v), "returned": v in set(neighborhood.ranking[:20]),
                        "query_similarity": float(scores[v]) if np.isfinite(scores[v]) else None,
                        "text": data["nodes"][v]["text"]} for v in sorted(gold)]})
        print(f"Diagnosed {context_name}: {len(queries)} queries", flush=True)
    stages = pd.DataFrame(stage_rows)
    ablation = pd.DataFrame(ablations)
    stages.to_csv(root / "stage_decomposition.csv", index=False)
    ablation.to_csv(root / "mechanism_ablations.csv", index=False)
    seed_frame = pd.DataFrame(seeds_log)
    seed_frame.to_csv(root / "seed_types.csv", index=False)
    pd.DataFrame(rank_records).to_csv(root / "gold_global_ranks.csv", index=False)
    write_json(root / "euler_and_failure_examples.json", actual_records)

    frames = {name: frame.copy() for name, frame in stages.groupby("condition")}
    random = stages[stages.condition.isin(["random20_s17", "random20_s29", "random20_s43"])]
    keys = ["qid", "group", "topic", "section"]
    numeric = list(random.select_dtypes(include="number").columns)
    frames["random20_mean"] = random.groupby(keys, as_index=False)[numeric].mean()
    summary = []
    for name, frame in frames.items():
        for column in COMPONENTS + ["two_hop_gold_coverage", "perfect_reranker_recall20", "seed_recall"]:
            lo, hi = cluster_interval(frame, column)
            summary.append({"condition": name, "metric": column, "mean": frame[column].mean(),
                            "ci_low": lo, "ci_high": hi})
    pd.DataFrame(summary).to_csv(root / "stage_summary.csv", index=False)
    clean = stages[stages.condition == "clean"].set_index("qid")
    variants = []
    for name, frame in ablation.groupby("variant"):
        frame = frame.copy()
        frame["gain"] = frame.qid.map(clean.recalled).rsub(frame.recall)
        lo, hi = cluster_interval(frame, "gain")
        variants.append({"variant": name, "recall20": frame.recall.mean(), "delta_vs_original": frame.gain.mean(),
            "ci_low": lo, "ci_high": hi, "coverage": frame.coverage.mean(),
            "perfect_reranker_recall20": frame.perfect_reranker_recall20.mean(),
            "mean_candidates": frame.reached_nodes.mean(), "mean_expanded_edge_entries": frame.expanded_edge_entries.mean(),
            "complete": frame.complete.mean()})
    variant_summary = pd.DataFrame(variants)
    variant_summary.to_csv(root / "mechanism_summary.csv", index=False)
    zero_cases = seed_frame[seed_frame.query_zero_traversal == 1]
    profile = {"queries": len(queries), "source_blocks": clean.group.nunique(),
        "zero_traversal_queries": int(clean.zero_traversal.sum()),
        "zero_gold_seed_queries": int(clean.zero_gold_seeds.sum()),
        "zero_hits_queries": int(clean.zero_hits.sum()),
        "no_more_than_20_candidate_queries": int(clean.no_more_than_20_candidates.sum()),
        "seed_types_all": {str(k): int(v) for k, v in seed_frame.category.value_counts().items()},
        "seed_types_zero_traversal": {str(k): int(v) for k, v in zero_cases.category.value_counts().items()},
        "zero_traversal_category_combinations": dict(Counter(
            "+".join(sorted(set(f.category))) for _, f in zero_cases.groupby("qid"))),
        "background_vocabulary_dimensions": matrix.shape[1],
        "invariants": {"saved_baselines_match": True, "first_four_results_are_fixed_seeds": True,
                       "seeds20_top20_equals_global_lexical": True, "deletion_cannot_improve_underfilled_candidates": True,
                       "no_future_or_same_signature_nodes_in_diagnostic_results": True},
        "seconds": time.perf_counter() - start}
    write_json(root / "profile.json", profile)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    colors = ["#1b9e77", "#e6ab02", "#d95f02", "#7570b3", "#999999", "#e7298a"]
    conditions = ["clean", "random20_mean", "hard20_s17", "missing20_s17"]
    labels = ["Retrieved", "Ranking loss", "Capacity limit", ">2 hops", "No directed path", "Excluded by rules"]
    left = np.zeros(len(conditions))
    for column, label, color in zip(COMPONENTS, labels, colors):
        vals = np.array([frames[c][column].mean() * 100 for c in conditions])
        ax.barh(np.arange(len(conditions)), vals, left=left, label=label, color=color)
        left += vals
    ax.set_yticks(np.arange(len(conditions)), ["Clean", "Random +20% (3 seeds)", "Lexical-hard +20%", "Missing 20%"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Mean per-query share of source-proof gold citations (%)")
    ax.set_title("Where the frozen local retriever loses gold citations")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.44), ncol=3, fontsize=9)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    for suffix in ["png", "svg"]:
        fig.savefig(root / ("failure_decomposition." + suffix), dpi=220)
    plt.close(fig)
    print(pd.DataFrame(summary).query("condition == 'clean'").to_string(index=False))
    print(variant_summary.to_string(index=False))
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
