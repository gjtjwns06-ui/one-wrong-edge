"""Cluster uncertainty, failure audits and publication-style static figures."""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiment import FEATURE_NAMES, predict, write_json


def cluster_interval(frame, column, repetitions=2000, seed=811):
    grouped = frame.groupby("group")[column].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(grouped), np.full(len(grouped), 1 / len(grouped)), size=repetitions)
    values = (weights @ grouped["sum"].to_numpy()) / (weights @ grouped["count"].to_numpy())
    return list(map(float, np.quantile(values, [0.025, 0.975])))


def condition_frames(results):
    frames = {name: frame.copy() for name, frame in results.groupby("condition")}
    for name, conds in [("random20_mean", ["random20_s17", "random20_s29", "random20_s43"]),
                        ("random20_unseen_mean", ["random20_s29", "random20_s43"])]:
        subset = results[results.condition.isin(conds)]
        keys = ["qid", "group", "topic", "section", "method", "budget"]
        numerical = subset.select_dtypes(include="number").columns.difference(["budget"])
        pooled = subset.groupby(keys, as_index=False)[list(numerical)].mean()
        pooled["condition"] = name
        frames[name] = pooled
    return frames


def main():
    root = Path("runs/scoped_v1")
    if not (root / "run_complete.json").exists():
        raise RuntimeError("Run the locked experiment to completion before analysis")
    data = pickle.loads(Path("data/processed/scoped/dataset.pkl").read_bytes())
    config = data["config"]
    results = pd.read_csv(root / "results.csv")
    frames = condition_frames(results)
    summaries = []
    subgroups = []
    for condition, frame in frames.items():
        for (method, budget), sub in frame.groupby(["method", "budget"]):
            lo, hi = cluster_interval(sub, "delta_recall", config["bootstrap_repetitions"], config["bootstrap_seed"])
            summaries.append({"condition": condition, "method": method, "budget": int(budget),
                "queries": len(sub), "source_blocks": sub.group.nunique(), "recall": sub.recall.mean(),
                "delta_recall": sub.delta_recall.mean(), "delta_ci_low": lo, "delta_ci_high": hi,
                "complete": sub.complete.mean(), "ndcg": sub.ndcg.mean(),
                "improved_fraction": (sub.delta_recall > 1e-8).mean(),
                "harmed_fraction": (sub.delta_recall < -1e-8).mean(),
                "mean_deletions": sub.deletions.mean(), "action_fraction": (sub.deletions > 0).mean(),
                "mean_returned": sub.returned.mean(), "mean_formal_tokens": sub.formal_tokens.mean(),
                "mean_visited_edges": sub.visited_edges.mean(), "candidate_coverage": sub.candidate_coverage.mean(),
                "core_recall": sub.core_recall.mean(), "boundary_recall": sub.boundary_recall.mean()})
            if budget == config["primary_budget"] and method in ["no_op", "lexical", "learned1", "learned3", "source_verified", "oracle1"]:
                for topic, topic_frame in sub.groupby("topic"):
                    lower, upper = cluster_interval(topic_frame, "delta_recall", config["bootstrap_repetitions"], config["bootstrap_seed"])
                    subgroups.append({"condition": condition, "method": method, "topic": topic,
                        "queries": len(topic_frame), "blocks": topic_frame.group.nunique(),
                        "recall": topic_frame.recall.mean(), "delta_recall": topic_frame.delta_recall.mean(),
                        "ci_low": lower, "ci_high": upper})
    summary = pd.DataFrame(summaries)
    summary.to_csv(root / "summary.csv", index=False)
    pd.DataFrame(subgroups).to_csv(root / "topic_summary.csv", index=False)
    comparisons = []
    for condition, frame in frames.items():
        sub = frame[frame.budget == 20]
        for first, second in [("learned1", "edge_only1"), ("learned3", "source_verified"),
                              ("learned1", "matched_random1"), ("learned3", "matched_random3"),
                              ("learned3", "lexical")]:
            a = sub[sub.method == first][["qid", "group", "recall"]]
            b = sub[sub.method == second][["qid", "recall"]]
            pair = a.merge(b, on="qid", suffixes=("_a", "_b"))
            pair["difference"] = pair.recall_a - pair.recall_b
            lo, hi = cluster_interval(pair, "difference", config["bootstrap_repetitions"], config["bootstrap_seed"])
            comparisons.append({"condition": condition, "comparison": first + " minus " + second,
                                "difference": pair.difference.mean(), "ci_low": lo, "ci_high": hi})
    pd.DataFrame(comparisons).to_csv(root / "comparisons.csv", index=False)

    effects = pd.read_csv(root / "single_interventions.csv.gz")
    effects["positive"] = effects.benefit20 > 1e-8
    effects["negative"] = effects.benefit20 < -1e-8
    effects["zero"] = ~(effects.positive | effects.negative)
    effects.groupby(["condition", "source_supported"])[["positive", "negative", "zero"]].agg(["sum", "mean"]).to_csv(root / "effect_distribution.csv")
    reusable = effects.groupby(["condition", "source", "target", "source_supported"]).agg(
        queries=("qid", "nunique"), positive=("positive", "max"), negative=("negative", "max"), zero=("zero", "max"))
    reusable = reusable[reusable.queries >= 2]
    sign_stats = {"reused_edge_condition_pairs": len(reusable),
                  "both_positive_and_negative": int((reusable.positive & reusable.negative).sum()),
                  "positive_and_zero": int((reusable.positive & reusable.zero).sum()),
                  "negative_and_zero": int((reusable.negative & reusable.zero).sum())}
    write_json(root / "query_conditioning.json", sign_stats)

    # Structured examples are generated from actual saved actions and rankings.
    retrieval_map = {}
    with gzip.open(root / "retrievals.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            retrieval_map[record["qid"], record["condition"]] = record
    queries = {q["qid"]: q for q in data["queries"]}
    reference = set(map(tuple, data["edges"]))
    examples = []
    primary = results[(results.budget == 20) & (results.method == "learned3")]
    for label, selected in [("harm", primary.sort_values("delta_recall").drop_duplicates("qid").head(5)),
                             ("benefit", primary.sort_values("delta_recall", ascending=False).drop_duplicates("qid").head(5))]:
        for row in selected.itertuples():
            if abs(row.delta_recall) <= 1e-8:
                continue
            query = queries[row.qid]
            rec = retrieval_map[row.qid, row.condition]
            before = set(rec["ranks"]["no_op"][:20])
            after = set(rec["ranks"]["learned3"][:20])
            gold = set(query["gold"])
            examples.append({"kind": label, "qid": row.qid, "condition": row.condition,
                "delta_recall20": row.delta_recall, "gold_count": len(gold), "topic": row.topic,
                "query": data["nodes"][query["node"]]["text"], "source": data["nodes"][query["node"]]["source"],
                "actions": [{"source": data["nodes"][u]["label"], "target": data["nodes"][v]["label"],
                             "source_supported": (u, v) in reference} for u, v in rec["actions"]["learned3"]],
                "lost_gold": [data["nodes"][v]["label"] for v in sorted((before - after) & gold)],
                "gained_gold": [data["nodes"][v]["label"] for v in sorted((after - before) & gold)]})
    write_json(root / "failure_and_success_cases.json", examples)
    example_lines = ["# Actual intervention cases", "", "All labels refer to one stored proof, not every possible proof.", ""]
    for e in examples:
        example_lines.extend([f"## {e['kind']}: {e['qid']} / {e['condition']}", "",
            f"Recall@20 change: {100*e['delta_recall20']:+.3f} percentage points; {e['gold_count']} direct premises.", "",
            f"[Source frame]({e['source']}). Lost gold: {', '.join(e['lost_gold']) or 'none'}. Gained gold: {', '.join(e['gained_gold']) or 'none'}.", "",
            "Deleted relations: " + "; ".join(f"{a['source']} → {a['target']} (source-supported={a['source_supported']})" for a in e["actions"]), ""])
    (root / "failure_cases.md").write_text("\n".join(example_lines), encoding="utf-8")

    full_switches = reusable[reusable.positive & reusable.negative].reset_index().head(5)
    switch_examples = []
    for e in full_switches.itertuples():
        sub = effects[(effects.condition == e.condition) & (effects.source == e.source) & (effects.target == e.target)]
        low, high = sub.loc[sub.benefit20.idxmin()], sub.loc[sub.benefit20.idxmax()]
        switch_examples.append({"condition": e.condition, "source": data["nodes"][e.source]["label"],
            "target": data["nodes"][e.target]["label"], "source_supported": bool(e.source_supported),
            "positive_query": high.qid, "positive_benefit": float(high.benefit20),
            "negative_query": low.qid, "negative_benefit": float(low.benefit20)})
    write_json(root / "same_edge_opposite_effects.json", switch_examples)

    training = pickle.loads((root / "train_validation_cases.pkl").read_bytes())
    training_y = np.concatenate([c["benefits"][:, 1] for c in training["training"]])
    write_json(root / "training_effects.json", {"candidate_effects": len(training_y),
        "positive": int((training_y > 1e-8).sum()), "negative": int((training_y < -1e-8).sum()),
        "zero": int((np.abs(training_y) <= 1e-8).sum())})

    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    conditions = ["clean", "random05_s17", "random20_mean", "hard20_s17", "missing20_s17"]
    labels = ["Clean", "Random +5%", "Random +20%\n3-seed mean", "Lexical-hard +20%", "Missing 20%"]
    methods = [("learned1", "Learned ≤1"), ("learned3", "Learned ≤3"),
               ("edge_only1", "Edge features ≤1"), ("source_verified", "Source verification")]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for index, (method, name) in enumerate(methods):
        sub = summary[(summary.method == method) & (summary.budget == 20)].set_index("condition").loc[conditions]
        center = sub.delta_recall.to_numpy() * 100
        lower = sub.delta_ci_low.to_numpy() * 100
        upper = sub.delta_ci_high.to_numpy() * 100
        # Percentile CIs need not contain the point estimate in degenerate cases.
        x = np.arange(len(conditions)) + (index - 1.5) * 0.15
        color = plt.get_cmap("tab10")(index)
        ax.vlines(x, lower, upper, linewidth=1.4, color=color)
        ax.scatter(x, center, s=27, label=name, color=color)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(conditions)), labels)
    ax.set_ylabel("Change in direct-premise Recall@20 (percentage points)")
    ax.set_title("Paired effects; 95% source-block bootstrap intervals")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    for suffix in ["png", "svg"]:
        fig.savefig(figures / ("intervention_effects." + suffix), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    recall_methods = [("lexical", "Global lexical"), ("no_op", "Local graph"),
                      ("learned3", "Learned ≤3"), ("source_verified", "Source verification"), ("oracle1", "Candidate oracle ≤1")]
    for index, (method, name) in enumerate(recall_methods):
        sub = summary[(summary.method == method) & (summary.budget == 20)].set_index("condition").loc[conditions]
        ax.plot(np.arange(len(conditions)), sub.recall * 100, marker="o", label=name)
    ax.set_xticks(np.arange(len(conditions)), labels)
    ax.set_ylabel("Direct-premise Recall@20 (%)")
    ax.set_title("Same statement index and output cap; different search costs")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    for suffix in ["png", "svg"]:
        fig.savefig(figures / ("retrieval_recall." + suffix), dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, condition in zip(axes, ["random20_s17", "hard20_s17"]):
        sub = effects[effects.condition == condition]
        positions = [0, 1]
        left = np.zeros(2)
        for sign, color in [("positive", "#1b9e77"), ("zero", "#c8c8c8"), ("negative", "#d95f02")]:
            values = [float(sub[sub.source_supported == supported][sign].mean()) * 100 for supported in [1, 0]]
            ax.barh(positions, values, left=left, color=color, label=sign)
            left += values
        ax.set_yticks(positions, ["Source-supported", "Wrong attribution"])
        ax.set_xlim(0, 100)
        ax.set_title(condition)
        ax.set_xlabel("Candidate effects (%)")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3)
    fig.suptitle("Correctness of attribution and deletion utility are distinct")
    fig.tight_layout(rect=[0, 0.16, 1, 0.95])
    fig.subplots_adjust(wspace=0.24)
    for suffix in ["png", "svg"]:
        fig.savefig(figures / ("signed_effects." + suffix), dpi=220)
    plt.close(fig)

    qa = {"test_queries": results.qid.nunique(), "test_blocks": results.group.nunique(),
          "finite_metrics": bool(np.isfinite(results[["recall", "delta_recall", "ndcg"]].to_numpy()).all()),
          "recall_in_unit_interval": bool(results.recall.between(0, 1).all()),
          "oracle_never_worse_at_primary_budget": bool((results[(results.method == "oracle1") & (results.budget == 20)].delta_recall >= -1e-8).all()),
          "all_output_caps_respected": bool((results.returned <= results.budget).all()),
          "lexical_is_invariant_to_graph_noise": bool(results[results.method == "lexical"].groupby(["qid", "budget"]).recall.nunique().max() == 1),
          "additive_source_verification_matches_clean": True}
    for condition in ["random05_s17", "random20_s17", "random20_s29", "random20_s43", "hard20_s17"]:
        before = results[(results.condition == "clean") & (results.method == "no_op")].sort_values(["qid", "budget"]).recall.to_numpy()
        after = results[(results.condition == condition) & (results.method == "source_verified")].sort_values(["qid", "budget"]).recall.to_numpy()
        qa["additive_source_verification_matches_clean"] &= bool(np.array_equal(before, after))
    write_json(root / "analysis_checks.json", qa)
    assert all(v for k, v in qa.items() if k not in ["test_queries", "test_blocks"])
    write_json(root / "analysis_complete.json", {"summary_rows": len(summary), "single_candidate_effects": len(effects),
        "query_conditioning": sign_stats, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    display = summary[(summary.budget == 20) & summary.condition.isin(conditions) & summary.method.isin(["lexical", "no_op", "learned1", "learned3", "source_verified", "oracle1"])]
    print(display[["condition", "method", "recall", "delta_recall", "delta_ci_low", "delta_ci_high", "harmed_fraction"]].to_string(index=False))
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
