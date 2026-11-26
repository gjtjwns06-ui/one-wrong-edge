"""Sequential candidate selection and proof-attribution reranker training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .experiment import write_json
from .prepare import stable_int
from .retrieval import metrics
from .sequential_retriever import HybridEngine, RANK_FEATURES, model_scores, predicted_ranking, prepare_contexts


ROOT = Path("runs/sequential_v2")


def load_study():
    config = json.loads(Path("configs/sequential_v2.json").read_text(encoding="utf-8"))
    data = pickle.loads(Path(config["dataset"]).read_bytes())
    source = Path(config["source_run"])
    contexts = prepare_contexts(pickle.loads((source / "graphs.pkl").read_bytes()))
    vectorizer = pickle.loads((source / "vectorizer.pkl").read_bytes())
    engine = HybridEngine(data, vectorizer, config)
    return config, data, contexts, engine


def sample_training(result, query, seed, config):
    y = np.isin(result.ids, query["gold"]).astype(np.int8)
    positive = np.flatnonzero(y)
    negative = np.flatnonzero(y == 0)
    order = negative[np.lexsort((result.ids[negative], -result.features[negative, 0]))]
    hard = order[:config["ranker_negative_top"]]
    remaining = np.setdiff1d(negative, hard)
    rng = np.random.default_rng(seed)
    random = rng.choice(remaining, size=min(len(remaining), config["ranker_negative_random"]), replace=False)
    selected = np.concatenate([positive, hard, random]).astype(int)
    labels = y[selected]
    weights = np.zeros(len(selected), dtype=np.float32)
    if len(positive) and len(selected) > len(positive):
        weights[labels == 1] = 0.5 / len(positive)
        weights[labels == 0] = 0.5 / (len(selected) - len(positive))
    else:
        weights[:] = 1 / max(1, len(selected))
    return result.features[selected], labels, weights


def training_arrays(engine, contexts, queries, variant, config, label):
    xs, ys, ws = [], [], []
    records = []
    for condition in config["training_conditions"]:
        for index, query in enumerate(queries):
            result = engine.retrieve(query, contexts[condition], variant)
            x, y, w = sample_training(result, query, stable_int(query["qid"] + condition + label), config)
            xs.append(x); ys.append(y); ws.append(w)
            records.append({"qid": query["qid"], "group": query["group"], "condition": condition,
                            "sampled": len(y), "positive": int(y.sum()), "candidates": len(result.ids)})
            engine.trim_cache()
            if (index + 1) % 600 == 0:
                print(f"{label}/{condition}: {index + 1}/{len(queries)} training queries", flush=True)
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    w *= len(w) / w.sum()
    return x, y, w, records


def fit_ranker(name, config, x, y, w):
    if name.startswith("logistic"):
        value = float(name.split("_")[1])
        scaler = StandardScaler().fit(x, sample_weight=w)
        estimator = LogisticRegression(C=value, max_iter=500, solver="lbfgs", random_state=config["random_seed"])
        estimator.fit(scaler.transform(x), y, sample_weight=w)
        return {"name": name, "estimator": estimator, "scaler": scaler}
    leaves = int(name.split("_")[1])
    estimator = HistGradientBoostingClassifier(max_iter=config["ranker_boost_iterations"],
        max_leaf_nodes=leaves, learning_rate=0.08, min_samples_leaf=50,
        l2_regularization=1.0, early_stopping=False, random_state=config["random_seed"])
    estimator.fit(x, y, sample_weight=w)
    return {"name": name, "estimator": estimator, "scaler": None}


def stage1(config, data, contexts, engine):
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "stage1_selection.json").exists():
        raise RuntimeError("Stage1 already selected; preserve the saved choice")
    original = json.loads(Path("ARTIFACTS.json").read_text(encoding="utf-8"))
    for name, record in original["files"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == record["sha256"], name
    manifest = {"version": config["version"], "test_set_previously_used": True,
                "protocol_sha256": hashlib.sha256(Path("SEQUENTIAL_V2_PROTOCOL.md").read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(Path("configs/sequential_v2.json").read_bytes()).hexdigest(),
                "dataset_sha256": hashlib.sha256(Path(config["dataset"]).read_bytes()).hexdigest(),
                "original_files_verified": len(original["files"]), "rank_features": RANK_FEATURES}
    write_json(ROOT / "manifest.json", manifest)
    rows = []
    queries = [q for q in data["queries"] if q["split"] == "validation"]
    for variant in config["candidate_variants"]:
        start = time.perf_counter()
        for condition in config["training_conditions"]:
            for query in queries:
                result = engine.retrieve(query, contexts[condition], variant)
                gold = set(query["gold"])
                hits = len(set(result.ids) & gold)
                lexical = engine.query_info(query)["lexical"][:config["candidate_cap"]]
                rows.append({"variant": variant["name"], "condition": condition,
                    "qid": query["qid"], "group": query["group"], "coverage": hits / len(gold),
                    "ceiling20": min(20, hits) / len(gold), "candidates": len(result.ids),
                    "transitions": result.transitions,
                    "lexical256_coverage": len(set(lexical) & gold) / len(gold)})
                engine.trim_cache()
        print(f"Stage1/{variant['name']}: {time.perf_counter()-start:.1f}s", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "stage1_validation.csv", index=False)
    summary = frame.groupby("variant", as_index=False)[["coverage", "ceiling20", "candidates", "transitions", "lexical256_coverage"]].mean()
    summary.to_csv(ROOT / "stage1_validation_summary.csv", index=False)
    best = summary.sort_values(["ceiling20", "coverage", "transitions"], ascending=[False, False, True]).iloc[0]
    variant = next(v for v in config["candidate_variants"] if v["name"] == best.variant)
    write_json(ROOT / "stage1_selection.json", {"variant": variant, "validation": best.to_dict(), "selected_before_stage2": True})
    print(summary.to_string(index=False))
    print("Stage1 locked: " + variant["name"], flush=True)


def stage2(config, data, contexts, engine):
    if (ROOT / "stage2_selection.json").exists():
        raise RuntimeError("Stage2 already selected; preserve the saved choice")
    stage1_record = json.loads((ROOT / "stage1_selection.json").read_text(encoding="utf-8"))
    variant = stage1_record["variant"]
    train = [q for q in data["queries"] if q["split"] == "train"]
    validation = [q for q in data["queries"] if q["split"] == "validation"]
    started = time.perf_counter()
    x, y, w, records = training_arrays(engine, contexts, train, variant, config, "stage2")
    np.savez_compressed(ROOT / "ranker_training.npz", features=x, labels=y, weights=w)
    pd.DataFrame(records).to_csv(ROOT / "ranker_training_cases.csv", index=False)
    validation_cases = []
    for condition in config["training_conditions"]:
        for query in validation:
            result = engine.retrieve(query, contexts[condition], variant)
            validation_cases.append({"query": query, "condition": condition, "retrieval": result})
            engine.trim_cache()
    with (ROOT / "ranker_validation_cases.pkl").open("wb") as f:
        pickle.dump(validation_cases, f, protocol=5)
    all_x = np.concatenate([c["retrieval"].features for c in validation_cases])
    models = {}
    summaries = []
    rows = []
    names = [f"logistic_{value:g}" for value in config["ranker_logistic_C"]]
    names += [f"boost_{leaves}" for leaves in config["ranker_boost_leaf_nodes"]]
    for name in names:
        fit_start = time.perf_counter()
        model = fit_ranker(name, config, x, y, w)
        scores = model_scores(model, all_x)
        models[name] = model
        offset = 0
        rec = []
        for case in validation_cases:
            result, query = case["retrieval"], case["query"]
            prediction = scores[offset:offset + len(result.ids)]
            offset += len(result.ids)
            ranking = predicted_ranking(result, prediction)
            measured = metrics(ranking, set(query["gold"]), 20, engine.lengths)
            row = {"model": name, "qid": query["qid"], "condition": case["condition"], "group": query["group"], **measured}
            rows.append(row); rec.append(row)
        frame = pd.DataFrame(rec)
        summaries.append({"model": name, "recall20": frame.recall.mean(), "complete": frame.complete.mean(),
                          "ndcg": frame.ndcg.mean(), "fit_and_validation_seconds": time.perf_counter()-fit_start})
        print(f"Stage2/{name}: validation Recall20={frame.recall.mean():.4f}", flush=True)
    for method in ["lexical", "frequency", "hand_mix"]:
        rec = []
        for case in validation_cases:
            result, query = case["retrieval"], case["query"]
            features = result.features
            if method == "lexical":
                scores = features[:, 0]
            elif method == "frequency":
                scores = features[:, 5]
            else:
                scores = features[:, 0] + 0.25 * features[:, 5] / max(1e-6, features[:, 5].max())
                scores += 0.25 * features[:, 26] / max(1e-6, features[:, 26].max())
            ranking = predicted_ranking(result, scores)
            measured = metrics(ranking, set(query["gold"]), 20, engine.lengths)
            row = {"model": method, "qid": query["qid"], "condition": case["condition"], "group": query["group"], **measured}
            rows.append(row); rec.append(row)
        frame = pd.DataFrame(rec)
        summaries.append({"model": method, "recall20": frame.recall.mean(), "complete": frame.complete.mean(),
                          "ndcg": frame.ndcg.mean(), "fit_and_validation_seconds": 0.0})
    summary = pd.DataFrame(summaries)
    summary.to_csv(ROOT / "stage2_validation_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "stage2_validation.csv", index=False)
    trained = summary[summary.model.isin(names)]
    best = trained.sort_values(["recall20", "complete"], ascending=False).iloc[0]
    with (ROOT / "ranker.pkl").open("wb") as f:
        pickle.dump(models[best.model], f, protocol=5)
    write_json(ROOT / "stage2_selection.json", {"model": best.model, "validation": best.to_dict(),
        "selected_before_stage3": True, "training_candidate_pairs": len(y), "training_positive_pairs": int(y.sum()),
        "stage1_selection_sha256": hashlib.sha256((ROOT / "stage1_selection.json").read_bytes()).hexdigest(),
        "seconds": time.perf_counter()-started})
    print(summary.to_string(index=False))
    print("Stage2 locked: " + best.model, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["stage1", "stage2"])
    args = parser.parse_args()
    config, data, contexts, engine = load_study()
    if args.stage == "stage1":
        stage1(config, data, contexts, engine)
    else:
        stage2(config, data, contexts, engine)


if __name__ == "__main__":
    main()
