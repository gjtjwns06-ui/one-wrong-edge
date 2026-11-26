"""Run the locked scoped study; select policies before constructing test cases."""
from __future__ import annotations
import bisect
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import platform
import time

import numpy as np
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .prepare import stable_int
from .retrieval import Neighborhood, metrics, ranked


FEATURE_NAMES = [
    "q_source_similarity", "q_target_similarity", "q_similarity_difference",
    "endpoint_similarity", "log_source_outdegree", "log_target_outdegree",
    "log_historical_target_indegree", "source_query_depth", "target_query_depth",
    "target_is_seed", "log_query_first_parents", "log_query_path_support",
    "source_hypothesis_count", "target_hypothesis_count", "log_source_tokens",
    "log_target_tokens", "same_source_section", "source_is_boundary",
    "target_is_boundary", "log_source_order_gap",
]
EDGE_ONLY_COLUMNS = [3, 4, 5, 6, 12, 13, 14, 15, 16, 17, 18, 19]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def make_context(data, condition, matrix):
    reference = set(map(tuple, data["edges"]))
    edges = set(reference)
    rng = np.random.default_rng(condition["seed"])
    changes = []
    if condition["mode"] != "clean":
        originals = sorted(reference)
        chosen = rng.choice(len(originals), size=int(len(originals) * condition["rate"]), replace=False)
        targets = np.asarray([v for _, v in originals], dtype=np.int32)
        for edge_index in chosen:
            u, original_v = originals[int(edge_index)]
            if condition["mode"] == "missing":
                edges.remove((u, original_v))
                changes.append(("removed", u, original_v))
                continue
            same_role = data["nodes"][original_v]["core"]
            accepted = None
            for _ in range(20):
                count = 96 if condition["mode"] == "hard" else 16
                proposals = rng.choice(targets, size=count, replace=True)
                valid = sorted({int(v) for v in proposals if v < u
                                and data["nodes"][int(v)]["core"] == same_role and (u, int(v)) not in edges})
                if valid:
                    if condition["mode"] == "hard":
                        scores = (matrix[valid] @ matrix[u].T).toarray().ravel()
                        accepted = valid[int(np.argmax(scores))]
                    else:
                        accepted = valid[int(rng.integers(len(valid)))]
                    break
            if accepted is not None:
                edges.add((u, accepted))
                changes.append(("added", u, accepted))
    adjacency = [[] for _ in data["nodes"]]
    incoming = [[] for _ in data["nodes"]]
    for u, v in sorted(edges):
        assert u > v
        adjacency[u].append(v)
        incoming[v].append(u)
    return {"name": condition["name"], "adjacency": adjacency, "incoming": incoming,
            "edges": edges, "audit_changes": changes,
            "actual_rate": len(changes) / len(reference)}


class QueryAdjacency:
    def __init__(self, adjacency, forbidden):
        self.adjacency = adjacency
        self.forbidden = forbidden

    def __getitem__(self, index):
        return [v for v in self.adjacency[index] if v not in self.forbidden]


class Study:
    def __init__(self, data, matrix, contexts, root):
        self.data = data
        self.config = data["config"]
        self.matrix = matrix
        self.contexts = contexts
        self.root = root
        self.nodes = data["nodes"]
        self.lengths = np.asarray([len(n["text"].split()) for n in self.nodes], dtype=np.int32)
        self.core = np.asarray([n["core"] for n in self.nodes])
        self.score_cache = {}
        self.endpoint_cache = {}
        self.signature_nodes = {}
        for n in self.nodes:
            self.signature_nodes.setdefault(n["signature"], set()).add(n["id"])

    def query_scores(self, query):
        if query["qid"] not in self.score_cache:
            scores = (self.matrix @ self.matrix[query["node"]].T).toarray().ravel()
            scores[query["node"]:] = -np.inf
            forbidden = self.signature_nodes[query["signature"]]
            scores[list(forbidden)] = -np.inf
            self.score_cache[query["qid"]] = scores
        return self.score_cache[query["qid"]]

    def neighborhood(self, query, context):
        scores = self.query_scores(query)
        seeds = ranked(np.flatnonzero(np.isfinite(scores)), scores)[:self.config["seed_count"]]
        adjacency = QueryAdjacency(context["adjacency"], self.signature_nodes[query["signature"]])
        return Neighborhood(adjacency, seeds, scores)

    def edge_features(self, query, context, neighborhood, edges):
        scores = self.query_scores(query)
        features = []
        for u, v in edges:
            if (u, v) not in self.endpoint_cache:
                self.endpoint_cache[u, v] = float(self.matrix[u].multiply(self.matrix[v]).sum())
            a, b = self.nodes[u], self.nodes[v]
            features.append([
                scores[u], scores[v], scores[v] - scores[u], self.endpoint_cache[u, v],
                np.log1p(len(context["adjacency"][u])), np.log1p(len(context["adjacency"][v])),
                np.log1p(bisect.bisect_left(context["incoming"][v], query["node"])),
                neighborhood.depth[u], neighborhood.depth[v], int(v in neighborhood.seeds),
                np.log1p(neighborhood.first_counts[v]), np.log1p(neighborhood.support[v]),
                a["hypothesis_count"], b["hypothesis_count"],
                np.log1p(self.lengths[u]), np.log1p(self.lengths[v]),
                int((a["chapter"], a["section"]) == (b["chapter"], b["section"])),
                int(not a["core"]), int(not b["core"]), np.log1p(u - v),
            ])
        return np.asarray(features, dtype=np.float32).reshape(-1, len(FEATURE_NAMES))

    def make_case(self, query, context):
        started = time.perf_counter()
        neighborhood = self.neighborhood(query, context)
        scores = self.query_scores(query)
        all_edges = sorted(neighborhood.edges, key=lambda e: (-float(scores[e[1]]), neighborhood.depth[e[0]], e))
        cap = self.config["candidate_cap"]
        if len(all_edges) <= cap:
            candidates = all_edges
        else:
            count = int(round(cap * self.config["candidate_relevant_fraction"]))
            rng = np.random.default_rng(stable_int(query["qid"] + context["name"]))
            rest = rng.choice(len(all_edges) - count, size=cap - count, replace=False)
            candidates = all_edges[:count] + [all_edges[count + int(i)] for i in sorted(rest)]
        candidates = sorted(candidates)
        features = self.edge_features(query, context, neighborhood, candidates)
        gold = set(query["gold"])
        base_ids = neighborhood.ranking[:max(self.config["output_budgets"])]
        base = [metrics(base_ids, gold, k)["recall"] for k in self.config["output_budgets"]]
        benefits = np.asarray([[metrics(neighborhood.remove_one(edge), gold, k)["recall"] - base[j]
                                for j, k in enumerate(self.config["output_budgets"])]
                               for edge in candidates], dtype=np.float32).reshape(-1, len(base))
        return {"qid": query["qid"], "condition": context["name"], "edges": candidates,
                "features": features, "benefits": benefits, "base_ids": base_ids,
                "visited_edges": len(neighborhood.edges), "reached_nodes": len(neighborhood.reached),
                "candidate_coverage": len(candidates) / max(1, len(all_edges)),
                "seeds": sorted(neighborhood.seeds), "offline_label_seconds": time.perf_counter() - started}

    def cases_for(self, split, condition_names):
        cases = []
        queries = [q for q in self.data["queries"] if q["split"] == split]
        for name in condition_names:
            context = self.contexts[name]
            start = time.perf_counter()
            for j, query in enumerate(queries):
                cases.append(self.make_case(query, context))
                if (j + 1) % 400 == 0:
                    print(f"{split}/{name}: {j + 1}/{len(queries)} queries", flush=True)
            print(f"{split}/{name}: {len(queries)} cases, {time.perf_counter()-start:.1f}s", flush=True)
        return cases


def predict(model, features):
    if not len(features):
        return np.zeros(0)
    x = features[:, model["columns"]]
    if model["scaler"] is not None:
        x = model["scaler"].transform(x)
    return np.clip(model["estimator"].predict(x), -1, 1)


def choose_edges(case, predictions, budget, threshold):
    order = np.argsort(-predictions, kind="stable")
    return [case["edges"][i] for i in order[:budget] if predictions[i] > threshold]


def fit_models(cases, config):
    x = np.concatenate([c["features"] for c in cases])
    y = np.concatenate([c["benefits"][:, 1] for c in cases])
    weights = np.concatenate([np.full(len(c["edges"]), 1.0 / max(1, len(c["edges"]))) for c in cases])
    weights *= len(weights) / weights.sum()
    assert np.isfinite(x).all(), "Forbidden or future nodes reached the learner"
    models = {}
    scaler = StandardScaler().fit(x, sample_weight=weights)
    normalized = scaler.transform(x)
    for alpha in config["ridge_alphas"]:
        name = f"ridge_{alpha:g}"
        estimator = Ridge(alpha=alpha).fit(normalized, y, sample_weight=weights)
        models[name] = {"estimator": estimator, "scaler": scaler, "columns": list(range(len(FEATURE_NAMES)))}
    for name, columns in [("boost", list(range(len(FEATURE_NAMES)))), ("edge_only_boost", EDGE_ONLY_COLUMNS)]:
        estimator = HistGradientBoostingRegressor(max_iter=config["boosting_iterations"],
            max_leaf_nodes=config["boosting_leaf_nodes"], l2_regularization=1.0,
            learning_rate=0.08, min_samples_leaf=40, early_stopping=False,
            random_state=config["split_seed"]).fit(x[:, columns], y, sample_weight=weights)
        models[name] = {"estimator": estimator, "scaler": None, "columns": columns}
    print(f"Fitted {len(models)} models on {len(y):,} candidate effects", flush=True)
    return models


def select_policies(study, models, validation):
    query_map = {q["qid"]: q for q in study.data["queries"]}
    records = []
    cached_utility = {}
    primary = study.config["primary_budget"]
    for model_name, model in models.items():
        predictions = [predict(model, c["features"]) for c in validation]
        for budget in study.config["batch_deletion_budgets"]:
            if model_name == "edge_only_boost" and budget != 1:
                continue
            for threshold in study.config["thresholds"]:
                gains, counts = [], []
                for j, (case, pred) in enumerate(zip(validation, predictions)):
                    selected = choose_edges(case, pred, budget, threshold)
                    counts.append(len(selected))
                    if not selected:
                        gains.append(0.0)
                    elif len(selected) == 1:
                        gains.append(float(case["benefits"][case["edges"].index(selected[0]), 1]))
                    else:
                        key = (j, tuple(selected))
                        if key not in cached_utility:
                            query = query_map[case["qid"]]
                            neighborhood = study.neighborhood(query, study.contexts[case["condition"]])
                            ids = neighborhood.remove_many(selected, study.query_scores(query))
                            base = metrics(case["base_ids"], set(query["gold"]), primary)["recall"]
                            cached_utility[key] = metrics(ids, set(query["gold"]), primary)["recall"] - base
                        gains.append(cached_utility[key])
                records.append({"model": model_name, "budget": budget, "threshold": threshold,
                                "validation_gain": float(np.mean(gains)),
                                "validation_mean_deletions": float(np.mean(counts))})
    selection = {}
    for name, budget, edge_only in [("learned1", 1, False), ("learned3", 3, False), ("edge_only1", 1, True)]:
        options = [r for r in records if r["budget"] == budget and (r["model"] == "edge_only_boost") == edge_only]
        best = max(options, key=lambda r: (round(r["validation_gain"], 12), -r["validation_mean_deletions"]))
        selection[name] = best
    write_json(study.root / "validation_grid.json", records)
    write_json(study.root / "selection.json", selection)
    print("Policy selection locked: " + json.dumps(selection), flush=True)
    return selection


def evaluate(study, models, selection, cases):
    queries = {q["qid"]: q for q in study.data["queries"]}
    reference = set(map(tuple, study.data["edges"]))
    rows = []
    diagnostic = Counter()
    labels_path = study.root / "single_interventions.csv.gz"
    with gzip.open(study.root / "retrievals.jsonl.gz", "wt", encoding="utf-8") as output, \
         gzip.open(labels_path, "wt", newline="", encoding="utf-8") as label_file:
        writer = csv.writer(label_file)
        writer.writerow(["qid", "condition", "source", "target", "source_supported", "benefit10", "benefit20", "benefit40"])
        for case_index, case in enumerate(cases):
            query = queries[case["qid"]]
            context = study.contexts[case["condition"]]
            scores = study.query_scores(query)
            gold = set(query["gold"])
            neighborhood = study.neighborhood(query, context)
            predictions = {name: predict(model, case["features"]) for name, model in models.items()}
            actions = {"no_op": []}
            for name, policy in selection.items():
                actions[name] = choose_edges(case, predictions[policy["model"]], policy["budget"], policy["threshold"])
            rng = np.random.default_rng(stable_int(query["qid"] + context["name"] + "random_policy"))
            order = rng.permutation(len(case["edges"]))
            for budget in [1, 3]:
                actions[f"random{budget}"] = [case["edges"][int(i)] for i in order[:budget]]
                count = len(actions[f"learned{budget}"])
                actions[f"matched_random{budget}"] = [case["edges"][int(i)] for i in order[:count]]
                heuristic = sorted(case["edges"], key=lambda e: (float(scores[e[1]]), e))[:budget]
                actions[f"relevance{budget}"] = heuristic
            if len(case["edges"]):
                best = int(np.argmax(case["benefits"][:, 1]))
                actions["oracle1"] = [case["edges"][best]] if case["benefits"][best, 1] > 0 else []
            else:
                actions["oracle1"] = []
            retrievals = {}
            for method, deleted in actions.items():
                retrievals[method] = neighborhood.remove_many(deleted, scores)
            lexical = ranked(np.flatnonzero(np.isfinite(scores)), scores)[:max(study.config["output_budgets"])]
            retrievals["lexical"] = lexical
            if context["name"].startswith(("random", "hard")):
                verified_neighborhood = study.neighborhood(query, study.contexts["clean"])
                retrievals["source_verified"] = verified_neighborhood.ranking[:40]
            else:
                retrievals["source_verified"] = case["base_ids"]
            for edge, effect in zip(case["edges"], case["benefits"]):
                supported = int(edge in reference)
                writer.writerow([query["qid"], context["name"], *edge, supported, *map(float, effect)])
                diagnostic[(context["name"], supported, "positive" if effect[1] > 1e-8 else "negative" if effect[1] < -1e-8 else "zero")] += 1
            for method, ids in retrievals.items():
                assert all(study.nodes[int(v)]["original_index"] < query["original_index"] for v in ids)
                assert all(study.nodes[int(v)]["signature"] != query["signature"] for v in ids)
                for budget in study.config["output_budgets"]:
                    measured = metrics(ids, gold, budget, study.lengths)
                    baseline = metrics(case["base_ids"], gold, budget)["recall"]
                    core_gold = {v for v in gold if study.core[v]}
                    boundary_gold = gold - core_gold
                    row = {"qid": query["qid"], "group": query["group"], "topic": query["topic"],
                           "section": query["section"], "condition": context["name"],
                           "method": method, "budget": budget, **measured,
                           "delta_recall": measured["recall"] - baseline,
                           "deletions": len(actions.get(method, [])) if method != "source_verified" else len(context["edges"] - reference),
                           "gold_count": len(gold), "core_gold_count": len(core_gold),
                           "core_recall": len(set(ids[:budget]) & core_gold) / len(core_gold) if core_gold else "",
                           "boundary_recall": len(set(ids[:budget]) & boundary_gold) / len(boundary_gold) if boundary_gold else "",
                           "visited_edges": case["visited_edges"], "reached_nodes": case["reached_nodes"],
                           "candidates": len(case["edges"]), "candidate_coverage": case["candidate_coverage"],
                           "offline_label_seconds": case["offline_label_seconds"]}
                    rows.append(row)
            output.write(json.dumps({"qid": query["qid"], "condition": context["name"],
                                     "actions": actions, "ranks": {m: list(map(int, ids)) for m, ids in retrievals.items()}}) + "\n")
            if (case_index + 1) % 700 == 0:
                print(f"Evaluated {case_index + 1}/{len(cases)} test graph-query cases", flush=True)
    with (study.root / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(study.root / "effect_sign_counts.json", [
        {"condition": condition, "source_supported": supported, "sign": sign, "count": count}
        for (condition, supported, sign), count in sorted(diagnostic.items())])
    return rows


def main():
    started = time.perf_counter()
    root = Path("runs/scoped_v1")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "run_complete.json").exists():
        raise RuntimeError("Completed run exists. Archive it explicitly before rerunning this fixed identifier.")
    lock = json.loads(Path("data/processed/scoped/design_lock.json").read_text(encoding="utf-8"))
    for path, expected in lock["hashes"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Locked design/data changed: {path}")
    data = pickle.loads(Path("data/processed/scoped/dataset.pkl").read_bytes())
    code_paths = list(Path("owe").glob("*.py"))
    manifest = {"protocol_lock": lock, "python": platform.python_version(),
                "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                "feature_names": FEATURE_NAMES, "edge_only_columns": EDGE_ONLY_COLUMNS,
                "code_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths},
                "source_commit": data["config"]["source_commit"],
                "test_used_for_selection": False}
    write_json(root / "manifest.json", manifest)
    query_signatures = {q["signature"] for q in data["queries"]}
    background = [n["text"] for n in data["nodes"] if n["signature"] not in query_signatures]
    vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                                ngram_range=(1, 2), min_df=1, max_features=150000, dtype=np.float32)
    vectorizer.fit(background)
    matrix = vectorizer.transform([n["text"] for n in data["nodes"]]).tocsr()
    with (root / "vectorizer.pkl").open("wb") as f:
        pickle.dump(vectorizer, f, protocol=5)
    contexts = {c["name"]: make_context(data, c, matrix) for c in data["config"]["conditions"]}
    write_json(root / "graph_audit.json", [{"condition": name, "nodes": len(data["nodes"]),
        "edges": len(c["edges"]), "changes": len(c["audit_changes"]), "actual_rate": c["actual_rate"],
        "edge_sha256": hashlib.sha256(json.dumps(sorted(c["edges"])).encode()).hexdigest()}
        for name, c in contexts.items()])
    with (root / "graphs.pkl").open("wb") as f:
        pickle.dump(contexts, f, protocol=5)
    print(f"Frozen {len(data['nodes'])} scoped/boundary nodes, {len(data['edges'])} searchable reference edges; TF-IDF {matrix.shape}", flush=True)
    study = Study(data, matrix, contexts, root)
    training = study.cases_for("train", data["config"]["training_conditions"])
    validation = study.cases_for("validation", data["config"]["training_conditions"])
    with (root / "train_validation_cases.pkl").open("wb") as f:
        pickle.dump({"training": training, "validation": validation}, f, protocol=5)
    fit_start = time.perf_counter()
    models = fit_models(training, data["config"])
    fit_seconds = time.perf_counter() - fit_start
    selection = select_policies(study, models, validation)
    with (root / "models.pkl").open("wb") as f:
        pickle.dump(models, f, protocol=5)
    selection_hash = hashlib.sha256((root / "selection.json").read_bytes()).hexdigest()
    # No test case/effect is constructed before the selection artifact is fixed.
    test_cases = study.cases_for("test", list(contexts))
    rows = evaluate(study, models, selection, test_cases)
    with (root / "test_cases.pkl").open("wb") as f:
        pickle.dump(test_cases, f, protocol=5)
    assert selection_hash == hashlib.sha256((root / "selection.json").read_bytes()).hexdigest()
    write_json(root / "run_complete.json", {"seconds": time.perf_counter() - started,
        "model_fit_seconds": fit_seconds, "training_cases": len(training), "validation_cases": len(validation),
        "test_cases": len(test_cases), "result_rows": len(rows), "selection_sha256": selection_hash})
    print(f"Completed {len(test_cases)} test cases; {len(rows)} policy/budget measurements; {time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
