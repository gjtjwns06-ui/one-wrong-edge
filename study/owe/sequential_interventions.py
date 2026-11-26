"""Question-conditioned edge effects for the frozen learned premise retriever."""
from __future__ import annotations
import numpy as np
from threadpoolctl import ThreadpoolController

_UTILITY_POOLS = None

from .prepare import stable_int
from .retrieval import metrics
from .sequential_retriever import model_scores, predicted_ranking


EDGE_FEATURES = [
    "q_source_cosine", "q_target_cosine", "endpoint_cosine", "source_predicted_use",
    "target_predicted_use", "source_in_pool", "target_in_pool", "target_predicted_rank",
    "target_score_minus_cutoff20", "target_lexical_anchor", "target_prior_anchor",
    "source_is_seed", "source_graph_depth", "target_graph_depth", "log_target_support",
    "log_target_path_mass", "log_source_outdegree", "log_target_outdegree",
    "log_historical_target_indegree", "source_hypotheses", "target_hypotheses",
    "source_is_boundary", "target_is_boundary", "source_target_goal_cosine",
    "source_hyp_to_target_goal_cosine", "base_cutoff_score", "log_candidate_count",
    "log_transition_count", "same_source_section"
]


def edge_features(engine, query, context, result, predictions, edges):
    info = engine.query_info(query)
    history = engine.history(query, context)
    relevance = {int(v): float(p) for v, p in zip(result.ids, predictions)}
    ranking = predicted_ranking(result, predictions)
    ranks = {int(v): j + 1 for j, v in enumerate(ranking)}
    sorted_scores = np.sort(predictions)[::-1]
    cutoff = float(sorted_scores[min(19, len(sorted_scores) - 1)])
    values = []
    for u, v in edges:
        a, b = engine.nodes[u], engine.nodes[v]
        if (u,v) not in engine.endpoint_cache:
            engine.endpoint_cache[u,v] = (
                float(engine.matrix[u].multiply(engine.matrix[v]).sum()),
                float(engine.goals[u].multiply(engine.goals[v]).sum()),
                float(engine.hypotheses[u].multiply(engine.goals[v]).sum()))
        text_cosine, goal_cosine, hypothesis_goal_cosine = engine.endpoint_cache[u,v]
        values.append([info["scores"][u], info["scores"][v],
            text_cosine, relevance.get(u, 0), relevance.get(v, 0),
            u in relevance, v in relevance, ranks.get(v, 257) / 256,
            relevance.get(v, 0) - cutoff, v in result.lexical_anchors, v in result.prior_anchors,
            u in result.seeds, result.graph_depth.get(u, 9), result.graph_depth.get(v, 9),
            np.log1p(result.support.get(v, 0)), np.log1p(result.mass.get(v, 0)),
            np.log1p(len(context["adjacency"][u])), np.log1p(len(context["adjacency"][v])), np.log1p(history[v]),
            a["hypothesis_count"], b["hypothesis_count"], not a["core"], not b["core"],
            goal_cosine, hypothesis_goal_cosine, cutoff,
            np.log1p(len(result.ids)), np.log1p(result.transitions),
            (a["chapter"], a["section"]) == (b["chapter"], b["section"])])
    x = np.asarray(values, dtype=np.float32).reshape(-1, len(EDGE_FEATURES))
    assert np.isfinite(x).all()
    return x


def batch_rank(ranker, results, baseline=None, baseline_predictions=None):
    if not results:
        return []
    if baseline is None:
        features = np.concatenate([r.features for r in results])
        predictions = model_scores(ranker, features)
        predictions_by_result = []
        offset = 0
        for result in results:
            predictions_by_result.append(predictions[offset:offset + len(result.ids)])
            offset += len(result.ids)
    else:
        if baseline_predictions is None:
            baseline_predictions = model_scores(ranker, baseline.features)
        changed_features, changed_masks, predictions_by_result = [], [], []
        for result in results:
            positions = np.searchsorted(baseline.ids, result.ids)
            safe = np.minimum(positions, len(baseline.ids) - 1)
            unchanged = (positions < len(baseline.ids)) & (baseline.ids[safe] == result.ids)
            unchanged &= np.all(result.features == baseline.features[safe], axis=1)
            values = baseline_predictions[safe].copy()
            changed = ~unchanged
            if changed.any():
                changed_features.append(result.features[changed])
            changed_masks.append(changed)
            predictions_by_result.append(values)
        if changed_features:
            changed_predictions = model_scores(ranker, np.concatenate(changed_features))
            offset = 0
            for values, changed in zip(predictions_by_result, changed_masks):
                count = int(changed.sum())
                values[changed] = changed_predictions[offset:offset + count]
                offset += count
    rankings = []
    for result, scores in zip(results, predictions_by_result):
        rankings.append(predicted_ranking(result, scores)[:40])
    return rankings


def candidate_edges(result, query, condition, predictions, cap):
    relevance = dict(zip(map(int, result.ids), predictions))
    edges = sorted(result.edges, key=lambda e: (-float(relevance.get(e[1], 0)), e))
    if len(edges) > cap:
        relevant = cap // 2
        rng = np.random.default_rng(stable_int(query["qid"] + condition + "v2-edge-sampling"))
        indices = rng.choice(len(edges) - relevant, cap - relevant, replace=False)
        edges = edges[:relevant] + [edges[relevant + int(i)] for i in sorted(indices)]
    return sorted(edges)


def make_case(engine, query, context, variant, ranker, condition, fold=None):
    base = engine.retrieve(query, context, variant)
    predictions = model_scores(ranker, base.features)
    edges = candidate_edges(base, query, condition, predictions, engine.config["utility_candidate_cap"])
    x = edge_features(engine, query, context, base, predictions, edges)
    ranking = predicted_ranking(base, predictions)[:40]
    variants = [engine.retrieve(query, context, variant, [edge]) for edge in edges]
    rankings = batch_rank(ranker, variants, base, predictions)
    gold = set(query["gold"])
    benefits = np.asarray([[metrics(ids, gold, k)["recall"] - metrics(ranking, gold, k)["recall"]
                            for k in engine.config["output_budgets"]] for ids in rankings], dtype=np.float32)
    benefits = benefits.reshape(-1, len(engine.config["output_budgets"]))
    return {"qid": query["qid"], "group": query["group"], "condition": condition, "fold": fold,
        "edges": edges, "features": x, "benefits": benefits, "single_rankings": rankings,
        "base_ranking": ranking, "base_candidates": base.ids, "base_scores": predictions,
        "base_transitions": base.transitions, "base_seeds": base.seeds,
        "visited_edges": len(base.edges), "sampled_edges": len(edges)}


def utility_predictions(model, x):
    global _UTILITY_POOLS
    if not len(x):
        return np.zeros(0)
    if model.get("scaler") is not None:
        x = model["scaler"].transform(x)
    if _UTILITY_POOLS is None:
        _UTILITY_POOLS = ThreadpoolController()
    if model["kind"] == "wrongness":
        with _UTILITY_POOLS.limit(limits=1, user_api="openmp"):
            return model["estimator"].predict_proba(x)[:, 1]
    with _UTILITY_POOLS.limit(limits=1, user_api="openmp"):
        return np.clip(model["estimator"].predict(x), -1, 1)


def choose_actions(case, predictions, budget, threshold):
    order = np.argsort(-predictions, kind="stable")
    return [case["edges"][int(i)] for i in order[:budget] if predictions[i] > threshold]
