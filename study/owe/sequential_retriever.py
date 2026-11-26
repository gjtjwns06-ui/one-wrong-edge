"""Frozen-data hybrid candidate generation and proof-use ranking features."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np
from sklearn.preprocessing import normalize
from threadpoolctl import ThreadpoolController

_PREDICTION_POOLS = None

from .retrieval import ranked


RANK_FEATURES = [
    "text_cosine", "goal_cosine", "query_hyp_to_premise_goal", "query_goal_to_premise_hyp",
    "hypothesis_cosine", "log_historical_indegree", "log_outdegree", "source_order_distance",
    "log_premise_tokens", "log_query_tokens", "premise_query_length_ratio",
    "premise_hypotheses", "query_hypotheses", "hypothesis_count_difference",
    "premise_distinct_constraints", "query_distinct_constraints", "premise_is_axiom_definition",
    "premise_is_definition", "premise_is_logical_axiom", "premise_is_boundary",
    "lexical_anchor", "prior_anchor", "lexical_seed", "expandable_seed",
    "graph_reached", "minimum_graph_depth", "graph_support", "graph_path_mass",
    "premise_implications", "premise_equivalences", "premise_forall", "premise_exists",
    "query_implications", "query_equivalences", "query_forall", "query_exists",
    "premise_memberships", "query_memberships", "premise_graph_tokens", "query_graph_tokens",
    "query_finite_tokens", "premise_finite_tokens"
]


def split_parts(text):
    pieces = text.split(" DISTINCT ")[0].split(" HYP ")
    return pieces[0].removeprefix("GOAL "), " ".join(pieces[1:])


@dataclass
class Retrieval:
    ids: np.ndarray
    features: np.ndarray
    seeds: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    lexical_anchors: frozenset[int]
    prior_anchors: frozenset[int]
    graph_depth: dict
    support: dict
    mass: dict
    transitions: int


class HybridEngine:
    def __init__(self, data, vectorizer, config):
        self.data = data
        self.nodes = data["nodes"]
        self.config = config
        self.vectorizer = vectorizer
        texts = [n["text"] for n in self.nodes]
        self.matrix = vectorizer.transform(texts).tocsr()
        parts = [split_parts(t) for t in texts]
        self.goals = normalize(vectorizer.transform([p[0] for p in parts])).tocsr()
        self.hypotheses = normalize(vectorizer.transform([p[1] for p in parts])).tocsr()
        self.lengths = np.array([len(t.split()) for t in texts], dtype=np.int32)
        self.hyp_counts = np.array([n["hypothesis_count"] for n in self.nodes], dtype=np.float32)
        self.distinct = np.array([n["distinct_count"] for n in self.nodes], dtype=np.float32)
        self.core = np.array([n["core"] for n in self.nodes])
        self.axioms = np.array([n["kind"] == "$a" for n in self.nodes], dtype=np.float32)
        self.definitions = np.array([n["label"].startswith("df-") for n in self.nodes], dtype=np.float32)
        self.logical_axioms = np.array([n["label"].startswith("ax-") for n in self.nodes], dtype=np.float32)
        graph_tokens = {"UPGraph", "UMGraph", "USGraph", "UHGraph", "Vtx", "iEdg", "Walks", "Paths", "EulerPaths", "Trails", "Circuits", "VtxDeg"}
        finite_tokens = {"Fin", "NN", "NN0", "#"}
        self.counts = np.array([[t.split().count(x) for x in ["->", "<->", "A.", "E.", "e."]]
                                + [sum(x in graph_tokens for x in t.split()), sum(x in finite_tokens for x in t.split())]
                               for t in texts], dtype=np.float32)
        self.query_cache = {}
        self.history_cache = {}
        self.endpoint_cache = {}

    def query_info(self, query):
        key = query["qid"]
        if key not in self.query_cache:
            u = query["node"]
            def cosine(a, b):
                return (a @ b[u].T).toarray().ravel().astype(np.float32)
            similarities = np.column_stack([cosine(self.matrix, self.matrix), cosine(self.goals, self.goals),
                cosine(self.goals, self.hypotheses), cosine(self.hypotheses, self.goals),
                cosine(self.hypotheses, self.hypotheses)])
            eligible = np.arange(len(self.nodes)) < u
            eligible &= np.array([n["signature"] != query["signature"] for n in self.nodes])
            scores = similarities[:, 0].copy()
            scores[~eligible] = -np.inf
            self.query_cache[key] = {"similarities": similarities, "eligible": eligible,
                                     "scores": scores, "lexical": ranked(np.flatnonzero(eligible), scores)}
        return self.query_cache[key]

    def trim_cache(self, keep=12):
        while len(self.query_cache) > keep:
            self.query_cache.pop(next(iter(self.query_cache)))
        while len(self.history_cache) > keep * 4:
            self.history_cache.pop(next(iter(self.history_cache)))

    def history(self, query, context, deleted=()):
        key = (query["qid"], context["name"])
        if key not in self.history_cache:
            source, target = context["edge_arrays"]
            counts = np.bincount(target[source < query["node"]], minlength=len(self.nodes)).astype(np.float32)
            self.history_cache[key] = counts
        counts = self.history_cache[key]
        if deleted:
            counts = counts.copy()
            for u, v in deleted:
                if u < query["node"]:
                    counts[v] -= 1
        return counts

    def retrieve(self, query, context, variant, deleted=()):
        info = self.query_info(query)
        deleted = frozenset(map(tuple, deleted))
        removed = defaultdict(set)
        for u, v in deleted:
            removed[u].add(v)
        def neighbors(u):
            return [v for v in context["adjacency"][u] if info["eligible"][v] and v not in removed[u]]
        lexical = info["lexical"]
        lexical_anchors = frozenset(map(int, lexical[:variant["lexical"]]))
        history = self.history(query, context, deleted)
        prior_scores = history.copy()
        prior_scores[~info["eligible"]] = -np.inf
        prior_order = ranked(np.flatnonzero(info["eligible"] & (history > 0)), prior_scores)
        prior_anchors = frozenset(int(v) for v in prior_order[:variant["prior"]] if history[v] > 0)
        lexical_seeds = set(map(int, lexical[:variant["lexical_seeds"]]))
        expandable = []
        for u in lexical[:variant["seed_pool"]]:
            if neighbors(int(u)):
                expandable.append(int(u))
            if len(expandable) == variant["expandable_seeds"]:
                break
        seeds = lexical_seeds | set(expandable)
        depth = {u: 0 for u in seeds}
        mass = defaultdict(float)
        support = defaultdict(int)
        frontier = {u: max(0.01, float(info["scores"][u])) for u in seeds}
        for u, value in frontier.items():
            mass[u] += value
        edges = set()
        transitions = 0
        expanded = set()
        for hop in range(1, variant["hops"] + 1):
            following = defaultdict(float)
            for u in sorted(frontier, key=lambda u: (-frontier[u], u)):
                if u in expanded:
                    continue
                expanded.add(u)
                targets = neighbors(u)
                amount = frontier[u] * 0.5 / max(1, len(targets))
                for v in targets:
                    if transitions >= self.config["transition_budget"]:
                        break
                    transitions += 1
                    edges.add((u, v))
                    support[v] += 1
                    mass[v] += amount
                    following[v] += amount
                    depth[v] = min(depth.get(v, hop), hop)
                if transitions >= self.config["transition_budget"]:
                    break
            frontier = following
            if transitions >= self.config["transition_budget"]:
                break
        candidates = set(lexical_anchors) | set(prior_anchors)
        graph_order = sorted(depth, key=lambda v: (-mass[v], -float(info["scores"][v]), v))
        for v in graph_order:
            if len(candidates) >= self.config["candidate_cap"]:
                break
            candidates.add(v)
        # Fill unoccupied slots without using any query gold.
        for v in lexical:
            if len(candidates) >= self.config["candidate_cap"]:
                break
            candidates.add(int(v))
        ids = np.asarray(sorted(candidates), dtype=np.int32)
        outdegree = np.array([len(context["adjacency"][int(v)]) - len(removed[int(v)]) for v in ids], dtype=np.float32)
        u = query["node"]
        values = [info["similarities"][ids], np.log1p(history[ids])[:, None], np.log1p(outdegree)[:, None],
            np.log1p(u - ids)[:, None], np.log1p(self.lengths[ids])[:, None],
            np.full((len(ids), 1), np.log1p(self.lengths[u])), (self.lengths[ids] / max(1, self.lengths[u]))[:, None],
            self.hyp_counts[ids, None], np.full((len(ids), 1), self.hyp_counts[u]),
            (self.hyp_counts[ids] - self.hyp_counts[u])[:, None], self.distinct[ids, None],
            np.full((len(ids), 1), self.distinct[u]), self.axioms[ids, None], self.definitions[ids, None],
            self.logical_axioms[ids, None], (~self.core[ids]).astype(float)[:, None],
            np.array([[v in lexical_anchors, v in prior_anchors, v in lexical_seeds, v in expandable,
                       v in depth, depth.get(int(v), 9), np.log1p(support[int(v)]), np.log1p(mass[int(v)])]
                      for v in ids], dtype=np.float32)]
        for index in range(4):
            values.append(self.counts[ids, index, None])
        for index in range(4):
            values.append(np.full((len(ids), 1), self.counts[u, index]))
        values.extend([self.counts[ids, 4, None], np.full((len(ids), 1), self.counts[u, 4]),
                       self.counts[ids, 5, None], np.full((len(ids), 1), self.counts[u, 5]),
                       np.full((len(ids), 1), self.counts[u, 6]), self.counts[ids, 6, None]])
        features = np.concatenate(values, axis=1).astype(np.float32)
        assert features.shape == (len(ids), len(RANK_FEATURES)), features.shape
        assert np.isfinite(features).all()
        assert all(info["eligible"][ids])
        return Retrieval(ids, features, tuple(sorted(seeds)), tuple(sorted(edges)), lexical_anchors,
                         prior_anchors, depth, dict(support), dict(mass), transitions)


def prepare_contexts(contexts):
    for context in contexts.values():
        edges = np.asarray(sorted(context["edges"]), dtype=np.int32)
        context["edge_arrays"] = (edges[:, 0], edges[:, 1])
    return contexts


def block_fold_sources(data, fold, folds, stable_int):
    signatures = {q["signature"] for q in data["queries"] if q["split"] == "train"
                  and stable_int(q["group"]) % folds == fold}
    return {n["id"] for n in data["nodes"] if n["signature"] in signatures}


def masked_contexts(contexts, sources, suffix):
    result = {}
    for name, context in contexts.items():
        edges = {(u, v) for u, v in context["edges"] if u not in sources}
        adjacency = [[] for _ in context["adjacency"]]
        incoming = [[] for _ in context["incoming"]]
        for u, v in sorted(edges):
            adjacency[u].append(v)
            incoming[v].append(u)
        result[name] = {"name": name + suffix, "edges": edges, "adjacency": adjacency, "incoming": incoming}
    return prepare_contexts(result)


def model_scores(model, features):
    global _PREDICTION_POOLS
    if model.get("scaler") is not None:
        features = model["scaler"].transform(features)
    if _PREDICTION_POOLS is None:
        _PREDICTION_POOLS = ThreadpoolController()
    with _PREDICTION_POOLS.limit(limits=1, user_api="openmp"):
        return model["estimator"].predict_proba(features)[:, 1]


def predicted_ranking(result, scores):
    order = np.lexsort((result.ids, -scores))
    return result.ids[order]
