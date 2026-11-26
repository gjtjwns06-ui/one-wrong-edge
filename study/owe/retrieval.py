"""Deterministic local retrieval and exact two-hop single-edge interventions."""
from __future__ import annotations
from collections import Counter
import numpy as np


def ranked(ids, scores):
    ids = np.asarray(sorted(ids), dtype=np.int32)
    return ids[np.lexsort((ids, -scores[ids]))]


def traverse(adjacency, seeds, hops=2, deleted=frozenset()):
    seen = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        following = {v for u in frontier for v in adjacency[u] if (u, v) not in deleted}
        frontier = following - seen
        seen.update(following)
    return seen


class Neighborhood:
    def __init__(self, adjacency, seeds, scores):
        self.adjacency = adjacency
        self.seeds = frozenset(int(s) for s in seeds)
        self.first_counts = Counter(v for u in self.seeds for v in adjacency[u])
        self.first = set(self.first_counts) - self.seeds
        self.support = self.first_counts.copy()
        for u in self.first:
            self.support.update(adjacency[u])
        self.reached = set(self.support) | self.seeds
        self.ranking = ranked(self.reached, scores)
        self.edges = sorted({(u, v) for u in self.seeds | self.first for v in adjacency[u]})
        self.depth = {v: 2 for v in self.reached}
        self.depth.update({v: 1 for v in self.first})
        self.depth.update({v: 0 for v in self.seeds})

    def lost_after_one(self, edge):
        u, v = edge
        delta = Counter({v: 1})
        if u in self.seeds and v not in self.seeds and self.first_counts[v] == 1:
            delta.update(self.adjacency[v])
        return {v for v, n in delta.items() if v not in self.seeds and self.support[v] <= n}

    def remove_one(self, edge, limit=40):
        lost = self.lost_after_one(edge)
        return np.asarray([v for v in self.ranking if v not in lost][:limit], dtype=np.int32)

    def remove_many(self, edges, scores, limit=40):
        if not edges:
            return self.ranking[:limit]
        if len(edges) == 1:
            return self.remove_one(edges[0], limit)
        return ranked(traverse(self.adjacency, self.seeds, 2, frozenset(edges)), scores)[:limit]


def metrics(ids, gold, budget, lengths=None):
    ids = ids[:budget]
    relevant = np.asarray([int(v) in gold for v in ids], dtype=np.float64)
    recall = float(relevant.sum() / len(gold))
    complete = float(relevant.sum() == len(gold))
    discounts = 1.0 / np.log2(np.arange(2, len(ids) + 2))
    ideal = float((1.0 / np.log2(np.arange(2, min(len(gold), budget) + 2))).sum())
    ndcg = float(relevant @ discounts / ideal) if ideal else 0.0
    tokens = int(lengths[ids].sum()) if lengths is not None else 0
    return {"recall": recall, "complete": complete, "ndcg": ndcg,
            "returned": len(ids), "formal_tokens": tokens}
