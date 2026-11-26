"""Exhaustive finite-model audit of omitted Euler-trail hypotheses.

This is a semantic audit, separate from the formal proof-attribution experiment.
"""
from __future__ import annotations
from collections import Counter
from functools import lru_cache
import itertools
import json
from pathlib import Path


def connected(vertices, edges):
    vertices = set(vertices)
    if not vertices:
        return True
    seen = {min(vertices)}
    while True:
        more = {b for a, b in edges if a in seen} | {a for a, b in edges if b in seen}
        expanded = seen | more
        if expanded == seen:
            return vertices <= seen
        seen = expanded


def trail_witness(n, edges):
    adjacency = [set() for _ in range(n)]
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    odd = [v for v in range(n) if len(adjacency[v]) % 2]
    active = [v for v in range(n) if adjacency[v]]
    start = odd[0] if odd else active[0] if active else 0
    stack = [start]
    reversed_path = []
    while stack:
        u = stack[-1]
        if adjacency[u]:
            v = min(adjacency[u])
            adjacency[u].remove(v)
            adjacency[v].remove(u)
            stack.append(v)
        else:
            reversed_path.append(stack.pop())
    path = reversed_path[::-1]
    remaining = set(edges)
    for a, b in zip(path, path[1:]):
        edge = tuple(sorted((a, b)))
        if edge not in remaining:
            return None
        remaining.remove(edge)
    return path if not remaining and len(path) == len(edges) + 1 else None


def exact_trail_exists(n, edges):
    incident = [[] for _ in range(n)]
    for index, (a, b) in enumerate(edges):
        incident[a].append((index, b))
        incident[b].append((index, a))

    @lru_cache(None)
    def search(vertex, remaining):
        if remaining == 0:
            return True
        return any(remaining & (1 << i) and search(other, remaining ^ (1 << i))
                   for i, other in incident[vertex])
    return any(search(v, (1 << len(edges)) - 1) for v in range(n))


def main():
    config = json.loads(Path("configs/scoped_v1.json").read_text(encoding="utf-8"))
    by_n = []
    failures = {}
    rule_counts = Counter()
    checked_dfs = 0
    for n in range(1, config["finite_graph_max_vertices"] + 1):
        pairs = list(itertools.combinations(range(n), 2))
        counts = Counter()
        for mask in range(1 << len(pairs)):
            edges = [e for i, e in enumerate(pairs) if mask & (1 << i)]
            degree = Counter(v for e in edges for v in e)
            odd = sum(degree[v] % 2 for v in range(n))
            support_connected = connected([v for v in range(n) if degree[v]], edges)
            all_connected = connected(range(n), edges)
            witness = trail_witness(n, edges)
            truth = witness is not None
            if n <= 5:
                assert exact_trail_exists(n, edges) == truth
                checked_dfs += 1
            criterion = support_connected and odd in (0, 2)
            assert criterion == truth, (n, edges, witness)
            rules = {"connected_only_sufficient": (all_connected, truth),
                     "parity_only_sufficient": (odd in (0, 2), truth),
                     "all_vertices_connected_parity_equivalent": (all_connected and odd in (0, 2), truth)}
            counts["graphs"] += 1
            counts["has_trail"] += int(truth)
            for name, (prediction, target) in rules.items():
                # Sufficient-condition claims count only false positives.
                false_positive = prediction and not target
                false_negative = not prediction and target
                rule_counts[name + "/claims"] += int(prediction)
                rule_counts[name + "/false_positives"] += int(false_positive)
                if name.endswith("equivalent"):
                    rule_counts[name + "/false_negatives"] += int(false_negative)
                if false_positive or (name.endswith("equivalent") and false_negative):
                    failures.setdefault(name, {"n": n, "edges": edges, "odd_vertices": [v for v in range(n) if degree[v] % 2],
                                               "has_trail": truth, "witness": witness,
                                               "error": "false_positive" if false_positive else "false_negative"})
        by_n.append({"n": n, **dict(counts)})
    examples = {"connected_star_without_trail": (4, [(0, 1), (0, 2), (0, 3)]),
                "two_disjoint_triangles": (6, [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]),
                "edge_and_isolated_vertex": (3, [(0, 1)])}
    report = {"definition": "simple undirected labelled graphs; n>=1; a zero-edge trail is allowed",
              "total_graphs": sum(x["graphs"] for x in by_n), "max_n": config["finite_graph_max_vertices"],
              "criterion_vs_checked_witness_mismatches": 0,
              "additional_exact_DFS_cross_checks_n_at_most_5": checked_dfs,
              "by_n": by_n, "rule_counts": dict(rule_counts), "first_counterexamples": failures,
              "named_examples": {name: {"n": n, "edges": edges, "witness": trail_witness(n, edges)}
                                 for name, (n, edges) in examples.items()}}
    root = Path("runs/scoped_v1")
    root.mkdir(parents=True, exist_ok=True)
    (root / "finite_graph_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("named_examples", "first_counterexamples")}, indent=2))


if __name__ == "__main__":
    main()
