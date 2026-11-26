"""Freeze the subject slice, source-block split and hidden proof boundaries."""
from __future__ import annotations
import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys

from .metamath import Statement


def canonical_document(statement, statements):
    types = {statements[h].expr[1]: statements[h].expr[0]
             for h in statement.hyps if statements[h].kind == "$f"}
    names = {}
    type_counts = Counter()

    def convert(expr):
        out = []
        for token in expr:
            if token in types:
                if token not in names:
                    typ = types[token]
                    names[token] = f"VAR_{typ}_{type_counts[typ]}"
                    type_counts[typ] += 1
                token = names[token]
            out.append(token)
        return " ".join(out)

    # Conclusion first makes alpha-renaming independent of hypothesis labels.
    goal = convert(statement.expr)
    hypotheses = [convert(statements[h].expr) for h in statement.hyps if statements[h].kind == "$e"]
    dv = sorted(tuple(sorted((names[a], names[b]))) for a, b in statement.distinct)
    text = "GOAL " + goal + "".join(" HYP " + h for h in hypotheses)
    text += "".join(f" DISTINCT {a} {b}" for a, b in dv)
    return text


def stable_int(value):
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scoped_v1.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    raw_manifest = json.loads(Path("data/raw/manifest.json").read_text(encoding="utf-8"))
    if raw_manifest["commit"] != config["source_commit"]:
        raise ValueError("Source commit differs from locked configuration")
    payload = pickle.loads(Path("data/processed/set.pkl").read_bytes())
    statements = payload["statements"]
    aset = set(payload["assertions"])
    core = [i for i in payload["assertions"]
            if (statements[i].chapter.startswith("ZF Set Theory") and statements[i].section in config["scope_set_sections"])
            or statements[i].chapter in config["scope_graph_chapters"]]
    core_set = set(core)
    node_ids = sorted(core_set | {j for i in core for j in statements[i].deps if j in aset})
    local = {i: j for j, i in enumerate(node_ids)}
    documents = [canonical_document(statements[i], statements) for i in node_ids]
    signatures = [hashlib.sha256(x.encode()).hexdigest() for x in documents]
    section_position = Counter()
    blocks = {}
    seen = set()
    queries = []
    exclusions = Counter()
    for i in core:
        s = statements[i]
        section_key = s.chapter + " / " + s.section
        block = section_position[section_key] // config["source_block_size"]
        section_position[section_key] += 1
        group = f"{section_key} / block-{block}"
        value = stable_int(f"{config['split_seed']}:{group}") % 1000000 / 1000000
        train_frac, val_frac, _ = config["split_fractions"]
        split = "train" if value < train_frac else "validation" if value < train_frac + val_frac else "test"
        blocks[group] = split
        if s.kind != "$p":
            exclusions["axiom_or_definition"] += 1
            continue
        gold = [local[j] for j in s.deps if j in aset]
        if len(gold) < config["query_min_premises"]:
            exclusions["too_few_direct_premises"] += 1
            continue
        signature = signatures[local[i]]
        if signature in seen:
            exclusions["duplicate_alpha_signature"] += 1
            continue
        seen.add(signature)
        queries.append({"qid": s.label, "node": local[i], "original_index": i,
                        "group": group, "split": split, "signature": signature,
                        "gold": gold, "topic": "graph" if s.chapter in config["scope_graph_chapters"] else "sets_relations_functions",
                        "chapter": s.chapter, "section": s.section})
    hidden_signatures = {q["signature"] for q in queries if q["split"] != "train"}
    hidden_nodes = {j for j, sig in enumerate(signatures) if sig in hidden_signatures}
    edges = [(local[i], local[j]) for i in core if local[i] not in hidden_nodes
             for j in statements[i].deps if j in aset]
    assert all(u > v for u, v in edges), "Reference graph must respect source order"
    assert not (hidden_nodes & {u for u, _ in edges})
    split_signatures = {split: {q["signature"] for q in queries if q["split"] == split}
                        for split in ["train", "validation", "test"]}
    assert not (split_signatures["train"] & split_signatures["test"])
    stats = {"source_commit": raw_manifest["commit"], "core_nodes": len(core),
             "boundary_nodes": len(node_ids) - len(core), "total_nodes": len(node_ids),
             "core_reference_edges_before_hiding": sum(sum(j in aset for j in statements[i].deps) for i in core),
             "searchable_reference_edges": len(edges), "hidden_outgoing_nodes": len(hidden_nodes),
             "essential_hypothesis_records": sum(sum(statements[h].kind == "$e" for h in statements[i].hyps) for i in node_ids),
             "query_count": len(queries), "query_splits": dict(Counter(q["split"] for q in queries)),
             "query_split_groups": {s: len({q["group"] for q in queries if q["split"] == s}) for s in split_signatures},
             "query_topics": dict(Counter(q["topic"] for q in queries)),
             "excluded": dict(exclusions),
             "core_sections": dict(Counter(statements[i].chapter + " / " + statements[i].section for i in core)),
             "gold_over_primary_budget": sum(len(q["gold"]) > config["primary_budget"] for q in queries),
             "gold_boundary_fraction": sum(sum(node_ids[j] not in core_set for j in q["gold"]) for q in queries) / sum(len(q["gold"]) for q in queries)}
    root = Path("data/processed/scoped")
    root.mkdir(parents=True, exist_ok=True)
    node_records = []
    for j, i in enumerate(node_ids):
        s = statements[i]
        node_records.append({"id": j, "original_index": i, "label": s.label,
                             "core": i in core_set, "kind": s.kind, "chapter": s.chapter,
                             "section": s.section, "line": s.line, "signature": signatures[j],
                             "text": documents[j], "hypothesis_count": sum(statements[h].kind == "$e" for h in s.hyps),
                             "distinct_count": len(s.distinct),
                             "source": f"https://github.com/metamath/set.mm/blob/{raw_manifest['commit']}/set.mm#L{s.line}"})
    for filename, rows in [("nodes.jsonl", node_records), ("queries_and_gold.jsonl", queries)]:
        (root / filename).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with (root / "reference_edges.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "relation"])
        writer.writerows((u, v, "proof_uses") for u, v in edges)
    dataset = {"nodes": node_records, "queries": queries, "edges": edges, "stats": stats, "config": config}
    with (root / "dataset.pkl").open("wb") as f:
        pickle.dump(dataset, f, protocol=5)
    (root / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    locked = [Path(args.config), Path("RESEARCH_DESIGN.md"), root / "nodes.jsonl", root / "queries_and_gold.jsonl", root / "reference_edges.csv"]
    lock = {"protocol": config["version"], "python": platform.python_version(),
            "hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in locked}}
    (root / "design_lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "core_sections"}, indent=2))


if __name__ == "__main__":
    main()
