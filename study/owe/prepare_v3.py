"""Rebuild a verified scoped corpus with previously unindexed topic holdouts.

The full source is parsed only for proof-frame verification and source extraction;
the experimental graph is the small subject allowlist plus terminal references.
No previous learned model, graph perturbation, split, or effect label is reused.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import pickle
import platform
import time

from .metamath import Statement, parse_text
from .prepare import canonical_document, stable_int


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/restart_v3_data.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["output"])
    if (root / "dataset.pkl").exists():
        raise RuntimeError("Completed v3 dataset already exists; preserve it in a separate study")
    root.mkdir(parents=True, exist_ok=True)
    raw_manifest = json.loads(Path(config["source_manifest"]).read_text(encoding="utf-8"))
    assert raw_manifest["commit"] == config["source_commit"]
    source_path = Path(config["source_file"])
    raw = source_path.read_bytes()
    raw_sha = hashlib.sha256(raw).hexdigest()
    assert raw_sha == raw_manifest["files"][source_path.name]["sha256"]
    old = pickle.loads(Path(config["prior_dataset"]).read_bytes())
    old_signatures = {n["signature"] for n in old["nodes"]}
    old_query_signatures = {q["signature"] for q in old["queries"]}
    base = json.loads(Path(config["development_scope_config"]).read_text(encoding="utf-8"))

    started = time.perf_counter()
    print("Reparsing pinned raw set.mm and locally checking every stored proof...", flush=True)
    statements = parse_text(raw.decode("utf-8"), verify=True)
    parse_seconds = time.perf_counter() - started
    assertions = [i for i, s in enumerate(statements) if s.kind in ("$a", "$p") and s.expr[0] == "|-"]
    aset = set(assertions)
    proof_count = sum(s.kind == "$p" for s in statements)
    with (root / "parsed.pkl").open("wb") as f:
        pickle.dump({"statements": statements, "assertions": assertions,
                     "stats": {"raw_source_sha256": raw_sha, "proofs_checked_by_local_checker": proof_count,
                               "all_statements": len(statements), "assertion_nodes": len(assertions),
                               "seconds": parse_seconds}}, f, protocol=5)
    print(f"Raw verification complete: {proof_count} proofs; {parse_seconds:.1f}s", flush=True)

    development = [i for i in assertions
        if (statements[i].chapter.startswith("ZF Set Theory") and statements[i].section in base["scope_set_sections"])
        or statements[i].chapter in base["scope_graph_chapters"]]
    fresh = [i for i in assertions
        if statements[i].chapter in config["fresh_test_chapters"]
        or (statements[i].chapter.startswith("ZF Set Theory") and statements[i].section in config["fresh_test_zf_sections"])]
    assert len(development) == old["stats"]["core_nodes"]
    assert set(development).isdisjoint(fresh), "Fresh topic core must not overlap original core"
    core = sorted(set(development) | set(fresh))
    core_set = set(core)
    # Split from the preselected core documents first.  No held-out proof
    # dependency participates in the choice of an auxiliary boundary document.
    node_ids = core.copy()
    local = {i: j for j, i in enumerate(node_ids)}
    documents = {i: canonical_document(statements[i], statements) for i in node_ids}
    signatures = {i: hashlib.sha256(documents[i].encode()).hexdigest() for i in node_ids}
    fresh_signatures = {signatures[i] for i in fresh}
    positions = Counter()
    groups = {}
    for i in core:
        s = statements[i]
        section = s.chapter + " / " + s.section
        block = positions[section] // config["source_block_size"]
        positions[section] += 1
        groups[i] = section + f" / block-{block}"

    def topic(i):
        s = statements[i]
        if i in fresh_set:
            if s.chapter == "Words over a set": return "words"
            if s.chapter == "Reflexive and transitive closures of relations": return "relation_closure"
            return {"Equivalence relations and classes": "equivalence_relations",
                    "Partial and total orderings": "orderings",
                    "Relations and functions (cont.)": "relations_functions_cont"}[s.section]
        return "graph" if s.chapter in base["scope_graph_chapters"] else "sets_relations_functions"

    def query(i, split):
        s = statements[i]
        return {"qid": s.label, "node": local[i], "original_index": i,
                "group": groups[i], "split": split, "signature": signatures[i],
                "gold_original_indices": [j for j in s.deps if j in aset], "topic": topic(i),
                "chapter": s.chapter, "section": s.section,
                "previously_indexed_signature": signatures[i] in old_signatures,
                "previously_queried_signature": signatures[i] in old_query_signatures}

    fresh_set = set(fresh)
    exclusions = {"development": Counter(), "fresh_test": Counter()}
    queries = []
    seen = set()
    # Reserve the fresh holdout first; exclude ANY previously indexed signature.
    for i in fresh:
        s = statements[i]; sig = signatures[i]
        if s.kind != "$p":
            exclusions["fresh_test"]["axiom_or_definition"] += 1; continue
        if sig in old_signatures:
            exclusions["fresh_test"]["previously_indexed_signature"] += 1; continue
        if len([j for j in s.deps if j in aset]) < config["query_min_premises"]:
            exclusions["fresh_test"]["too_few_direct_premises"] += 1; continue
        if sig in seen:
            exclusions["fresh_test"]["duplicate_alpha_signature"] += 1; continue
        seen.add(sig)
        queries.append(query(i, "test"))
    for i in development:
        s = statements[i]; sig = signatures[i]
        if s.kind != "$p":
            exclusions["development"]["axiom_or_definition"] += 1; continue
        if sig in fresh_signatures:
            exclusions["development"]["fresh_topic_signature_alias"] += 1; continue
        if len([j for j in s.deps if j in aset]) < config["query_min_premises"]:
            exclusions["development"]["too_few_direct_premises"] += 1; continue
        if sig in seen:
            exclusions["development"]["duplicate_alpha_signature"] += 1; continue
        seen.add(sig)
        value = stable_int(f"{config['split_seed']}:{groups[i]}") % 1000000 / 1000000
        a, b, _ = config["development_split_fractions"]
        split = "rank_train" if value < a else "policy_train" if value < a + b else "validation"
        queries.append(query(i, split))
    hidden_signatures = fresh_signatures | {q["signature"] for q in queries if q["split"] in ("policy_train", "validation")}
    public_sources = [i for i in core if signatures[i] not in hidden_signatures]
    boundary_provenance = {}
    for i in public_sources:
        for j in statements[i].deps:
            if j in aset and j not in core_set:
                boundary_provenance.setdefault(j, []).append(i)
    node_ids = sorted(core_set | set(boundary_provenance))
    local = {i: j for j, i in enumerate(node_ids)}
    for i in boundary_provenance:
        documents[i] = canonical_document(statements[i], statements)
        signatures[i] = hashlib.sha256(documents[i].encode()).hexdigest()
    for q in queries:
        q["node"] = local[q["original_index"]]
        q["gold_in_index"] = [local[j] for j in q["gold_original_indices"] if j in local]
        q["gold"] = [local[j] if j in local else len(node_ids) + j for j in q["gold_original_indices"]]
        q["out_of_index_count"] = len(q["gold"]) - len(q["gold_in_index"])
    queries.sort(key=lambda q: q["node"])
    hidden_nodes = {local[i] for i in node_ids if signatures[i] in hidden_signatures}
    indexed_full_edges = [(local[i], local[j]) for i in core for j in statements[i].deps if j in local]
    edges = [(local[i], local[j]) for i in public_sources for j in statements[i].deps if j in aset]
    assert all(u > v for u, v in indexed_full_edges)
    assert hidden_nodes.isdisjoint(u for u, v in edges)
    assert {local[i] for i in fresh}.issubset(hidden_nodes)
    splits = {s: [q for q in queries if q["split"] == s] for s in ["rank_train", "policy_train", "validation", "test"]}
    assert all(splits[s] for s in splits)
    assert len(queries) == len({q["signature"] for q in queries})
    assert not ({q["signature"] for q in splits["test"]} & old_signatures)
    assert not ({q["group"] for q in splits["rank_train"]} & {q["group"] for q in splits["policy_train"]})
    assert not ({q["group"] for q in splits["rank_train"]} & {q["group"] for q in splits["validation"]})
    assert not ({q["group"] for q in splits["policy_train"]} & {q["group"] for q in splits["validation"]})
    for q in queries:
        assert all(i < q["original_index"] for i in q["gold_original_indices"])
        assert all(v < q["node"] for v in q["gold_in_index"])
    assert all(signatures[i] not in hidden_signatures for parents in boundary_provenance.values() for i in parents)
    node_records = []
    for j, i in enumerate(node_ids):
        s = statements[i]
        node_records.append({"id": j, "original_index": i, "label": s.label,
            "core": i in core_set, "fresh_topic_core": i in fresh_set,
            "kind": s.kind, "chapter": s.chapter, "section": s.section, "line": s.line,
            "signature": signatures[i], "text": documents[i],
            "hypothesis_count": sum(statements[h].kind == "$e" for h in s.hyps),
            "distinct_count": len(s.distinct),
            "previously_indexed_signature": signatures[i] in old_signatures,
            "source": f"https://github.com/metamath/set.mm/blob/{config['source_commit']}/set.mm#L{s.line}"})
    stats = {"version": config["version"], "source_commit": config["source_commit"],
        "raw_source_sha256": raw_sha, "proofs_checked_by_local_checker": proof_count,
        "raw_parse_and_verification_seconds": parse_seconds,
        "development_core_nodes": len(development), "fresh_topic_core_nodes": len(fresh),
        "core_nodes": len(core), "boundary_nodes": len(node_ids) - len(core), "total_nodes": len(node_ids),
        "core_reference_edges_before_hiding": sum(sum(j in aset for j in statements[i].deps) for i in core),
        "indexed_core_reference_edges_before_hiding": len(indexed_full_edges), "searchable_reference_edges": len(edges),
        "hidden_outgoing_nodes": len(hidden_nodes), "all_fresh_topic_core_proofs_hidden": True,
        "query_count": len(queries), "query_splits": {s: len(qs) for s, qs in splits.items()},
        "query_split_groups": {s: len({q['group'] for q in qs}) for s, qs in splits.items()},
        "query_split_topics": {s: dict(Counter(q['topic'] for q in qs)) for s, qs in splits.items()},
        "test_questions_with_prior_indexed_signature": sum(q['previously_indexed_signature'] for q in splits['test']),
        "test_questions_with_prior_query_signature": sum(q['previously_queried_signature'] for q in splits['test']),
        "excluded": {key: dict(value) for key, value in exclusions.items()},
        "core_sections": dict(Counter(statements[i].chapter + ' / ' + statements[i].section for i in core)),
        "fresh_topic_core_sections": dict(Counter(statements[i].chapter + ' / ' + statements[i].section for i in fresh)),
        "gold_over_primary_budget": {s: sum(len(q['gold']) > config['primary_budget'] for q in qs) for s, qs in splits.items()},
        "gold_boundary_fraction": sum(sum(i not in core_set for i in q['gold_original_indices']) for q in queries) / sum(len(q['gold']) for q in queries),
        "boundary_derived_only_from_public_proofs": True,
        "out_of_index_gold": {s: {'references': sum(q['out_of_index_count'] for q in qs),
            'all_references': sum(len(q['gold']) for q in qs),
            'macro_missing_fraction': sum(q['out_of_index_count']/len(q['gold']) for q in qs)/len(qs),
            'questions_with_missing_references': sum(q['out_of_index_count'] > 0 for q in qs)} for s, qs in splits.items()},
        "unindexed_gold_id_encoding": "len(indexed_nodes) + original_statement_index; retained in denominators and never candidate IDs",
        "novelty_note": "Test statement signatures were absent from all prior indexed nodes. The same raw library was globally parsed previously; this is fresh-topic retrieval evaluation, not a new source or fully chronological model-training evaluation."}
    dataset = {"nodes": node_records, "queries": queries, "edges": edges, "stats": stats, "config": config}
    with (root / "dataset.pkl").open("wb") as f: pickle.dump(dataset, f, protocol=5)
    for name, rows in [("nodes.jsonl", node_records), ("queries_and_gold.jsonl", queries)]:
        (root / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding="utf-8")
    with (root / "reference_edges.csv").open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(['source', 'target', 'relation'])
        writer.writerows((u, v, 'proof_uses') for u, v in edges)
    write_json(root / "stats.json", stats)
    write_json(root / "boundary_provenance.json", {str(local[j]): {'original_index': j,
        'public_parent_original_indices': parents} for j, parents in sorted(boundary_provenance.items())})
    inputs = [config_path, source_path, Path(config["source_manifest"]), Path(config["prior_dataset"]),
              Path(config["development_scope_config"]), Path("owe/metamath.py"), Path("owe/prepare.py"), Path(__file__)]
    old_query_file = Path(config["prior_dataset"]).parent / "queries_and_gold.jsonl"
    if old_query_file.exists(): inputs.append(old_query_file)
    write_json(root / "manifest.json", {"version": config["version"], "created_date": "2026-11-25",
        "python": platform.python_version(), "inputs": {p.resolve().relative_to(Path.cwd().resolve()).as_posix(): sha(p) for p in inputs},
        "outputs": {name: sha(root / name) for name in ['dataset.pkl', 'nodes.jsonl', 'queries_and_gold.jsonl', 'reference_edges.csv', 'stats.json', 'boundary_provenance.json']},
        "old_gold_or_model_outputs_reused": False,
        "all_previous_index_signatures_excluded_from_test": True,
        "boundary_derived_only_from_public_unhidden_source_proofs": True,
        "complete_raw_verification_passed": True,
        "proofs_checked_by_local_checker": proof_count,
        "local_checker_note": "Research implementation checks substitutions, essential hypotheses, distinct variables and final stacks; it is not an independently certified upstream checker."})
    print(json.dumps({k: v for k, v in stats.items() if k not in ('core_sections', 'fresh_topic_core_sections')}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
