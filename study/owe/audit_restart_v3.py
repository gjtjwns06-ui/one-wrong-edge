"""Independent data/provenance and materialized-intervention audit for restart-v3.

This audit reads the freshly reparsed formal source, but does not repeat its
expensive proof verification.  It never modifies prior study artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data/processed/restart_v3'
RUN = BASE / 'runs/restart_v3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return pickle.loads(Path(path).read_bytes())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def public_query(query):
    """Minimal observable formal-query identifiers; no labels or gold counts."""
    return {key: query[key] for key in ('qid', 'node', 'signature')}


def check_context(context, nodes, hidden):
    edges = set(map(tuple, context['edges']))
    assert len(edges) == len(context['edges'])
    assert all(0 <= v < u < len(nodes) for u, v in edges)
    assert hidden.isdisjoint(u for u, _ in edges)
    assert all(nodes[u]['core'] for u, _ in edges)
    assert len(context['adjacency']) == len(nodes) == len(context['incoming'])
    adjacency_edges = {(u, v) for u, row in enumerate(context['adjacency']) for v in row}
    incoming_edges = {(u, v) for v, row in enumerate(context['incoming']) for u in row}
    assert edges == adjacency_edges == incoming_edges
    assert sum(map(len, context['adjacency'])) == len(edges)
    assert sum(map(len, context['incoming'])) == len(edges)
    if 'edge_arrays' in context:
        assert set(zip(*map(list, context['edge_arrays']))) == edges


def independent_metrics(ids, gold, budget, lengths):
    top = list(map(int, ids[:budget]))
    assert len(top) == len(set(top))
    gold = set(map(int, gold))
    matches = np.asarray([v in gold for v in top], dtype=np.float64)
    denominators = np.log2(np.arange(len(top), dtype=np.float64) + 2)
    ideal = np.sum(1 / np.log2(np.arange(min(len(gold), budget), dtype=np.float64) + 2))
    return {'recall': float(matches.sum() / len(gold)),
            'complete': float(gold.issubset(top)),
            'ndcg': float(np.sum(matches / denominators) / ideal),
            'returned': len(top), 'formal_tokens': int(sum(lengths[v] for v in top))}


def check_ids(ids, query, nodes):
    ids = list(map(int, ids))
    assert len(ids) == len(set(ids))
    assert all(0 <= v < query['node'] and nodes[v]['signature'] != query['signature'] for v in ids)


def audit_prior_files():
    manifest = json.loads((BASE / 'ARTIFACTS_V2.json').read_text(encoding='utf-8'))
    for name, record in manifest['files'].items():
        assert sha(BASE / name) == record['sha256'], f'Prior study changed: {name}'
    return len(manifest['files'])


def audit_data():
    data = load(DATA / 'dataset.pkl')
    nodes, queries = data['nodes'], data['queries']
    count = len(nodes)
    assert [n['id'] for n in nodes] == list(range(count))
    original_ids = [n['original_index'] for n in nodes]
    assert original_ids == sorted(set(original_ids))
    local = {original: i for i, original in enumerate(original_ids)}
    assert len(queries) == len({q['qid'] for q in queries}) == len({q['signature'] for q in queries})
    assert all(hashlib.sha256(n['text'].encode()).hexdigest() == n['signature'] for n in nodes)
    splits = {s: [q for q in queries if q['split'] == s]
              for s in ['rank_train', 'policy_train', 'validation', 'test']}
    assert all(splits.values())
    assert sum(map(len, splits.values())) == len(queries)
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert {q['group'] for q in splits[a]}.isdisjoint(q['group'] for q in splits[b]), (a, b)
            assert {q['signature'] for q in splits[a]}.isdisjoint(q['signature'] for q in splits[b]), (a, b)
    old = load(BASE / data['config']['prior_dataset'])
    old_signatures = {n['signature'] for n in old['nodes']}
    assert old_signatures.isdisjoint(q['signature'] for q in splits['test'])
    assert all(not q['previously_indexed_signature'] and not q['previously_queried_signature'] for q in splits['test'])

    parsed = load(DATA / 'parsed.pkl')
    statements = parsed['statements']
    assertions = set(parsed['assertions'])
    assert assertions == {i for i, s in enumerate(statements)
                          if s.kind in ('$a', '$p') and s.expr[0] == '|-'}
    from .prepare import canonical_document
    for n in nodes:
        statement = statements[n['original_index']]
        assert canonical_document(statement, statements) == n['text']
        assert statement.label == n['label'] and statement.kind == n['kind']
        assert sum(statements[h].kind == '$e' for h in statement.hyps) == n['hypothesis_count']
        assert len(statement.distinct) == n['distinct_count']
    core = {n['original_index'] for n in nodes if n['core']}
    fresh = {n['original_index'] for n in nodes if n['fresh_topic_core']}
    cfg = data['config']
    actual_fresh = {i for i in assertions
                   if statements[i].chapter in cfg['fresh_test_chapters']
                   or (statements[i].chapter.startswith('ZF Set Theory')
                       and statements[i].section in cfg['fresh_test_zf_sections'])}
    assert fresh == actual_fresh and fresh <= core
    fresh_signatures = {nodes[local[i]]['signature'] for i in fresh}
    hidden_signatures = fresh_signatures | {q['signature'] for q in queries
                                          if q['split'] in ('policy_train', 'validation')}
    hidden = {i for i, n in enumerate(nodes) if n['signature'] in hidden_signatures}
    public_originals = {i for i in core if nodes[local[i]]['signature'] not in hidden_signatures}
    edges = set(map(tuple, data['edges']))
    assert len(edges) == len(data['edges'])
    assert all(0 <= v < u < count for u, v in edges)
    assert hidden.isdisjoint(u for u, _ in edges)
    expected_edges = {(local[i], local[j]) for i in public_originals
                      for j in statements[i].deps if j in assertions}
    assert edges == expected_edges
    parents = defaultdict(set)
    for i in public_originals:
        for j in statements[i].deps:
            if j in assertions and j not in core:
                parents[j].add(i)
    assert set(original_ids) == core | set(parents)
    provenance = json.loads((DATA / 'boundary_provenance.json').read_text(encoding='utf-8'))
    assert {int(k) for k in provenance} == {i for i, n in enumerate(nodes) if not n['core']}
    for local_id, record in provenance.items():
        n = nodes[int(local_id)]
        assert n['original_index'] == record['original_index']
        assert set(record['public_parent_original_indices']) == parents[n['original_index']]
        assert all(parent in public_originals for parent in record['public_parent_original_indices'])

    out_of_index = Counter()
    for q in queries:
        n = nodes[q['node']]
        assert n['original_index'] == q['original_index'] and n['signature'] == q['signature']
        assert n['label'] == q['qid']
        gold_original = sorted(j for j in statements[q['original_index']].deps if j in assertions)
        assert sorted(q['gold_original_indices']) == gold_original
        gold = sorted(local[j] if j in local else count + j for j in gold_original)
        assert sorted(q['gold']) == gold
        assert sorted(q['gold_in_index']) == sorted(local[j] for j in gold_original if j in local)
        assert all(j < q['original_index'] for j in gold_original)
        assert all(v < q['node'] for v in q['gold_in_index'])
        missing = sum(v >= count for v in gold)
        assert missing == q['out_of_index_count']
        out_of_index[q['split']] += missing
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['outputs'].items():
        assert sha(DATA / name) == expected, name
    for name, expected in manifest['inputs'].items():
        assert sha(BASE / name) == expected, name
    return data, hidden, {'queries': {s: len(qs) for s, qs in splits.items()},
                         'fresh_test_signatures_absent_all_prior_index_nodes': len(splits['test']),
                         'split_group_and_signature_disjoint': True,
                         'hidden_sources': len(hidden), 'reference_edges_checked': len(edges),
                         'public_boundary_provenance_records_checked': len(provenance),
                         'full_gold_and_unindexed_denominators_verified': True,
                         'unindexed_gold_counts': dict(out_of_index),
                         'fresh_raw_verification_receipt_present': manifest['complete_raw_verification_passed']}


def audit_experiment(data, hidden):
    from sklearn.base import clone
    from .restart_v3 import load_study, infer
    from .restart_support import context_from_edges, query_context, TEXT_COLUMNS, GRAPH_COLUMNS
    from .sequential_retriever import model_scores, predicted_ranking
    from .sequential_interventions import edge_features

    assert (RUN / 'run_complete.json').exists(), 'Complete and freeze the new test before final audit'
    complete = json.loads((RUN / 'run_complete.json').read_text())
    lock = json.loads((RUN / 'test_lock.json').read_text())
    assert lock['selected_before_test'] and lock['fresh_signature_test']
    for name, expected in lock['files'].items():
        assert sha(BASE / name) == expected, name
    config, loaded_data, contexts, engine = load_study()
    assert loaded_data['nodes'] == data['nodes']
    assert loaded_data['queries'] == data['queries']
    variant = json.loads((RUN / 'stage1_selection.json').read_text())['variant']
    ranker = load(RUN / 'ranker.pkl')
    text_ranker = load(RUN / 'text_ranker.pkl')
    models = load(RUN / 'utility_models.pkl')
    policy_selection = json.loads((RUN / 'stage3_selection.json').read_text())
    assert policy_selection['selected_before_test']
    policies = policy_selection['policies']
    assert set(TEXT_COLUMNS).isdisjoint(GRAPH_COLUMNS)
    assert sorted(TEXT_COLUMNS + GRAPH_COLUMNS) == list(range(42))
    np.testing.assert_array_equal(text_ranker['estimator'].columns, TEXT_COLUMNS)

    index = json.loads((RUN / 'index_manifest.json').read_text())
    assert sha(BASE / 'RESTART_V3_PROTOCOL.md') == index['protocol_sha256']
    assert sha(BASE / 'configs/restart_v3.json') == index['config_sha256']
    assert sha(DATA / 'dataset.pkl') == index['dataset_sha256']
    assert sha(DATA / 'manifest.json') == index['data_manifest_sha256']
    expected_fit = [i for i in range(len(data['nodes'])) if i not in hidden]
    assert index['vectorizer_fitted_nodes'] == expected_fit
    reference_vectorizer = clone(engine.vectorizer)
    reference_vectorizer.fit([data['nodes'][i]['text'] for i in expected_fit])
    assert reference_vectorizer.vocabulary_ == engine.vectorizer.vocabulary_
    np.testing.assert_array_equal(reference_vectorizer.idf_, engine.vectorizer.idf_)
    assert set(map(tuple, data['edges'])) == set(contexts['clean']['edges'])
    for context in contexts.values():
        check_context(context, data['nodes'], hidden)

    rank_receipt = json.loads((RUN / 'stage2_training_receipt.json').read_text())
    policy_receipt = json.loads((RUN / 'stage3_training_receipt.json').read_text())
    rank_queries = [q for q in data['queries'] if q['split'] == 'rank_train']
    policy_queries = [q for q in data['queries'] if q['split'] == 'policy_train']
    assert set(rank_receipt['rank_qids']) == {q['qid'] for q in rank_queries}
    assert set(rank_receipt['rank_groups']) == {q['group'] for q in rank_queries}
    assert set(policy_receipt['policy_qids']) == {q['qid'] for q in policy_queries}
    assert set(policy_receipt['policy_groups']) == {q['group'] for q in policy_queries}
    assert set(rank_receipt['rank_qids']).isdisjoint(policy_receipt['policy_qids'])
    assert set(rank_receipt['rank_groups']).isdisjoint(policy_receipt['policy_groups'])
    assert policy_receipt['frozen_ranker_sha256'] == sha(RUN / 'ranker.pkl')
    fit_rows = pd.read_csv(RUN / 'ranker_training_cases.csv')
    assert set(fit_rows.qid) == set(rank_receipt['rank_qids'])
    assert set(fit_rows.condition) == set(config['training_conditions'])
    policy_cases = load(RUN / 'policy_training_cases.pkl')
    assert {c['qid'] for c in policy_cases} == set(policy_receipt['policy_qids'])
    assert {c['condition'] for c in policy_cases} == set(config['policy_conditions'])
    assert len(policy_cases) == len(policy_queries) * len(config['policy_conditions'])
    policy_x = np.concatenate([c['features'] for c in policy_cases])
    policy_y = np.concatenate([c['benefits'][:, 1] for c in policy_cases])
    policy_wrong = np.asarray([e not in contexts['clean']['edges']
                               for c in policy_cases for e in c['edges']], dtype=np.int8)
    with np.load(RUN / 'utility_training.npz') as training:
        np.testing.assert_array_equal(training['features'], policy_x)
        np.testing.assert_array_equal(training['benefits'], policy_y)
        np.testing.assert_array_equal(training['wrongness'], policy_wrong)
        assert np.isfinite(training['weights']).all() and (training['weights'] > 0).all()
    assert np.isfinite(policy_x).all() and np.isfinite(policy_y).all()
    assert all(len(c['edges']) <= config['utility_candidate_cap'] for c in policy_cases)
    for model in models.values():
        if model.get('scaler') is not None:
            assert np.isfinite(model['scaler'].mean_).all()
            assert np.isfinite(model['scaler'].scale_).all()

    test_queries = [q for q in data['queries'] if q['split'] == 'test']
    qmap = {q['qid']: q for q in test_queries}
    frame = pd.read_csv(RUN / 'results.csv')
    assert len(frame) == complete['measurement_rows']
    assert frame.qid.nunique() == len(test_queries) == complete['test_queries']
    assert set(frame.qid) == set(qmap)
    assert set(frame.condition) == set(config['test_conditions'])
    assert frame.condition.nunique() == complete['conditions']
    assert frame.method.nunique() == complete['methods']
    assert set(frame.budget) == set(config['output_budgets'])
    assert not frame.duplicated(['qid', 'condition', 'method', 'budget']).any()
    numeric = frame[['recall', 'ndcg', 'complete', 'returned', 'formal_tokens',
                     'delta_vs_none', 'candidate_coverage', 'candidate_ceiling', 'deletions']]
    assert np.isfinite(numeric.to_numpy()).all()
    assert ((frame.recall >= 0) & (frame.recall <= 1)).all()
    assert ((frame.ndcg >= 0) & (frame.ndcg <= 1 + 1e-12)).all()
    assert (frame.returned <= frame.budget).all()
    # The legacy two-hop neighborhood has no256-document cap.  Its pool is
    # audited separately rather than treating it as a matched-budget method.
    assert (frame.loc[frame.method != 'old_local', 'candidates'] <= config['candidate_cap']).all()
    assert (frame.loc[frame.method == 'proof_small64', 'candidates'] <= 64).all()
    assert (frame.deletions >= frame.wrong_deletions).all()
    assert (frame.wrong_deletions >= 0).all()
    rowmap = {(r.qid, r.condition, r.method, int(r.budget)): r for r in frame.itertuples(index=False)}
    rows_checked = 0
    sampled_cases = []
    rng = np.random.default_rng(20261125)
    for condition in config['test_conditions']:
        cases = load(RUN / f'test_{condition}_cases.pkl')
        assert {c['qid'] for c in cases} == set(qmap) and len(cases) == len(qmap)
        case_map = {c['qid']: c for c in cases}
        picked = set(rng.choice(len(cases), size=min(6, len(cases)), replace=False).tolist())
        with gzip.open(RUN / f'test_{condition}_rankings.jsonl.gz', 'rt', encoding='utf-8') as stream:
            records = [json.loads(line) for line in stream]
        assert len(records) == len(qmap) and {r['qid'] for r in records} == set(qmap)
        record_map = {r['qid']: r for r in records}
        for record in records:
            query = qmap[record['qid']]
            gold = set(query['gold'])
            assert record['condition'] == condition
            check_ids(record['candidates'], query, data['nodes'])
            assert len(record['candidates']) <= config['candidate_cap']
            hits = len(set(record['candidates']) & gold)
            case = case_map[query['qid']]
            assert len(case['edges']) <= config['utility_candidate_cap']
            assert np.isfinite(case['features']).all() and np.isfinite(case['benefits']).all()
            assert np.isfinite(case['wrongness_scores']).all()
            assert all(0 <= p <= 1 for p in case['wrongness_scores'])
            assert list(case['source_supported']) == [tuple(e) in contexts['clean']['edges'] for e in case['edges']]
            for method, ids in record['rankings'].items():
                check_ids(ids, query, data['nodes'])
                selected = list(map(tuple, record['actions'].get(method, [])))
                assert len(selected) == len(set(selected))
                if method in policies:
                    assert len(selected) <= policies[method]['budget']
                for budget in config['output_budgets']:
                    row = rowmap[query['qid'], condition, method, budget]
                    assert row.group == query['group'] and row.topic == query['topic']
                    measured = independent_metrics(ids, gold, budget, engine.lengths)
                    for name, value in measured.items():
                        assert abs(getattr(row, name) - value) < 1e-10, (condition, query['qid'], method, budget, name)
                    base_recall = independent_metrics(record['rankings']['proof_ranker'], gold, budget, engine.lengths)['recall']
                    assert abs(row.delta_vs_none - (measured['recall'] - base_recall)) < 1e-10
                    if method not in ('lexical', 'old_local', 'proof_small64', 'source_verified'):
                        assert abs(row.candidate_coverage - hits / len(gold)) < 1e-10
                        assert abs(row.candidate_ceiling - min(budget, hits) / len(gold)) < 1e-10
                    assert 0 <= row.candidate_ceiling <= row.candidate_coverage + 1e-10 <= 1 + 1e-10
                    assert row.candidate_ceiling <= budget / len(gold) + 1e-10
                    assert row.deletions == len(selected)
                    assert row.wrong_deletions == sum(e not in contexts['clean']['edges'] for e in selected)
                    if method == 'oracle1' and budget == config['primary_budget']:
                        assert row.delta_vs_none >= -1e-10
                    rows_checked += 1
        for i in sorted(picked):
            case = cases[i]
            sampled_cases.append((condition, qmap[case['qid']], case, record_map[case['qid']]))
    assert rows_checked == len(frame)

    # Only these observable fields are given to production inference.  No gold,
    # gold count, source-support target, or corruption flag is passed in query.
    prefix_checks = single_replays = inference_replays = nonempty_actions = 0
    timings = []
    for condition, query, case, record in sampled_cases:
        observable = public_query(query)
        engine.query_cache.pop(query['qid'], None)
        context = query_context(engine, observable, contexts, condition, variant)
        check_context(context, data['nodes'], hidden)
        assert all(tuple(e) in context['edges'] for selected in record['actions'].values() for e in selected)
        if condition.startswith('local'):
            assert context['audit_changes'] == case['local_changes']
            assert len(context['edges'] - contexts['clean']['edges']) <= config['local_injection_count']
        base = engine.retrieve(observable, context, variant)
        np.testing.assert_array_equal(base.ids, case['base_candidates'])
        base_scores = model_scores(ranker, base.features)
        np.testing.assert_array_equal(base_scores, case['base_scores'])
        np.testing.assert_array_equal(predicted_ranking(base, base_scores)[:40], record['rankings']['proof_ranker'])
        computed_edge_features = edge_features(engine, observable, context, base, base_scores, case['edges'])
        np.testing.assert_array_equal(computed_edge_features, case['features'])
        info = engine.query_info(observable)
        legacy_seeds = set(map(int, info['lexical'][:4]))
        legacy_first = {v for u in legacy_seeds for v in context['adjacency'][u] if info['eligible'][v]}
        legacy_sources = legacy_seeds | legacy_first
        legacy_edges = {(u, v) for u in legacy_sources for v in context['adjacency'][u] if info['eligible'][v]}
        legacy_ids = legacy_seeds | legacy_first | {v for u in legacy_first for v in context['adjacency'][u] if info['eligible'][v]}
        legacy_ranking = sorted(legacy_ids, key=lambda v: (-info['scores'][v], v))
        np.testing.assert_array_equal(legacy_ranking[:40], record['rankings']['old_local'])
        saved_cap = engine.config['candidate_cap']
        engine.config['candidate_cap'] = 64
        try:
            small = engine.retrieve(observable, context, {**variant, 'lexical': 32, 'prior': 16})
        finally:
            engine.config['candidate_cap'] = saved_cap
        verified = engine.retrieve(observable, contexts['clean'], variant) if condition.startswith(('random', 'hard', 'local')) else base
        if condition.startswith('mixed'):
            verified_context = context_from_edges(data['nodes'], context['edges'] & contexts['clean']['edges'],
                                                   f'audit-verified/{condition}/{query["qid"]}')
            verified = engine.retrieve(observable, verified_context, variant)
        direct_pools = {'lexical': (info['lexical'][:config['candidate_cap']], 0),
                        'old_local': (legacy_ranking, len(legacy_edges)),
                        'proof_small64': (small.ids, small.transitions),
                        'source_verified': (verified.ids, verified.transitions)}
        for method, (pool, transitions) in direct_pools.items():
            check_ids(pool, query, data['nodes'])
            hits = len(set(map(int, pool)) & set(query['gold']))
            for budget in config['output_budgets']:
                row = rowmap[query['qid'], condition, method, budget]
                assert row.candidates == len(pool) and row.transitions == transitions
                assert abs(row.candidate_coverage - hits / len(query['gold'])) < 1e-10
                assert abs(row.candidate_ceiling - min(budget, hits) / len(query['gold'])) < 1e-10
        prefix = context_from_edges(data['nodes'], {(u, v) for u, v in context['edges'] if u < query['node']},
                                    f'audit-prefix/{condition}/{query["qid"]}')
        prefix_base = engine.retrieve(observable, prefix, variant)
        np.testing.assert_array_equal(prefix_base.ids, base.ids)
        np.testing.assert_array_equal(prefix_base.features, base.features)
        assert prefix_base.seeds == base.seeds and prefix_base.edges == base.edges
        prefix_edge_features = edge_features(engine, observable, prefix, prefix_base, base_scores, case['edges'])
        np.testing.assert_array_equal(prefix_edge_features, computed_edge_features)
        prefix_checks += 1
        for i in sorted({0, len(case['edges']) // 2, len(case['edges']) - 1}):
            if i < 0 or not case['edges']:
                continue
            edge = tuple(case['edges'][i])
            materialized = context_from_edges(data['nodes'], context['edges'] - {edge},
                f'audit-single/{condition}/{query["qid"]}/{i}')
            changed = engine.retrieve(observable, materialized, variant)
            ranking = predicted_ranking(changed, model_scores(ranker, changed.features))[:40]
            np.testing.assert_array_equal(ranking, case['single_rankings'][i])
            for j, budget in enumerate(config['output_budgets']):
                gain = independent_metrics(ranking, query['gold'], budget, engine.lengths)['recall']
                gain -= independent_metrics(record['rankings']['proof_ranker'], query['gold'], budget, engine.lengths)['recall']
                assert abs(gain - case['benefits'][i, j]) < 1e-7
            single_replays += 1
        engine.query_cache.pop(query['qid'], None)
        baseline = infer(engine, observable, context, variant, ranker)
        np.testing.assert_array_equal(baseline['ranking'], record['rankings']['proof_ranker'])
        for name, specification in policies.items():
            engine.query_cache.pop(query['qid'], None)
            actual = infer(engine, observable, context, variant, ranker,
                           {**specification, 'condition_name': condition}, models)
            np.testing.assert_array_equal(actual['ranking'], record['rankings'][name])
            actions = list(map(tuple, actual['actions']))
            assert actions == list(map(tuple, record['actions'][name]))
            physical = context_from_edges(data['nodes'], context['edges'] - set(actions),
                f'audit-policy/{condition}/{query["qid"]}/{name}')
            changed = engine.retrieve(observable, physical, variant)
            ranking = predicted_ranking(changed, model_scores(ranker, changed.features))[:40]
            np.testing.assert_array_equal(ranking, actual['ranking'])
            inference_replays += 1
            nonempty_actions += bool(actions)
            timings.append({'qid': query['qid'], 'condition': condition, 'policy': name,
                            'ranker_seconds': baseline['seconds'], 'policy_seconds': actual['seconds'],
                            'deletions': len(actions)})
        engine.trim_cache()
    pd.DataFrame(timings).to_csv(RUN / 'inference_timings.csv', index=False)
    return {'all_frozen_file_hashes_valid': len(lock['files']),
            'fresh_idf_fit_reconstructed_without_heldout_nodes': len(expected_fit),
            'global_graph_contexts_checked': len(contexts),
            'rank_policy_training_queries_disjoint': True,
            'rank_training_queries': len(rank_queries), 'policy_training_queries': len(policy_queries),
            'policy_effect_targets_and_features_reconstructed': len(policy_y),
            'text_ranker_excludes_graph_columns': True,
            'all_metric_rows_independently_recalculated': rows_checked,
            'candidate_and_output_eligibility_checked_for_all_saved_rankings': True,
            'inference_query_contains_only_qid_node_signature': True,
            'future_edge_prefix_invariance_checks': prefix_checks,
            'single_deletion_materialized_replays': single_replays,
            'production_policy_and_materialized_replays': inference_replays,
            'policy_replays_with_nonempty_actions': nonempty_actions,
            'timing_sample_policy_rows': len(timings),
            'timing_note': 'Already indexed formal queries and loaded models; pairwise-feature caches may be warm; model/index loading and offline intervention labels excluded.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-only', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    RUN.mkdir(parents=True, exist_ok=True)
    prior = audit_prior_files()
    data, hidden, checks = audit_data()
    payload = {'all_passed': True, 'old_files_unchanged': prior, 'data_checks': checks,
               'audit_scope': 'data_only' if args.data_only else 'full_experiment',
               'audit_script_sha256': sha(Path(__file__))}
    if not args.data_only:
        payload['experiment_checks'] = audit_experiment(data, hidden)
    payload['audit_seconds'] = time.perf_counter() - started
    destination = RUN / ('data_audit.json' if args.data_only else 'audit.json')
    write_json(destination, payload)
    (RUN / ('data_audit_testlog.txt' if args.data_only else 'audit_testlog.txt')).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
