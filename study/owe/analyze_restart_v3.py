"""Analyze the fresh-topic restart without reusing earlier experiment outcomes.

Raw corruption seeds are first averaged within query. The primary global20
comparison weights the random and lexical-hard families equally. Local exposure
stress is always a separate conditional diagnostic.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from .experiment import write_json
from .report import table


ROOT = Path('runs/restart_v3')
DATA = Path('data/processed/restart_v3/dataset.pkl')
CONFIG = Path('configs/restart_v3.json')
KEYS = ['qid', 'group', 'topic', 'method', 'budget']
FAMILIES = {
    'random20_mean': ['random20_s401', 'random20_s409'],
    'hard20_mean': ['hard20_s419', 'hard20_s421'],
}
DISPLAY = [
    ('clean', '정상', 'Clean'),
    ('random20_mean', '무작위+20%, 두 시드', 'Random +20%'),
    ('hard20_mean', '어휘유사+20%, 두 시드', 'Lexical-hard +20%'),
    ('global20_mean', '전역20%: 두 오류 유형 평균', 'Global20 family mean'),
    ('missing20_s431', '관계20% 누락', 'Missing20%'),
    ('mixed20_s433', '삭제·추가 혼합', 'Mixed'),
    ('local8_s439', '탐색 출발점 오류: 별도 스트레스', 'Local exposure stress'),
]
_WEIGHTS = {}


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def interval(frame, column, repetitions=2000, seed=20261125):
    grouped = frame.groupby('group')[column].agg(['sum', 'count'])
    if grouped.empty:
        return np.nan, np.nan
    key = (tuple(map(str, grouped.index)), repetitions, seed)
    if key not in _WEIGHTS:
        _WEIGHTS[key] = np.random.default_rng(seed).multinomial(
            len(grouped), np.full(len(grouped), 1 / len(grouped)), size=repetitions)
    w = _WEIGHTS[key]
    values = (w @ grouped['sum'].to_numpy()) / (w @ grouped['count'].to_numpy())
    return tuple(map(float, np.quantile(values, [.025, .975])))


def mean_conditions(frame, names, label):
    sub = frame[frame.condition.isin(names)]
    assert set(sub.condition) == set(names), (label, set(sub.condition))
    assert sub.groupby(KEYS, dropna=False).size().eq(len(names)).all(), label
    numeric = list(sub.select_dtypes(include='number').columns.difference(KEYS))
    pooled = sub.groupby(KEYS, as_index=False, dropna=False)[numeric].mean()
    pooled['condition'] = label
    return pooled


def condition_frames(frame):
    frames = {name: sub.copy() for name, sub in frame.groupby('condition')}
    for label, names in FAMILIES.items():
        frames[label] = mean_conditions(frame, names, label)
    frames['global20_mean'] = mean_conditions(
        pd.concat([frames['random20_mean'], frames['hard20_mean']], ignore_index=True),
        ['random20_mean', 'hard20_mean'], 'global20_mean')
    return frames


def summarize(frame, config, topic=None):
    rows = []
    repetitions = config.get('bootstrap_repetitions', 2000)
    seed = config.get('bootstrap_seed', 20261125)
    for (method, budget), sub in frame.groupby(['method', 'budget']):
        lo, hi = interval(sub, 'delta_vs_none', repetitions, seed)
        deletions = sub.deletions.sum()
        row = {'condition': str(sub.condition.iloc[0]), 'method': method, 'budget': int(budget),
            'queries': int(sub.qid.nunique()), 'source_blocks': int(sub.group.nunique()),
            'recall': float(sub.recall.mean()), 'ndcg': float(sub.ndcg.mean()),
            'complete': float(sub.complete.mean()), 'delta_vs_none': float(sub.delta_vs_none.mean()),
            'ci_low': lo, 'ci_high': hi, 'improved_fraction': float(sub.improved_event.mean()),
            'harmed_fraction': float(sub.harmed_event.mean()), 'action_fraction': float(sub.acted_event.mean()),
            'mean_deletions': float(sub.deletions.mean()),
            'wrong_deletion_precision': float(sub.wrong_deletions.sum() / deletions) if deletions else np.nan,
            'wrong_deletions': float(sub.wrong_deletions.sum()), 'all_deletions': float(deletions),
            'returned': float(sub.returned.mean()), 'formal_tokens': float(sub.formal_tokens.mean()),
            'candidate_coverage': float(sub.candidate_coverage.mean()),
            'candidate_ceiling': float(sub.candidate_ceiling.mean()),
            'candidates': float(sub.candidates.mean()), 'transitions': float(sub.transitions.mean()),
            'exposed_false_edges': float(sub.exposed_false_edges.mean()),
            'injected_edges': float(sub.injected_edges.mean())}
        if topic is not None:
            row['topic'] = topic
        rows.append(row)
    return rows


def comparisons(frames, config):
    rows = []
    pairs = [('proof_ranker', 'old_local'), ('proof_ranker', 'lexical'),
             ('proof_ranker', 'expanded_text'), ('proof_ranker', 'text_ranker'),
             ('proof_ranker', 'proof_small64'), ('utility', 'proof_ranker'),
             ('wrongness', 'proof_ranker'), ('utility', 'matched_utility'),
             ('wrongness', 'matched_wrongness'), ('source_verified', 'proof_ranker'),
             ('oracle1', 'proof_ranker')]
    for condition, frame in frames.items():
        for budget in sorted(frame.budget.unique()):
            sub = frame[frame.budget == budget]
            for first, second in pairs:
                a = sub[sub.method == first][['qid', 'group', 'recall']]
                b = sub[sub.method == second][['qid', 'recall']]
                joined = a.merge(b, on='qid', suffixes=('_a', '_b'), validate='one_to_one')
                assert len(joined) == sub.qid.nunique(), (condition, first, second)
                joined['difference'] = joined.recall_a - joined.recall_b
                lo, hi = interval(joined, 'difference', config.get('bootstrap_repetitions', 2000),
                                  config.get('bootstrap_seed', 20261125))
                rows.append({'condition': condition, 'budget': int(budget),
                    'comparison': first + ' minus ' + second, 'queries': len(joined),
                    'source_blocks': joined.group.nunique(),
                    'difference': joined.difference.mean(), 'ci_low': lo, 'ci_high': hi})
    return pd.DataFrame(rows)


def corruption_damage(results, config):
    """Paired corruption effects with the retrieval model held fixed."""
    base = results[(results.method == 'proof_ranker') & (results.budget == 20)].copy()
    clean = base[base.condition == 'clean'][['qid', 'recall']].rename(columns={'recall': 'clean_recall'})
    base = base.merge(clean, on='qid', validate='many_to_one')
    base['corruption_delta'] = base.recall - base.clean_recall
    base['corruption_harmed_event'] = (base.corruption_delta < -1e-8).astype(float)
    base['corruption_helped_event'] = (base.corruption_delta > 1e-8).astype(float)
    base.to_csv(ROOT / 'corruption_damage_queries.csv', index=False)
    rows = []
    for condition, frame in condition_frames(base).items():
        lo, hi = interval(frame, 'corruption_delta', config.get('bootstrap_repetitions', 2000),
                          config.get('bootstrap_seed', 20261125))
        rows.append({'condition': condition, 'queries': frame.qid.nunique(), 'source_blocks': frame.group.nunique(),
            'clean_recall': frame.clean_recall.mean(), 'corrupted_recall': frame.recall.mean(),
            'difference': frame.corruption_delta.mean(), 'ci_low': lo, 'ci_high': hi,
            'corruption_harmed_fraction': frame.corruption_harmed_event.mean(),
            'corruption_helped_fraction': frame.corruption_helped_event.mean(),
            'mean_exposed_false_edges': frame.exposed_false_edges.mean(),
            'exposed_episode_fraction': frame.exposure_event.mean()})
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / 'corruption_damage.csv', index=False)
    return frame


def exposure_strata(results, config):
    """Condition on raw query-seed exposure events, retaining query/block units.

Numerators and exposure weights are averaged within query/family before taking
their ratio. This is not a union-over-seeds query subset.
    """
    rows = []
    for stratum, exposed in [('observed_false_edge', True), ('no_observed_false_edge', False)]:
        source = results[(results.budget == 20) & results.method.isin(['utility', 'wrongness'])].copy()
        source['stratum_weight'] = (source.exposure_event.astype(bool) == exposed).astype(float)
        for column in ['delta_vs_none', 'recall', 'acted_event', 'harmed_event', 'deletions', 'wrong_deletions']:
            source['stratum_' + column] = source[column] * source.stratum_weight
        frames = condition_frames(source)
        for condition in ['random20_mean', 'hard20_mean', 'global20_mean']:
            for method, sub in frames[condition].groupby('method'):
                weight = float(sub.stratum_weight.sum())
                row = {'condition': condition, 'method': method, 'stratum': stratum,
                    'all_queries': sub.qid.nunique(), 'all_source_blocks': sub.group.nunique(),
                    'queries_with_stratum_episodes': int((sub.stratum_weight > 0).sum()),
                    'episode_fraction': float(sub.stratum_weight.mean()),
                    'note': 'Raw seed exposure event first, within-query/family average second; conditional ratio over episode weights. No false traversal does not imply no contaminated prior features.'}
                for column in ['delta_vs_none', 'recall', 'acted_event', 'harmed_event', 'deletions', 'wrong_deletions']:
                    row[column] = float(sub['stratum_' + column].sum() / weight) if weight else np.nan
                if weight:
                    grouped = sub.groupby('group')[['stratum_delta_vs_none', 'stratum_weight']].sum()
                    key = (tuple(map(str, grouped.index)), config.get('bootstrap_repetitions', 2000), config.get('bootstrap_seed', 20261125))
                    if key not in _WEIGHTS:
                        _WEIGHTS[key] = np.random.default_rng(key[2]).multinomial(len(grouped), np.full(len(grouped), 1 / len(grouped)), size=key[1])
                    denominator = _WEIGHTS[key] @ grouped.stratum_weight.to_numpy()
                    numerator = _WEIGHTS[key] @ grouped.stratum_delta_vs_none.to_numpy()
                    valid = denominator > 0
                    row['ci_low'], row['ci_high'] = map(float, np.quantile(numerator[valid] / denominator[valid], [.025, .975]))
                    row['valid_bootstrap_resamples'] = int(valid.sum())
                else:
                    row.update(ci_low=np.nan, ci_high=np.nan, valid_bootstrap_resamples=0)
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / 'exposure_strata.csv', index=False)
    return frame


def availability(data):
    nodes = data['nodes']
    tests = [q for q in data['queries'] if q['split'] == 'test']
    old = pickle.loads(Path('data/processed/scoped/dataset.pkl').read_bytes())
    old_signatures = {n['signature'] for n in old['nodes']}
    records = []
    for q in tests:
        total = q.get('gold_total_count', q.get('total_gold_count', len(q['gold'])))
        indexed = q.get('indexed_gold_count', len([v for v in q['gold'] if isinstance(v, int) and 0 <= v < len(nodes)]))
        records.append({'qid': q['qid'], 'topic': q['topic'], 'group': q['group'],
            'total_gold': total, 'indexed_gold': indexed,
            'indexed_fraction': indexed / total, 'unindexed_gold': total - indexed,
            'old_index_signature_overlap': q['signature'] in old_signatures})
    frame = pd.DataFrame(records)
    frame.to_csv(ROOT / 'test_gold_availability.csv', index=False)
    result = {'test_queries': len(tests), 'test_blocks': frame.group.nunique(),
        'test_topics': frame.topic.value_counts().to_dict(),
        'old_index_signature_overlaps': int(frame.old_index_signature_overlap.sum()),
        'gold_count': int(frame.total_gold.sum()), 'indexed_gold_count': int(frame.indexed_gold.sum()),
        'mean_indexed_gold_fraction': float(frame.indexed_fraction.mean()),
        'questions_with_unindexed_gold': int((frame.unindexed_gold > 0).sum()),
        'questions_with_gold_over20': int((frame.total_gold > 20).sum()),
        'availability_note': 'Use manifest/index construction notes to distinguish fixed-index availability from a transductive proof-informed corpus.'}
    assert result['old_index_signature_overlaps'] == 0
    write_json(ROOT / 'test_novelty_and_availability.json', result)
    return result


def diagnose_clean(data):
    """Disjoint macro gold-mass accounting, with eligibility independent of the engine."""
    nodes = data['nodes']
    tests = {q['qid']: q for q in data['queries'] if q['split'] == 'test'}
    records = {}
    with gzip.open(ROOT / 'test_clean_rankings.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            records[record['qid']] = record
    vocab = pickle.loads((ROOT / 'vectorizer.pkl').read_bytes()).vocabulary_
    rows = []
    oov_rows = []
    novel_tokens = {}
    for qid, q in tests.items():
        record = records[qid]
        gold = set(q['gold'])
        indexed = {v for v in gold if 0 <= v < len(nodes)}
        eligible = {v for v in indexed if v < q['node'] and nodes[v]['signature'] != q['signature']}
        pool = set(record['candidates'])
        in_pool = gold & pool
        retrieved = gold & set(record['rankings']['proof_ranker'][:20])
        counts = {'retrieved': len(retrieved), 'out_of_index': len(gold - indexed),
                  'indexed_but_ineligible': len(indexed - eligible),
                  'eligible_but_outside_pool': len(eligible - pool),
                  'capacity20': max(0, len(in_pool) - 20),
                  'avoidable_within_pool': min(20, len(in_pool)) - len(retrieved)}
        assert set(retrieved) <= in_pool <= eligible
        assert sum(counts.values()) == len(gold), (qid, counts, len(gold))
        assert min(counts.values()) >= 0
        row = {'qid': qid, 'topic': q['topic'], 'group': q['group'], 'gold_count': len(gold),
               **{name + '_count': value for name, value in counts.items()},
               **{name + '_fraction': value / len(gold) for name, value in counts.items()}}
        rows.append(row)
        tokens = nodes[q['node']]['text'].split()
        missing = [token for token in tokens if token not in vocab]
        unique = set(tokens)
        oov_rows.append({'qid': qid, 'topic': q['topic'], 'group': q['group'],
            'tokens': len(tokens), 'oov_tokens': len(missing), 'oov_token_fraction': len(missing) / len(tokens),
            'unique_tokens': len(unique), 'unique_oov_tokens': len({t for t in unique if t not in vocab}),
            'unique_oov_fraction': len({t for t in unique if t not in vocab}) / len(unique)})
        novel_tokens.setdefault(q['topic'], Counter()).update(missing)
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / 'clean_failure_decomposition.csv', index=False)
    fractions = [c for c in frame if c.endswith('_fraction')]
    summary = {'queries': len(frame), 'source_blocks': frame.group.nunique(),
               'macro_gold_mass': {c.removesuffix('_fraction'): float(frame[c].mean()) for c in fractions},
               'note': 'Disjoint gold mass per query, then macro averaged; exact stored-proof citations, K20.'}
    write_json(ROOT / 'clean_failure_decomposition_summary.json', summary)
    frame.groupby('topic')[fractions].mean().to_csv(ROOT / 'clean_failure_decomposition_by_topic.csv')
    oov = pd.DataFrame(oov_rows)
    oov.to_csv(ROOT / 'query_oov.csv', index=False)
    by_topic = oov.groupby('topic').agg(queries=('qid', 'count'),
        mean_oov_token_fraction=('oov_token_fraction', 'mean'),
        mean_unique_oov_fraction=('unique_oov_fraction', 'mean'),
        all_tokens=('tokens', 'sum'), all_oov_tokens=('oov_tokens', 'sum')).reset_index()
    by_topic['pooled_oov_token_fraction'] = by_topic.all_oov_tokens / by_topic.all_tokens
    by_topic.to_csv(ROOT / 'oov_topic_summary.csv', index=False)
    write_json(ROOT / 'oov_tokens_by_topic.json', {topic: counts.most_common(20) for topic, counts in novel_tokens.items()})
    assert np.allclose(frame[fractions].sum(axis=1), 1)
    return summary


def effect_signs(config):
    rows = []
    edge_rows = []
    budget_index = config.get('output_budgets', [10, 20, 40]).index(20)
    for path in sorted(ROOT.glob('test_*_cases.pkl')):
        condition = path.stem.removeprefix('test_').removesuffix('_cases')
        cases = pickle.loads(path.read_bytes())
        if isinstance(cases, dict):
            cases = list(cases.values())
        for case in cases:
            values = np.asarray(case.get('benefits', []))
            if not len(values):
                continue
            gains = values[:, budget_index] if values.ndim == 2 else values
            supported = case.get('source_supported', [None] * len(gains))
            predictions = case.get('wrongness_scores', [np.nan] * len(gains))
            for (u, v), gain, truth, prediction in zip(case.get('edges', []), gains, supported, predictions):
                edge_rows.append({'condition': condition, 'qid': case['qid'], 'source': u, 'target': v,
                    'source_supported': truth, 'wrongness_score': float(prediction), 'benefit20': float(gain)})
            rows.append({'condition': condition, 'qid': case['qid'],
                'group': case.get('group', ''), 'candidate_edges': len(gains),
                'positive_edges': int((gains > 1e-8).sum()),
                'negative_edges': int((gains < -1e-8).sum()),
                'zero_edges': int((np.abs(gains) <= 1e-8).sum()),
                'best_single_delta': max(0., float(np.max(gains))),
                'worst_single_delta': min(0., float(np.min(gains)))})
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / 'query_effect_signs.csv', index=False)
    if not frame.empty:
        totals = frame.groupby('condition')[['candidate_edges', 'positive_edges', 'negative_edges', 'zero_edges']].sum()
        totals['positive_fraction'] = totals.positive_edges / totals.candidate_edges
        totals['negative_fraction'] = totals.negative_edges / totals.candidate_edges
        totals['zero_fraction'] = totals.zero_edges / totals.candidate_edges
        totals.to_csv(ROOT / 'effect_sign_summary.csv')
    edges = pd.DataFrame(edge_rows)
    if not edges.empty:
        edges.to_csv(ROOT / 'edge_effects.csv.gz', index=False)
        classifier_rows = []
        for condition, sub in edges.dropna(subset=['source_supported', 'wrongness_score']).groupby('condition'):
            unique = sub.groupby(['source', 'target'], as_index=False).agg(
                source_supported=('source_supported', 'first'), score=('wrongness_score', 'mean'))
            wrong = (~unique.source_supported.astype(bool)).astype(int)
            classifier_rows.append({'condition': condition, 'unique_edges': len(unique),
                'wrong_edges': int(wrong.sum()),
                'ROC_AUC': roc_auc_score(wrong, unique.score) if wrong.nunique() == 2 else np.nan,
                'average_precision': average_precision_score(wrong, unique.score) if wrong.nunique() == 2 else np.nan,
                'note': 'Unique-edge mean query score; descriptive, without independent-edge significance claims'})
        pd.DataFrame(classifier_rows).to_csv(ROOT / 'wrongness_classification.csv', index=False)
    return frame


def examples(results, data):
    qmap = {q['qid']: q for q in data['queries']}
    nodes = data['nodes']
    base = results[(results.budget == 20) & (results.condition == 'clean')]
    joined = base[base.method == 'proof_ranker'][['qid', 'group', 'topic', 'recall']].merge(
        base[base.method == 'old_local'][['qid', 'recall']], on='qid', suffixes=('_new', '_old'))
    joined['difference'] = joined.recall_new - joined.recall_old
    joined.to_csv(ROOT / 'base_ranker_query_changes.csv', index=False)
    counts = {'queries': len(joined), 'improved': int((joined.difference > 1e-8).sum()),
              'harmed': int((joined.difference < -1e-8).sum()),
              'unchanged': int((joined.difference.abs() <= 1e-8).sum())}
    write_json(ROOT / 'base_ranker_change_counts.json', counts)
    ranked = pd.concat([joined.nlargest(3, 'difference').assign(kind='base_improvement'),
                        joined.nsmallest(3, 'difference').assign(kind='base_regression')]).drop_duplicates('qid')
    interventions = results[(results.budget == 20) & results.method.isin(['utility', 'wrongness'])]
    harmed = interventions[interventions.delta_vs_none < -1e-8].nsmallest(5, 'delta_vs_none').drop_duplicates('qid')
    helped = interventions[interventions.delta_vs_none > 1e-8].nlargest(3, 'delta_vs_none').drop_duplicates('qid')
    picked = pd.concat([harmed.assign(kind='policy_harm'), helped.assign(kind='policy_gain')])
    wanted = {(r.qid, 'clean') for r in ranked.itertuples()}
    wanted.update((r.qid, r.condition) for r in picked.itertuples())
    records = {}
    for path in sorted(ROOT.glob('*rankings*.jsonl.gz')):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                key = (r['qid'], r['condition'])
                if key in wanted:
                    records[key] = r
    def label(v):
        return nodes[int(v)]['label'] if isinstance(v, (int, np.integer)) and 0 <= int(v) < len(nodes) else str(v)
    reference = set(map(tuple, data['edges']))
    out = []
    for row in ranked.itertuples():
        q = qmap[row.qid]
        record = records.get((row.qid, 'clean'))
        if record is None:
            continue
        gold = set(q['gold'])
        old = set(record['rankings']['old_local'][:20])
        new = set(record['rankings']['proof_ranker'][:20])
        out.append({'kind': row.kind, 'qid': row.qid, 'topic': row.topic, 'condition': 'clean',
            'source': nodes[q['node']]['source'], 'gold_count': len(gold),
            'old_recall20': row.recall_old, 'new_recall20': row.recall_new,
            'delta_recall20': row.difference,
            'lost_gold': [label(v) for v in sorted((old - new) & gold)],
            'gained_gold': [label(v) for v in sorted((new - old) & gold)]})
    for row in picked.itertuples():
        q = qmap[row.qid]
        record = records.get((row.qid, row.condition))
        if record is None:
            continue
        gold = set(q['gold'])
        before = set(record['rankings']['proof_ranker'][:20])
        after = set(record['rankings'][row.method][:20])
        out.append({'kind': row.kind, 'qid': row.qid, 'topic': q['topic'], 'condition': row.condition,
            'policy': row.method, 'source': nodes[q['node']]['source'], 'gold_count': len(gold),
            'delta_recall20': row.delta_vs_none,
            'actions': [{'source': label(u), 'target': label(v), 'source_supported': (u, v) in reference}
                        for u, v in record.get('actions', {}).get(row.method, [])],
            'lost_gold': [label(v) for v in sorted((before - after) & gold)],
            'gained_gold': [label(v) for v in sorted((after - before) & gold)]})
    write_json(ROOT / 'failure_and_worked_examples.json', out)
    return out


def figure(summary, topic_summary):
    directory = ROOT / 'figures'
    directory.mkdir(exist_ok=True)
    conditions = ['clean', 'global20_mean', 'missing20_s431', 'mixed20_s433', 'local8_s439']
    labels = ['Clean', 'Global20\nfamily mean', 'Missing20%', 'Mixed', 'Local stress\n(separate)']
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    for method, label in [('old_local', 'Original local'), ('lexical', 'Lexical'),
                          ('text_ranker', 'Learned static features'), ('proof_ranker', 'Learned proof-use')]:
        s = summary[(summary.method == method) & (summary.budget == 20)].set_index('condition').loc[conditions]
        axes[0].plot(range(len(conditions)), 100 * s.recall.to_numpy(), marker='o', label=label)
    axes[0].set_xticks(range(len(conditions)), labels)
    axes[0].set_ylabel('Stored-proof citation Recall@20 (%)')
    axes[0].set_title('Fresh-topic retrieval')
    axes[0].legend(fontsize=8, loc='center', bbox_to_anchor=(.5, .63))
    for i, (method, label) in enumerate([('utility', 'Utility ML'), ('wrongness', 'Attribution ML'),
                                       ('source_verified', 'Source verification'), ('oracle1', 'Limited oracle')]):
        s = summary[(summary.method == method) & (summary.budget == 20)].set_index('condition').loc[conditions]
        x = np.arange(len(conditions)) + (i - 1.5) * .15
        color = plt.get_cmap('tab10')(i)
        axes[1].vlines(x, s.ci_low.to_numpy() * 100, s.ci_high.to_numpy() * 100, color=color)
        axes[1].scatter(x, s.delta_vs_none.to_numpy() * 100, color=color, s=23, label=label)
    axes[1].axhline(0, color='black', linewidth=.7)
    axes[1].set_xticks(range(len(conditions)), labels)
    axes[1].set_ylabel('Change vs proof-use ranker (percentage points)')
    axes[1].set_title('Additional removal value; 95% block intervals')
    axes[1].legend(fontsize=8, loc='lower left')
    fig.suptitle('Restart-v3: unseen statement signatures in related mathematical topics', fontsize=12)
    fig.tight_layout()
    for extension in ['png', 'svg']:
        fig.savefig(directory / f'restart_stages.{extension}', dpi=220)
    plt.close(fig)
    topics = sorted(topic_summary.topic.unique())
    fig, ax = plt.subplots(figsize=(9.8, 4.7))
    for i, (method, label) in enumerate([('lexical', 'Lexical'), ('text_ranker', 'Learned static features'),
                                       ('proof_ranker', 'Learned proof-use'), ('utility', '+ Utility ML')]):
        s = topic_summary[(topic_summary.condition == 'global20_mean') &
                          (topic_summary.method == method)].set_index('topic').loc[topics]
        ax.bar(np.arange(len(topics)) + (i - 1.5) * .18, s.recall.to_numpy() * 100, width=.18, label=label)
    ax.set_xticks(np.arange(len(topics)), [t.replace('_', '\n') for t in topics])
    ax.set_ylabel('Stored-proof citation Recall@20 (%)')
    ax.set_title('Topic transfer under global20 corruption (equal family mean)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    for extension in ['png', 'svg']:
        fig.savefig(directory / f'topic_transfer.{extension}', dpi=220)
    plt.close(fig)


def render_reports(data, config, summary, paired, topics, novelty, cases):
    stats = data['stats']
    selections = {i: read_json(ROOT / f'stage{i}_selection.json', {}) for i in [1, 2, 3]}
    completed = read_json(ROOT / 'run_complete.json', {})
    audit = read_json(ROOT / 'replay_audit.json', read_json(ROOT / 'audit.json', {}))
    experiment_audit = audit.get('experiment_checks', {})
    timings = pd.read_csv(ROOT / 'inference_timings.csv') if (ROOT / 'inference_timings.csv').exists() else pd.DataFrame()
    timing_ko = '별도 추론 시간 표본이 아직 기록되지 않았다.'
    timing_en = 'A separate inference timing sample is not yet recorded.'
    if not timings.empty:
        unique_times = timings.drop_duplicates(['qid', 'condition'])
        base_ms = unique_times.ranker_seconds.median() * 1000
        utility_ms = timings[timings.policy == 'utility'].policy_seconds.median() * 1000
        wrong_ms = timings[timings.policy == 'wrongness'].policy_seconds.median() * 1000
        timing_ko = (f'별도 추론 표본 {len(unique_times)}개 질문·조건에서 기본 순위 검색 중앙값은 {base_ms:.2f}ms, '
                     f'효용 정책 포함은 {utility_ms:.2f}ms, 오류 분류 정책 포함은 {wrong_ms:.2f}ms였다. '
                     '두 정책 행에 반복된 기본 시간은 질문·조건별 한 번만 집계했다. 색인·모델은 이미 적재했고 형식 질문을 사용했으며 명제쌍 특징 캐시는 예열되었을 수 있다. '
                     '모델/색인 로딩과 사후 효과 정답 생성은 제외하므로 자연어 서비스 지연으로 해석하지 않는다.')
        timing_en = (f'In {len(unique_times)} distinct query-condition timing samples, median baseline inference is {base_ms:.2f}ms, '
                     f'utility-policy inference {utility_ms:.2f}ms and attribution-policy inference {wrong_ms:.2f}ms. '
                     'Baseline timings repeated across two policy rows are deduplicated. Models/indexes are loaded and query inputs are formal; pairwise-feature caches may be warm. '
                     'Loading and all offline effect labels are excluded, so these are not natural-language service latency estimates.')
    train = read_json(ROOT / 'utility_training_summary.json', {})
    def get(condition, method, budget=20):
        return summary[(summary.condition == condition) & (summary.method == method) & (summary.budget == budget)].iloc[0]
    def pct(value):
        return f'{value * 100:.2f}'
    def change(row):
        return f'{row.delta_vs_none * 100:+.3f} [{row.ci_low * 100:+.3f}, {row.ci_high * 100:+.3f}]'
    def comp(condition, a, b, budget=20):
        return paired[(paired.condition == condition) & (paired.budget == budget) &
                      (paired.comparison == a + ' minus ' + b)].iloc[0]
    def pair_change(row):
        return f'{row.difference * 100:+.3f} [{row.ci_low * 100:+.3f}, {row.ci_high * 100:+.3f}]'
    clean = get('clean', 'proof_ranker')
    primary = get('global20_mean', 'utility')
    wrong = get('global20_mean', 'wrongness')
    if primary.ci_low > 0:
        verdict = '사전 지정한 전역 오류 평균에서 효용 ML의 양의 추가 효과를 관측했다.'
        verdict_en = 'The prespecified global-corruption mean showed a positive additional utility-policy effect.'
    elif primary.ci_high < 0:
        verdict = '사전 지정한 전역 오류 평균에서 효용 ML이 검색을 악화시켰다.'
        verdict_en = 'The prespecified global-corruption mean showed degradation from the utility policy.'
    else:
        verdict = '사전 지정한 전역 오류 평균에서 효용 ML의 추가 개선을 확정할 수 없었다.'
        verdict_en = 'The prespecified global-corruption mean did not establish additional utility-policy improvement.'
    methods = ['old_local', 'lexical', 'text_ranker', 'proof_ranker', 'utility', 'wrongness']
    result_table = table(['조건', '기존 지역', '어휘', '텍스트·정적 특징', '증명 인용 학습', '+효용 ML', '+오류 분류 ML'],
        [[ko] + [pct(get(c, m).recall) for m in methods] for c, ko, _ in DISPLAY])
    result_en = table(['Condition', 'Old local', 'Lexical', 'Static-feature ranker', 'Proof-use ranker', '+ Utility', '+ Attribution'],
        [[en] + [pct(get(c, m).recall) for m in methods] for c, _, en in DISPLAY])
    incremental = table(['조건', '효용 ML 변화(pp)', '오류 분류 변화(pp)', '출처 확인 변화(pp)', '제한된 단일 삭제 오라클(pp)'],
        [[ko] + [change(get(c, m)) for m in ['utility', 'wrongness', 'source_verified', 'oracle1']] for c, ko, _ in DISPLAY])
    incremental_en = table(['Condition', 'Utility delta (pp)', 'Attribution delta (pp)', 'Source-verification delta (pp)', 'Limited single-removal oracle (pp)'],
        [[en] + [change(get(c, m)) for m in ['utility', 'wrongness', 'source_verified', 'oracle1']] for c, _, en in DISPLAY])
    component_table = table(['조건', '증명 인용 − 어휘(pp)', '주 순위 − 같은 후보 정적 특징 순위(pp)', '증명 인용256 − 동일 모델64(pp)'],
        [[ko] + [pair_change(comp(c, 'proof_ranker', m)) for m in ['lexical', 'text_ranker', 'proof_small64']] for c, ko, _ in DISPLAY])
    coverage_table = table(['조건', '후보 포함률(%)', 'Top20 후보 상한(%)', '평균 후보 수', '평균 전이 수', '관측 거짓 간선'],
        [[ko, pct(get(c, 'proof_ranker').candidate_coverage), pct(get(c, 'proof_ranker').candidate_ceiling),
          f'{get(c, "proof_ranker").candidates:.1f}', f'{get(c, "proof_ranker").transitions:.1f}',
          f'{get(c, "proof_ranker").exposed_false_edges:.2f}'] for c, ko, _ in DISPLAY])
    clean_candidate_table = table(['정상 조건 방법', '평균 후보 수', '정답 후보 포함률(%)', 'Top20 후보 상한(%)', 'Recall@20(%)'],
        [[m, f'{get("clean", m).candidates:.1f}', pct(get('clean', m).candidate_coverage),
          pct(get('clean', m).candidate_ceiling), pct(get('clean', m).recall)]
         for m in ['old_local', 'lexical', 'expanded_text', 'proof_small64', 'proof_ranker']])
    deletion_rows = []
    for c, ko, _ in [DISPLAY[0], DISPLAY[3], DISPLAY[-1]]:
        for m in ['utility', 'wrongness', 'matched_utility', 'matched_wrongness']:
            r = get(c, m)
            precision = '—' if pd.isna(r.wrong_deletion_precision) else pct(r.wrong_deletion_precision) + '%'
            deletion_rows.append([ko, m, f'{r.mean_deletions:.3f}', pct(r.action_fraction) + '%',
                precision, pct(r.harmed_fraction) + '%'])
    deletion_table = table(['조건', '정책', '평균 삭제 수', '실행 비율', '삭제 중 거짓 귀속', '회수율 악화 비율'], deletion_rows)
    topic_rows = []
    for topic in sorted(topics.topic.unique()):
        sub = topics[(topics.topic == topic) & (topics.condition == 'global20_mean')]
        base = sub[sub.method == 'proof_ranker'].iloc[0]
        utility = sub[sub.method == 'utility'].iloc[0]
        lexical = sub[sub.method == 'lexical'].iloc[0]
        topic_rows.append([topic, int(base.queries), int(base.source_blocks), pct(lexical.recall),
                           pct(base.recall), change(utility)])
    topic_table = table(['주제', '질문', '원문 블록', '어휘 Recall', '증명 인용 Recall', '효용 ML 추가(pp)'], topic_rows)
    topic_en = table(['Topic', 'Questions', 'Blocks', 'Lexical Recall', 'Proof-use Recall', 'Utility delta (pp)'], topic_rows)
    budget_table = table(['K', '증명 인용 Recall', '+효용 ML Recall', '추가 변화(pp)', '전체 인용 회수율'],
        [[k, pct(get('global20_mean', 'proof_ranker', k).recall), pct(get('global20_mean', 'utility', k).recall),
          change(get('global20_mean', 'utility', k)), pct(get('global20_mean', 'proof_ranker', k).complete)] for k in [10, 20, 40]])
    worked_table = table(['유형', '질문', '주제', '조건', '회수율 변화(pp)', '놓친 실제 인용'],
        [[r['kind'], r['qid'], r['topic'], r['condition'], f'{r["delta_recall20"] * 100:+.2f}',
          ', '.join(r['lost_gold']) or '—'] for r in cases])
    counts = read_json(ROOT / 'base_ranker_change_counts.json', {})
    effect = pd.read_csv(ROOT / 'effect_sign_summary.csv') if (ROOT / 'effect_sign_summary.csv').exists() else pd.DataFrame()
    effect_text = ''
    if not effect.empty:
        effect_text = table(['조건', '평가 간선', '양의 효과', '음의 효과', '0효과 비율'],
            [[r.condition, int(r.candidate_edges), int(r.positive_edges), int(r.negative_edges),
              pct(r.zero_fraction) + '%'] for r in effect.itertuples()])
    purity_text = '삭제를 선택하지 않아 정밀도를 계산할 수 없다.' if pd.isna(wrong.wrong_deletion_precision) else (
        f'전역 오류 평균에서 오류 분류 정책이 삭제한 관계 중 {pct(wrong.wrong_deletion_precision)}%가 거짓 귀속이었다. '
        f'이 정책의 실제 회수율 변화는 {change(wrong)}pp였다.')
    index_note = stats.get('index_construction_note', stats.get('corpus_assembly_note',
        '사전 선택 핵심 주제의 문서와 공개 증명에서만 유도한 단일 단계 경계로 구성했다. 시험 증명의 의존 목록으로 경계를 확대하지 않고 색인 밖 정답을 분모에 보존했다.'
        if stats.get('boundary_derived_only_from_public_proofs') else
        '색인 구성과 정답 접근 범위는 데이터 manifest 및 프로토콜의 기록을 따른다.'))
    index_note_en = ('The index contains preselected core-topic documents and one-hop boundaries derived only from public proofs. Heldout proof dependencies do not expand the index; unindexed gold stays in recall denominators.'
        if stats.get('boundary_derived_only_from_public_proofs') else
        'Index assembly and gold access are specified in the data manifest and protocol.')
    main_choice = str(selections[2].get('model', 'see selection record'))
    chosen_note = (f'검증이 선택한 주 모델은 `{main_choice}`으로 텍스트·정적 메타데이터를 쓰고 관측 그래프 특징을 제외한 모델이다. 이 경우 표의 주 검색기와 텍스트 학습 비교가 같으며 그래프 특징의 추가 우월성을 주장하지 않는다.'
        if main_choice.startswith('text_') else
        f'검증이 선택한 주 모델은 `{main_choice}`이다. 텍스트·정적 메타데이터를 쓰고 관측 그래프 특징을 제외한 별도 모델과 같은 후보에서 비교했다.')
    chosen_note_en = (f'Validation selected the static-feature model `{main_choice}` as the primary proof-use predictor. Consequently the primary and static-feature controls coincide; no incremental graph-feature advantage is claimed.'
        if main_choice.startswith('text_') else
        f'Validation selected `{main_choice}` as the primary proof-use predictor, with a separately fitted graph-feature-excluded/static-feature model evaluated on the same candidates.')
    classifier = pd.read_csv(ROOT / 'wrongness_classification.csv') if (ROOT / 'wrongness_classification.csv').exists() else pd.DataFrame()
    classifier_table = ''
    if not classifier.empty:
        classifier_table = table(['조건', '서로 다른 간선', '거짓 귀속', 'ROC-AUC', 'Average precision'],
            [[r.condition, int(r.unique_edges), int(r.wrong_edges),
              '—' if pd.isna(r.ROC_AUC) else f'{r.ROC_AUC:.3f}',
              '—' if pd.isna(r.average_precision) else f'{r.average_precision:.3f}'] for r in classifier.itertuples()])
    decomposition = read_json(ROOT / 'clean_failure_decomposition_summary.json', {})
    decomposition_table = table(['정상 조건 정답 인용의 위치', '질문별 평균 비중(%)'],
        [[name, pct(value)] for name, value in decomposition.get('macro_gold_mass', {}).items()])
    oov = pd.read_csv(ROOT / 'oov_topic_summary.csv') if (ROOT / 'oov_topic_summary.csv').exists() else pd.DataFrame()
    oov_table = table(['주제', '질문', '평균 OOV 토큰 비율(%)', '평균 OOV 고유 토큰 비율(%)'],
        [[r.topic, int(r.queries), pct(r.mean_oov_token_fraction), pct(r.mean_unique_oov_fraction)] for r in oov.itertuples()])
    damage = pd.read_csv(ROOT / 'corruption_damage.csv')
    global_damage = damage[damage.condition == 'global20_mean'].iloc[0]
    damage_table = table(['조건', '오류 조건 − 동일 질문 정상(pp)', '오류로 회수율 악화 비율', '거짓 간선 탐색 노출 비율'],
        [[ko, pair_change(damage[damage.condition == c].iloc[0]),
          pct(damage[damage.condition == c].iloc[0].corruption_harmed_fraction) + '%',
          pct(damage[damage.condition == c].iloc[0].exposed_episode_fraction) + '%'] for c, ko, _ in DISPLAY if c != 'clean'])
    damage_table_en = table(['Condition', 'Corrupted minus clean (pp)', 'Corruption harmed (%)', 'False-edge traversal exposure (%)'],
        [[en, pair_change(damage[damage.condition == c].iloc[0]),
          pct(damage[damage.condition == c].iloc[0].corruption_harmed_fraction),
          pct(damage[damage.condition == c].iloc[0].exposed_episode_fraction)] for c, _, en in DISPLAY if c != 'clean'])
    exposure = pd.read_csv(ROOT / 'exposure_strata.csv')
    exposure_table = table(['탐색 노출 층(전역20%)', '정책', '해당 질문×시드 비중', '추가 효과(pp)', '악화 비율'],
        [[r.stratum, r.method, pct(r.episode_fraction) + '%', change(r), pct(r.harmed_event) + '%']
         for r in exposure[exposure.condition == 'global20_mean'].itertuples()])
    clean_static_gain = comp('clean', 'proof_ranker', 'text_ranker')
    matched_utility = comp('global20_mean', 'utility', 'matched_utility')
    matched_wrongness = comp('global20_mean', 'wrongness', 'matched_wrongness')
    missing_utility = get('missing20_s431', 'utility')
    local_classifier = classifier[classifier.condition == 'local8_s439'].iloc[0] if not classifier.empty else None
    policy_selection = json.dumps(selections[3], ensure_ascii=False, indent=2)
    policy_rows = [[name, p.get('model'), p.get('budget'), p.get('threshold'),
                    f'{p.get("validation_gain", 0) * 100:+.4f}']
                   for name, p in selections[3].get('policies', {}).items()]
    policy_table = table(['정책', '모델', '선택한 최대 삭제 수', '임계값', '검증 변화(pp)'], policy_rows)
    policy_table_en = table(['Policy', 'Model', 'Selected max deletions', 'Threshold', 'Validation delta (pp)'], policy_rows)
    headline = (f'새 주제 시험 {novelty["test_queries"]}개에서 정상 조건의 증명 인용 Recall@20은 '
                f'**{pct(clean.recall)}%**였다. {verdict} 주 비교의 추가 효과는 **{change(primary)}pp**였다.')
    korean = f'''# 처음부터 재실행한 수학 근거 검색 실험

2026-11-25 · `restart-v3` · 완료한 실제 실행에서 생성.

## 확인한 결과

{headline}

이전 실행의 모델·오류 그래프·효과 정답을 재사용하지 않고 원문 확인, 데이터 분리, 색인, 순위 모델, 개입 정답, 정책을 새로 만들었다. 개발 자료는 이전에 사용한 범위를 다시 분할했다. 시험은 기존 색인에 canonical signature가 전혀 없었던 관련 주제의 정리로 제한했다. 같은 Metamath 원문은 과거에 전역 파싱·확인한 적이 있으므로 새로운 원문 출처에 대한 검증이라고 주장하지 않는다. 정리 서명의 신규성은 수학적 의미나 유사 정리 전체의 독립성을 보장하지 않는다.

평가 정답은 **한 저장 증명의 직접 인용**이다. 이를 모든 가능한 증명의 논리적 필수 관계로 부르지 않는다. 자연어 답변 정확도와 새로운 증명 생성 성공률은 이번 측정 범위에 포함하지 않았다.

## 자료와 분리

주 실험 그래프는 집합·관계·함수·유한집합과 그래프 경로·오일러 정리의 관련 범위에 단어/문자열, 관계의 폐포, 동치관계, 순서, 관계·함수 후속 단원을 더한 범위다. 전체 Metamath 검증은 출처 확인 절차이고 전체 라이브러리를 주 검색 실험에 사용한 것이 아니다.

- 핵심 정리 {stats.get('core_nodes', '기록 참조'):,}개, 경계 문서 {stats.get('boundary_nodes', '기록 참조'):,}개, 전체 색인 {stats.get('total_nodes', '기록 참조'):,}개.
- 검색 가능한 원문 인용 관계 {stats.get('searchable_reference_edges', 0):,}개. 새 주제 전체의 증명 출발 관계를 가렸고, 질문·동일 서명·미래 정리를 후보에서 제외했다.
- 질문 분리: `{json.dumps(stats.get('query_splits', {}), ensure_ascii=False)}`. 순위 학습·정책 학습·검증·시험의 원문 블록은 분리했다.
- 신규 시험 {novelty['test_queries']}개, 원문 블록 {novelty['test_blocks']}개. 이전 색인 서명과의 겹침은 {novelty['old_index_signature_overlaps']}개였다.
- 시험 직접 인용 {novelty['gold_count']:,}개 중 색인 안의 인용은 {novelty['indexed_gold_count']:,}개다. 질문별 평균 색인 포함률은 {pct(novelty['mean_indexed_gold_fraction'])}%이며, 색인 밖 정답이 있는 질문은 {novelty['questions_with_unindexed_gold']}개다. [정답 가용성 표](runs/restart_v3/test_gold_availability.csv)를 남겼다.

색인 구성 기록: {index_note}

## 검색과 학습 순서

후보 확보는 어휘 기본 후보·확장 가능한 출발점·과거 관측 증명의 인용 정보를 조합한다. 전역 어휘와 기존 고정 지역 탐색을 함께 실행했다. 기존 지역 비교는 예전 저장 점수를 가져온 것이 아니라 **새 공통 색인·어휘 점수·오류 그래프에서4시드/2단계 원리를 다시 실행**한 결과다. 새 TF-IDF는 모든 비교에 같은 public-fit 설정과 sublinear TF를 사용했다. 따라서 이전 v2의38.02%와 새 점수를 직접 비교해 변화 원인을 주장하지 않는다.

순위 학습은 실제 학습 증명의 직접 인용을 예측하며, 텍스트·정적 메타데이터를 사용하는 순위 비교와 관측 그래프 지지까지 쓰는 비교를 구분했다. 관측 그래프 특징10개를 제외한 모델도 원문 순서 거리, 공리/정의/경계 역할, 가정·서로 다른 변수 제약 수, 형식 기호 수는 유지한다. 이 비교도 그래프가 만든 같은 후보를 사용한다. **관측 그래프 특징이 없는 재순위**이며 순수한 수식 유사도나 그래프를 전혀 쓰지 않는 전체 검색기는 아니다. 동일 후보의 확장 후 어휘 순위와 작은64개 후보 비교도 저장했다.

순위 모델을 고정한 뒤 별도의 `policy_train` 블록에서 실제 간선을 억제하고 후보·특징·순위를 다시 계산해 개입 효과를 새로 만들었다. 정책 학습 질문은 순위 모델의 정답 학습에 사용하지 않았다. 이 분리로 이전 OOF 모델과 최종 배포 모델 사이의 차이를 피했다. 검증에서 후보 설정·순위·간선 정책을 순서대로 선택하고 시험 전에 고정했다.

{chosen_note}

1단계 선택: `{json.dumps(selections[1], ensure_ascii=False)}`

2단계 선택: `{json.dumps(selections[2], ensure_ascii=False)}`

추가 간선 학습은 부호 있는 실제 회수율 변화를 예측하는 **효용 모델**과 **거짓 증명 인용 귀속 분류기**로 나눴다. 두 종류는 다른 목표이며 올바른 인용도 질문별 효용 때문에 억제될 수 있다. 주입 간선을 거짓 수학 명제라고 표현하지 않는다.

검증에서는 최대1/3/8개 후보 정책을 실제로 재검색해 비교했다. 시험과 배포 재실행 검사는 아래에서 선택한 정책에 대한 결과다. 선택되지 않은 다중 삭제 정책의 배포 성능을 검증했다고 주장하지 않는다.

{policy_table}

## 전체 성능과 후보 한계

아래 값은 질문별 직접 인용 Recall@20의 퍼센트다. 오류 시드 반복은 각 질문 안에서 평균했고, 전역20%는 무작위와 어휘유사 두 유형의 평균을 동일 가중했다. 정상, 누락, 혼합, 탐색 출발점 스트레스는 주 전역20% 평균에 합치지 않았다.

{result_table}

후보 집합과 순위의 효과를 별도로 확인했다. 아래 차이의 단위는 퍼센트포인트다.

{component_table}

{coverage_table}

같은256개 한도의 전역 어휘 후보와 비교하면 정상 후보 포함률은 {pct(get('clean', 'lexical').candidate_coverage)}%에서 {pct(clean.candidate_coverage)}%로 높아졌다. 각 직접 검색 비교의 실제 후보 수를 함께 제시한다.

{clean_candidate_table}

이 표의 후보·전이·포함률은 삭제 없는 기본 검색의 값이다. 원시 결과에서 전역 어휘·기존 지역·작은64개 직접 비교와 출처 확인은 각자의 실제 후보 수와 전이를 기록하며, 추가 학습·무작위·오라클 정책의 항목은 개입 전 기본 검색 값이다. 정책별 추가 재검색까지 합친 총량이 아니다. 작은64개 조건은 같은 모델에 후보 한도와 어휘/과거 인용 앵커를 줄인 사전 정의 조합을 적용한 이동 검사이며, 한도만 바꾼 인과 효과나64개에서 재학습한 최적 모델과의 비교가 아니다.

정상 조건의 놓친 근거를 서로 겹치지 않게 분해했다. `out_of_index`는 색인 밖, `indexed_but_ineligible`은 원문 순서·서명 제한, `eligible_but_outside_pool`은 후보 미포함, `capacity20`은 후보 안 정답 수가20을 넘는 부분, `avoidable_within_pool`은 같은 후보의 완벽한 순위가 추가로 찾을 수 있는 부분이다. 각 질문 안에서 모두 더하면 전체 정답 질량이 된다.

{decomposition_table}

## 오류 자체가 검색을 얼마나 바꾸었는가

동일 질문·고정 검색기에서 오류 조건과 정상 조건을 직접 짝지었다. 음의 값은 악화이고 양의 값은 개선이다. 오류가 언제나 해롭다고 가정하지 않는다. 악화/노출 사건은 개별 오류 시드에서 먼저 계산한 뒤 질문 안에서 평균했다.

{damage_table}

정상 조건에서 기존 지역 검색보다 좋아진 질문은 {counts.get('improved', 0)}개, 나빠진 질문은 {counts.get('harmed', 0)}개, 같은 질문은 {counts.get('unchanged', 0)}개였다. 같은 출력 상한이 같은 후보 구성·시간·형식 토큰 비용을 뜻하지 않는다. 실제 반환 수와 형식 공백 토큰 수를 원시 표에 남겼다.

## 새 검색기 위 간선 제거의 추가 효과

{incremental}

같은 수의 무작위 삭제 대비 전역20%의 추가 차이는 효용 ML {pair_change(matched_utility)}pp, 오류 분류 ML {pair_change(matched_wrongness)}pp였다. 두 차이 모두0을 포함하는 구간으로, 이번 선택 정책이 같은 행동 수의 무작위 삭제보다 개선한다고 확정할 수 없다. 누락 조건에서는 효용 정책 변화가 {change(missing_utility)}pp로 악화 방향이었다. 이 수치도 여러 보조 비교 중 하나라는 제한을 유지한다.

95% 구간은 시드를 먼저 질문 안에서 평균한 뒤 원문 블록을 bootstrap 2,000회 재표집한 기술적 구간이다. 고정된 선택 모델에 조건부이며 학습 변동 전체나 다중 비교를 보정한 확정적 유의성 검정이 아니다. 특히 주제별 작은 블록 집합의 구간을 넓은 수학 영역의 보장으로 해석하지 않는다.

개수 일치 무작위 삭제는 각 질문의 행동 수와 허용 간선 후보를 맞췄다. 출처 확인은 공개 증명 인용표로 잘못된 귀속을 확인하는 더 강한 입력 비교다. 단일 삭제 오라클은 제한된 후보 안에서 사후 효과를 보는 진단이며 달성 가능한 예측기 성능이 아니다.

{deletion_table}

전역 오류를 실제 탐색에서 만났는지도 구분했다. 아래 층은 정답이나 사후 효과가 아니라 **기본 탐색에서 거짓 귀속을 하나라도 지났는가**로 정했다. 사건을 개별 시드에서 먼저 계산하고, 질문/유형별 사건 가중치와 효과 분자를 평균한 후 비율을 구했다. 따라서 시드 중 한 번이라도 노출되었는지로 만든 합집합 질문 비율이 아니다. 구간도 같은 질문·원문 블록 단위를 유지한다. 노출 없는 재표집은 분모0이므로 제외하며 유효 재표집 수를 CSV에 남겼다.

{exposure_table}

탐색에서 거짓 간선이 없더라도 전역 인용 prior가 오류의 영향을 받을 수 있다. 이 층 분해는 관측적 기전 진단이며 노출 자체의 인과 효과를 식별한 결과가 아니다.

{purity_text}

분류 점수 자체의 기술적 진단도 남겼다. 한 간선이 여러 질문에 등장하면 질문별 점수를 간선 안에서 평균한 뒤 서로 다른 간선을 대상으로 계산한다. 독립 간선 모집단에 대한 유의성 검정이 아니다. 거짓 귀속이 없는 조건은 AUC/AP를 계산하지 않는다.

{classifier_table}

단일 간선 개입의 실제 효과 분포도 남겼다. 효과 대부분이 0이면 판별력이 부족한 것과 검색 구조에 개선 여지가 작은 것을 구분해야 한다.

{effect_text}

## 주제 전이와 실패 사례

아래는 전역20% 두 오류 유형 평균이다.

{topic_table}

공개 자료에만 적합한 어휘집에서 새 질문의 공백 구분 단일 토큰이 빠진 정도도 진단했다. 아래 비율은 각 질문 안에서 계산한 뒤 평균했다. 구조·변수 토큰을 포함하며 OOV만으로 검색 실패의 인과 원인을 확정하지 않는다. 원시 값과 실제 빠진 토큰은 별도 CSV/JSON에 있다.

{oov_table}

{worked_table}

실제 억제한 출발/목적 정리 이름, 원문 링크, 잃거나 새로 찾은 정답 인용은 [실패·근거 사례](runs/restart_v3/failure_and_worked_examples.json)에 있다. 개선 질문만 선택해 평균 결과를 설명하지 않는다.

## 예산과 오류 해석

K=20에서 선택한 모델·정책을 재학습하지 않고 K=10/40에도 적용했다.

{budget_table}

전역 무작위/어휘유사 오류의 새 시드는 같은 오류 모형 안의 일반화다. 관계 누락·혼합은 별도의 오류 이동 조건이다. 탐색 출발점에 오류를 추가하는 조건은 정답을 보지 않고 관측 탐색과 수식 유사도로 결정한 **결정론적 조건부 스트레스**이며 현실의 평균 오류 분포가 아니다. `local8_s439`의439는 조건 이름이며 주입에 쓰는 난수 시드가 아니다. 하나의 스트레스 규칙을 여러 독립 시드 반복처럼 세지 않는다. 주입 불가능한 질문도 보존하고 주입/노출량을 보고했다. 이 조건에서만 얻는 개선을 일반적인 간선 제거 효과로 확대하지 않는다.

실행 시간 중 `offline_seconds`는 질문·조건별 개입 효과 정답 생성의 실제 측정 시간이며 서비스 추론 지연이 아니다. 같은 값이 방법·K 행에 반복되므로 모든 행을 더해 비용으로 계산하지 않는다. 총 wall time과 CPU 병렬 작업의 합계 시간도 구분한다. 후보·전이 카운터가 기본 검색 값이면 추가 정책 재검색 비용 전체로 읽지 않는다. 비용 항목과 독립 재실행 확인의 구체적 범위는 실행·감사 기록을 따른다.

{timing_ko}

## 독립 재계산과 재실행 확인

감사에서 기존 파일 {audit.get('old_files_unchanged', 0)}개의 해시와 시험 전에 고정한 입력/예측 파일 {experiment_audit.get('all_frozen_file_hashes_valid', 0)}개를 확인했다. 공개 문서 {experiment_audit.get('fresh_idf_fit_reconstructed_without_heldout_nodes', 0):,}개만으로 어휘집·IDF 적합을 재구성했고, 저장된 측정 {experiment_audit.get('all_metric_rows_independently_recalculated', 0):,}행을 독립적으로 다시 계산했다. 미래 간선 접두 불변성 {experiment_audit.get('future_edge_prefix_invariance_checks', 0)}회, 실제 그래프에서의 단일 삭제 {experiment_audit.get('single_deletion_materialized_replays', 0)}회, 정답을 전달하지 않은 선택 정책/물리적 그래프 재실행 {experiment_audit.get('production_policy_and_materialized_replays', 0)}회가 통과했다. 정책 재실행 중 실제 삭제가 있었던 것은 {experiment_audit.get('policy_replays_with_nonempty_actions', 0)}회이며 나머지는 보류였다. 이를 선택되지 않은 다중 정책의 재실행 수로 해석하지 않는다.

원문 파싱 때 연구 구현 체커로 {stats.get('proofs_checked_by_local_checker', 0):,}개 저장 증명의 치환·필수 가정·변수 제약·최종 스택을 확인했다. 이 로컬 체커의 통과를 독립 인증된 공식 검증기의 인증으로 표현하지 않는다. [감사 원문](runs/restart_v3/audit.json)과 [분석 일관성 확인](runs/restart_v3/analysis_checks.json)을 보존했다.

## 선택·재현 기록과 남은 한계

새 주제에서 확인한 가장 큰 남은 문제는 단일 관계 오류만이 아니다. 정상 정답 질량 중 {pct(decomposition.get('macro_gold_mass', {}).get('eligible_but_outside_pool', 0))}%는 검색 가능한데 후보에 들어오지 않았고, {pct(decomposition.get('macro_gold_mass', {}).get('avoidable_within_pool', 0))}%는 같은 후보 안의 순위가 놓친 부분이었다. 그래프 특징을 더한 순위 모델은 정상 조건에서 같은 후보의 정적 특징 모델보다 {pair_change(clean_static_gain)}pp 높았다. 그러나 고정 모델의 작은64개 조합과256개 조합의 차이는 {pair_change(comp('clean', 'proof_ranker', 'proof_small64'))}pp여서 큰 후보 집합의 가치도 자동으로 전제하지 않는다.

전역 오류 자체의 평균 변화는 {pair_change(global_damage)}pp로 작고 불확실했다. 그럼에도 질문×오류 시드의 {pct(global_damage.exposed_episode_fraction)}%가 실제 거짓 간선을 탐색했다. 따라서 추가 ML의 작은 효과를 단순히 오류가 검색 경로에 나타나지 않은 탓으로 설명하기 어렵다. 단일 개입 효과의 희소성, 제한된 오라클의 작은 개선 여지, 새 주제에서의 분류·효용 전이를 함께 봐야 한다. 결정론적 local 스트레스에서 분류 AUC는 {local_classifier.ROC_AUC:.3f}였다. 전역 오류 학습이 국소 오류 판별까지 일반화한다는 보장도 얻지 못했다.

3단계 선택 원문:

```json
{policy_selection}
```

완료 기록: `{json.dumps(completed, ensure_ascii=False)}`

감사 기록: `{json.dumps(audit, ensure_ascii=False)}`

정책 학습 효과 기록: `{json.dumps(train, ensure_ascii=False)}`

[프로토콜](RESTART_V3_PROTOCOL.md), [재현 안내](README_V3.md), [원시 요약](runs/restart_v3/summary.csv), [짝지은 차이](runs/restart_v3/comparisons.csv), [주제별 표](runs/restart_v3/topic_summary.csv)를 함께 제공한다. 단일 형식 출처, 기존 개발 범위 재사용, 정리의 의미적 유사성, 저장 증명 인용 정답의 불완전성, 인위적 오류 모형이라는 한계를 유지한다. 미실행된 자연어 답변·논리적 가정 누락·새 증명 검증의 성능은 보고하지 않는다.

형식 Metamath의 배경 증명에는 정확한 인용 출처가 있다. 운영 시 이 원문 인용표에 완전히 접근할 수 있다면, 거짓 증명 사용 귀속은 학습 없이 직접 대조해 확인할 수 있다. 이번 ML은 오염된 그래프와 관측 특징만 받으며 출처 확인 비교는 정상 인용표에 접근한다. 따라서 이 통제 실험만으로 Metamath 원래 메타데이터보다 ML이 실용적으로 필요하다고 주장하지 않는다. 출처가 없거나 확인 비용이 높은 상황, 또는 관계의 진위와 다른 질문별 효용 판단에서 ML의 가치가 생길 수 있지만 그 운영 조건의 실용성은 이번 실험으로 확립하지 않았다. 비형식 수학 문헌에서 실제 추출된 지식그래프 오류를 검증한 결과로도 확대하지 않는다.

![새 주제 검색과 추가 간선 개입](runs/restart_v3/figures/restart_stages.png)

![주제별 전이 결과](runs/restart_v3/figures/topic_transfer.png)
'''
    Path('RESULTS_V3.md').write_text(korean, encoding='utf-8')
    english = f'''# One Wrong Edge on Fresh Mathematical Topics

## Rebuilding premise retrieval and measuring the additional value of edge removal

25 November 2026 · Empirical working draft 0.4 · restart-v3

## Abstract

We rebuild a scoped Metamath retrieval experiment from its pinned raw source and evaluate related topics whose canonical statement signatures were absent from all previously indexed documents. The new test comprises {novelty['test_queries']} questions in {novelty['test_blocks']} source blocks. Earlier subject material is repartitioned into disjoint ranker-training, policy-training and validation blocks. Candidate generation and proof-use ranking are frozen before a separate policy-training set supplies newly rerun signed removal effects. Clean stored-proof citation Recall@20 is {clean.recall:.4f}. On the prespecified equal-family mean of random and lexical-hard global corruption, additional utility learning changes Recall@20 by {primary.delta_vs_none*100:+.3f} percentage points, with a 95% source-block bootstrap interval [{primary.ci_low*100:+.3f}, {primary.ci_high*100:+.3f}]pp. {verdict_en} We distinguish source-attribution accuracy from retrieval utility and keep traversal-local stress separate from global corruption. These are new-topic results within one previously parsed formal library, not a new-source or natural-language reasoning benchmark.

## 1. Motivation and contribution

The previous development evaluation found both candidate exclusion and lexical ranking loss. It also showed that accurate detection of false proof-use attributions need not improve retrieval. This restart tests those findings on additional related topics, rather than treating a new random split of previously inspected questions as independent confirmation. Learned premise ranking, graph expansion and query-conditioned decisions are established ideas. The contribution sought here is reproducible sequential evidence about candidate availability, actual stored-proof citation ranking, removal headroom and the gap between attribution correctness and query utility. No superiority over named external retrieval or theorem-proving systems is claimed.

## 2. Source, scope and heldout semantics

The pinned source is [Metamath set.mm](https://github.com/metamath/set.mm/tree/{stats.get('source_commit', '')}), with the original source licence and checksum retained. The full raw parse/check is source quality assurance; the retrieval graph is restricted to a related subject allowlist and terminal supporting documents. The scope contains {stats.get('core_nodes', 0):,} core assertions, {stats.get('boundary_nodes', 0):,} boundary documents and {stats.get('searchable_reference_edges', 0):,} searchable public proof-use edges. It covers elementary sets, relations, functions and finite sets; graph walks/paths/Eulerian paths; and additional words, relation closures, equivalence relations, orderings and relation/function material.

Gold consists of actual direct `|-` citations in one checked proof. It is not the set of premises necessary in every proof, and alternative sufficient proofs are not exhaustively annotated. Documents preserve essential hypotheses and distinct-variable constraints. The test excludes any canonical signature previously present even as an indexed boundary document; this is stricter than excluding old questions alone. Nevertheless, canonical novelty does not establish semantic independence from related lemmas, and the raw library was previously parsed for source verification. Old development material is reused with a new block assignment. This is topic-transfer evaluation within one source.

All fresh-topic proof-source edges are withheld, not only edges from selected test questions. The retriever rejects the query, equivalent signatures and later-source candidates. Historical citation features use only observable source references preceding the query. The test has {novelty['gold_count']:,} annotated direct citations, of which {novelty['indexed_gold_count']:,} are indexed; mean per-question indexed availability is {pct(novelty['mean_indexed_gold_fraction'])}%. {index_note_en}

## 3. Sequential procedure and controls

First, candidate generation combines lexical anchors, expandable lexical seeds and public historical-citation priors. Validation chooses the candidate variant. Second, supervised ranking predicts actual training-proof citations; a static-feature learned ablation excludes ten observed-graph feature columns but retains source-order distance, axiom/definition/boundary roles, hypothesis/distinct-variable counts and formal-text features. This reranker uses the same graph-generated candidates, so it is neither a pure-formula-only comparator nor a graph-free end-to-end retriever. Expanded-pool lexical ranking, historical local retrieval, global lexical retrieval and a smaller64-candidate condition serve as controls. The old-local algorithm is rerun using the new common index, publicly fitted IDF, sublinear TF and corruption graphs; previous saved scores are not reused. Earlier v2 scores on different questions/indexing are not direct causal comparators. Query/premise identity embeddings, gold, corruption flags, heldout proof references and clean/observed graph differences are excluded from prediction features.

{chosen_note_en}

Third, the selected ranker is frozen. A separate policy-training partition, whose questions supplied no ranker labels, supplies signed removal effects by rerunning candidates, graph features and ranking after intervention. This avoids both reusing the prior experiment's effects and shifting from an out-of-fold training ranker to a different final ranker. Utility regression and false-attribution classification are distinguished. Thresholds, action budgets and abstention are selected on validation before test evaluation.

Matched-count random deletion uses the same allowable action pool. Source verification has stronger provenance access and is not a lightweight action-budget-matched learner. The single-deletion oracle observes actual effects inside a limited candidate action set; it diagnoses headroom and is not attainable predictor performance. Validation evaluates budgets1/3/8 by actually rerunning multi-edge outcomes rather than adding single-edge effects. Test evaluation and deployment replays concern the selected policies below; they do not validate deployment of unselected multi-edge policies.

{policy_table_en}

## 4. Corruption and statistical evaluation

Random and lexical-hard global additions, missing references, a mixed removal/addition condition and a separate traversal-local stress are fixed in advance. False additions mean unsupported proof-use attribution, not false mathematical propositions. Source order and admissible relation endpoints are preserved; actual injection and exposure counts are recorded. Training, validation and test corruption seeds are distinct. New seeds test within-mechanism generalization, while untrained missing/mixed mechanisms provide error-distribution shifts.

The primary incremental comparison is utility policy versus no deletion on the frozen proof-use retriever at K=20. Repeated random seeds and repeated hard seeds are each averaged within query; the two family means then receive equal weight. Clean, missing, mixed and local stress are reported separately. Local stress chooses injection sites deterministically from observable baseline traversal and formula similarity without gold or realized retrieval losses, and includes questions where injection is unavailable. The439 suffix is a condition label, not an RNG seed or independent random repetition. It is a conditional mechanism diagnostic rather than a representative population error model.

We report macro citation Recall, nDCG, full citation recovery, candidate availability/ceiling, K=10/40, action/harm/purity rates, topic transfer and costs. Paired 95% intervals resample source blocks 2,000 times after within-query seed averaging. Intervals are descriptive and conditional on the fixed selected models. They neither cover all training uncertainty nor provide multiplicity-adjusted confirmatory guarantees. Small topic-specific block counts further limit interpretation.

## 5. Results

Values below are macro direct-citation Recall@20 percentages.

{result_en}

Corruption itself is also paired against the same question's clean result under the frozen no-deletion retriever. Negative changes indicate harm and positive changes indicate improvement; damage is measured rather than assumed. Event rates are formed per raw seed before averaging.

{damage_table_en}

Additional changes relative to the same frozen proof-use retriever are in percentage points.

{incremental_en}

The utility policy's primary change is {change(primary)}pp. The attribution classifier's primary change is {change(wrong)}pp. Accuracy among deleted attributions and signed retrieval benefit are different outcomes, so a high deletion purity alone is not evidence of better retrieval. Observable exposure strata use each raw query-seed traversal's presence/absence of any false edge, then average effect numerators and exposure weights within query/family before taking their ratio. Thus strata do not use a union-over-seeds question subset or gold-dependent selection, and source blocks remain the resampling units. A no-false-traversal episode can still have contaminated global citation priors, so this stratification does not identify a causal exposure effect. On clean questions, the proof-use ranker improves {counts.get('improved', 0)} questions, harms {counts.get('harmed', 0)} and ties {counts.get('unchanged', 0)} versus the original local retriever. Actual lost/gained citations and suppressed edges are preserved in the worked-example artifact, including adverse cases.

Against matched-count random deletion, the utility policy changes global20 recall by {pair_change(matched_utility)}pp and the attribution policy by {pair_change(matched_wrongness)}pp; neither interval establishes an advantage. The utility policy's missing-edge change is {change(missing_utility)}pp, illustrating adverse distribution transfer among the descriptive secondary comparisons.

Topic results below use the equal-family global20 mean.

{topic_en}

## 6. Mechanism, costs and limitations

The candidate ceiling separates unavailable evidence from within-pool ranking loss. The limited removal oracle and signed effect distribution separate predictor failure from a near-zero controllable effect. A result concentrated in local exposure stress cannot establish average utility under global corruption. Source verification can remove many false attributions using exact public proof provenance, whereas the learned policy operates with observable features and bounded actions. Those input and action differences constrain cost-effectiveness claims.

Clean gold mass is partitioned into retrieved citations, unindexed citations, indexed but ineligible citations, eligible citations outside the candidate pool, unavoidable K20 capacity loss within the pool, and avoidable within-pool ranking loss. These components are disjoint and sum to one per query. This distinguishes coverage limits from ranking limits without attributing every failure to graph quality. Per-topic query unigram OOV fractions against the publicly fitted vocabulary are descriptive diagnostics, not causal proof that OOV causes retrieval failure.

In particular, {pct(decomposition.get('macro_gold_mass', {}).get('eligible_but_outside_pool', 0))}% of clean gold mass is eligible but outside the pool and {pct(decomposition.get('macro_gold_mass', {}).get('avoidable_within_pool', 0))}% is avoidable within-pool ranking loss. Clean proof-use ranking exceeds static-feature ranking by {pair_change(clean_static_gain)}pp. Increasing the frozen-model recipe from64 to256 candidates changes clean recall by {pair_change(comp('clean', 'proof_ranker', 'proof_small64'))}pp, so larger candidate sets are not automatically justified. Global corruption changes no-deletion recall by {pair_change(global_damage)}pp even though {pct(global_damage.exposed_episode_fraction)}% of query-seed episodes traverse at least one false edge. The weak incremental policy result therefore cannot be explained simply by absent error exposure. The local-stress attribution AUC is {local_classifier.ROC_AUC:.3f}, which also cautions against transferring the global error classifier to that deterministic local mechanism.

Raw `offline_seconds` measures counterfactual effect-label construction per query/context, not production inference latency. It is repeated across method and K rows and must not be summed across those duplicates. Direct lexical, old-local, smaller64 and source-verification controls record their own candidate/transition quantities; learned, matched-random and oracle intervention rows retain the base no-deletion pool counters and do not include all subsequent policy reruns. The smaller64 recipe changes lexical/prior anchors as well as the candidate cap and is applied to a frozen model; it is neither a pure cap-only causal comparison nor an optimized64-candidate model. Formal whitespace-token counts are not model tokenizer counts, and equal K does not imply equal computation or evidence-token cost.

{timing_en}

This study concerns citation retrieval in one source, rather than generated-answer correctness or new-proof success. The old development range has been reused; fresh test canonical signatures do not rule out mathematical resemblance. Artificial source-attribution errors omit many real mathematical failures, such as missing assumptions or incorrect theorem statements. Future confirmation needs independent questions/sources and reasoning-level checks, with model retraining uncertainty and stronger semantic duplicate controls.

Formal Metamath background proofs supply exact proof-use provenance. When that complete provenance is operationally available, direct source checking can identify false attributions without learning. Our ML receives the corrupted graph and observable features, whereas source verification has stronger clean-provenance access. This controlled corruption benchmark therefore does not establish a practical need for ML over native Metamath metadata. Learning may have value when provenance is missing or expensive to check, or when optimizing question-conditioned utility rather than relation truth; those operational conditions are not validated here. We do not claim to have evaluated real extraction errors in nonformal mathematical knowledge graphs.

## 7. Reproducibility

The separate restart directory preserves raw-source/data manifests, all new split records, stage selection hashes, freshly fitted models, new policy effects, query-condition-method measurements and actual ranked/action outputs. See [protocol](RESTART_V3_PROTOCOL.md), [reproduction guide](README_V3.md), [summary](runs/restart_v3/summary.csv), [paired comparisons](runs/restart_v3/comparisons.csv), [topic results](runs/restart_v3/topic_summary.csv) and [worked adverse cases](runs/restart_v3/failure_and_worked_examples.json). The completion and audit records define exactly which computations and replays passed; no missing reasoning evaluation is reported as completed.

The independent audit verifies {audit.get('old_files_unchanged', 0)} preserved prior files, {experiment_audit.get('all_frozen_file_hashes_valid', 0)} frozen prediction/input hashes, the publicly fitted IDF reconstructed from {experiment_audit.get('fresh_idf_fit_reconstructed_without_heldout_nodes', 0):,} documents, and {experiment_audit.get('all_metric_rows_independently_recalculated', 0):,} recomputed metric rows. It also checks {experiment_audit.get('future_edge_prefix_invariance_checks', 0)} future-edge prefix invariances, {experiment_audit.get('single_deletion_materialized_replays', 0)} materialized single removals and {experiment_audit.get('production_policy_and_materialized_replays', 0)} selected-policy/materialized-graph replays. Of the policy replays, {experiment_audit.get('policy_replays_with_nonempty_actions', 0)} have nonempty actions. The raw parse locally checks {stats.get('proofs_checked_by_local_checker', 0):,} proofs with the research implementation; this is not certification by an independently verified upstream checker.

![Fresh-topic stages and incremental removal effects](runs/restart_v3/figures/restart_stages.png)

![Topic-transfer results](runs/restart_v3/figures/topic_transfer.png)
'''
    Path('One_Wrong_Edge_Paper_Draft_Restart_v3.md').write_text(english, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports-only', action='store_true', help='Refresh prose from previously computed tables')
    args = parser.parse_args()
    if not (ROOT / 'run_complete.json').exists():
        raise RuntimeError('Complete the frozen restart-v3 test before analysis')
    config = read_json(CONFIG, {})
    data = pickle.loads(DATA.read_bytes())
    if args.reports_only:
        summary = pd.read_csv(ROOT / 'summary.csv')
        paired = pd.read_csv(ROOT / 'comparisons.csv')
        topics = pd.read_csv(ROOT / 'topic_summary.csv')
        novelty = read_json(ROOT / 'test_novelty_and_availability.json')
        cases = read_json(ROOT / 'failure_and_worked_examples.json', [])
        render_reports(data, config, summary, paired, topics, novelty, cases)
        print('Refreshed RESULTS_V3.md and English restart-v3 draft', flush=True)
        return
    results = pd.read_csv(ROOT / 'results.csv')
    assert not results.duplicated(['qid', 'condition', 'method', 'budget']).any()
    results['acted_event'] = (results.deletions > 0).astype(float)
    results['improved_event'] = (results.delta_vs_none > 1e-8).astype(float)
    results['harmed_event'] = (results.delta_vs_none < -1e-8).astype(float)
    results['exposure_event'] = (results.exposed_false_edges > 0).astype(float)
    frames = condition_frames(results)
    summary = pd.DataFrame([r for frame in frames.values() for r in summarize(frame, config)])
    summary.to_csv(ROOT / 'summary.csv', index=False)
    topics = pd.DataFrame([r for frame in frames.values()
        for topic, sub in frame[frame.budget == 20].groupby('topic') for r in summarize(sub, config, topic)])
    topics.to_csv(ROOT / 'topic_summary.csv', index=False)
    paired = comparisons(frames, config)
    paired.to_csv(ROOT / 'comparisons.csv', index=False)
    corruption_damage(results, config)
    exposure_strata(results, config)
    novelty = availability(data)
    decomposition = diagnose_clean(data)
    effect_signs(config)
    cases = examples(results, data)
    figure(summary, topics)
    qa = {'unique_query_condition_method_budget': True,
          'all_output_caps_respected': bool((results.returned <= results.budget).all()),
          'finite_metrics': bool(np.isfinite(results[['recall', 'ndcg', 'complete', 'delta_vs_none']]).all().all()),
          'metrics_in_unit_interval': bool(((results[['recall', 'ndcg', 'complete']] >= -1e-9) &
                                           (results[['recall', 'ndcg', 'complete']] <= 1 + 1e-9)).all().all()),
          'action_rate_bounded_by_mean_deletions': bool((summary.action_fraction <= summary.mean_deletions + 1e-12).all()),
          'wrong_deletions_bounded_by_all_deletions': bool((results.wrong_deletions <= results.deletions).all()),
          'fresh_test_signature_overlap_zero': novelty['old_index_signature_overlaps'] == 0,
          'clean_gold_mass_accounting_sums_to_one': bool(abs(sum(decomposition['macro_gold_mass'].values()) - 1) < 1e-10),
          'single_oracle_nonnegative_at20': bool((results[(results.method == 'oracle1') &
                                                         (results.budget == 20)].delta_vs_none >= -1e-8).all())}
    base = results[results.method == 'proof_ranker'][['qid', 'condition', 'budget', 'recall']]
    joined = results.merge(base, on=['qid', 'condition', 'budget'], suffixes=('', '_base'), validate='many_to_one')
    qa['stored_deltas_match_paired_recall'] = bool(np.allclose(joined.delta_vs_none, joined.recall - joined.recall_base, atol=1e-8))
    for policy in ['utility', 'wrongness']:
        first = results[results.method == policy][['qid', 'condition', 'budget', 'deletions']]
        second = results[results.method == 'matched_' + policy][['qid', 'condition', 'budget', 'deletions']]
        matched = first.merge(second, on=['qid', 'condition', 'budget'], suffixes=('_policy', '_random'), validate='one_to_one')
        qa[policy + '_matched_action_counts_equal'] = bool(np.array_equal(matched.deletions_policy, matched.deletions_random))
    qa['source_verification_restores_clean_for_additions'] = True
    for condition in ['random20_s401', 'random20_s409', 'hard20_s419', 'hard20_s421', 'local8_s439']:
        clean = results[(results.condition == 'clean') & (results.method == 'proof_ranker')].sort_values(['qid', 'budget']).recall.to_numpy()
        verified = results[(results.condition == condition) & (results.method == 'source_verified')].sort_values(['qid', 'budget']).recall.to_numpy()
        qa['source_verification_restores_clean_for_additions'] &= bool(np.array_equal(clean, verified))
    # Report action events before within-query seed averaging; never use a union across seeds.
    assert all(qa.values()), qa
    write_json(ROOT / 'analysis_checks.json', qa)
    render_reports(data, config, summary, paired, topics, novelty, cases)
    print(summary[(summary.budget == 20) & summary.condition.isin(['clean', 'global20_mean', 'local8_s439']) &
                  summary.method.isin(['old_local', 'lexical', 'text_ranker', 'proof_ranker', 'utility', 'wrongness', 'oracle1'])]
          [['condition', 'method', 'recall', 'delta_vs_none', 'ci_low', 'ci_high', 'mean_deletions']].to_string(index=False), flush=True)
    print('Wrote fresh-topic summaries, paired intervals, adverse cases, figures and both reports', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
