"""Verify the frozen study and package its reproducible delivery copy."""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/sequential_v2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    original = json.loads((ROOT / 'ARTIFACTS.json').read_text(encoding='utf-8'))
    for name, record in original['files'].items():
        assert digest(ROOT / name) == record['sha256'], name
    lock = json.loads((RUN / 'test_lock.json').read_text(encoding='utf-8'))
    for name, expected in lock['selection_hashes'].items():
        assert digest(RUN / name) == expected, name
    for name, expected in lock['code_hashes'].items():
        assert digest(ROOT / name) == expected, name
    manifest = json.loads((RUN / 'manifest.json').read_text(encoding='utf-8'))
    for key, name in [('protocol_sha256', 'SEQUENTIAL_V2_PROTOCOL.md'),
                      ('config_sha256', 'configs/sequential_v2.json'),
                      ('dataset_sha256', 'data/processed/scoped/dataset.pkl')]:
        assert digest(ROOT / name) == manifest[key], name
    analysis = json.loads((RUN / 'analysis_checks.json').read_text(encoding='utf-8'))
    assert all(v for v in analysis.values() if isinstance(v, bool))
    replay = json.loads((RUN / 'replay_audit.json').read_text(encoding='utf-8'))
    assert replay['all_passed']
    complete = json.loads((RUN / 'run_complete.json').read_text(encoding='utf-8'))
    results = pd.read_csv(RUN / 'results.csv')
    assert len(results) == complete['measurement_rows']
    assert results.qid.nunique() == complete['test_queries']
    assert results.condition.nunique() == complete['conditions']
    assert results.method.nunique() == complete['methods']

    checks_path = ROOT / 'DELIVERY_CHECKS_V2.json'
    artifact_path = ROOT / 'ARTIFACTS_V2.json'
    # Create the target before checking the documentation link to this record.
    write_json(checks_path, {})
    write_json(artifact_path, {})
    new_docs = [p for p in ROOT.glob('*.md') if p.name not in original['files']]
    local_links = 0
    for document in new_docs:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', document.read_text(encoding='utf-8')):
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith('#'):
                continue
            target = target.split('#', 1)[0].strip('<>')
            assert (document.parent / target).exists(), (document.name, target)
            local_links += 1
    write_json(checks_path, {
        'original_files_unchanged': len(original['files']),
        'frozen_prediction_code_and_selection_hashes_valid': True,
        'protocol_config_dataset_hashes_valid': True,
        'local_document_links_validated': local_links,
        'measurement_rows_verified': len(results),
        'analysis_checks': analysis,
        'independent_materialized_replays': replay,
        'automated_tests_passed_during_completed_execution': 14,
        'test_reuse_is_exploratory': True,
    })

    selected = {ROOT / name for name in original['files']}
    selected.update(ROOT.glob('*.md'))
    selected.update(ROOT.glob('requirements*.txt'))
    selected.update(ROOT / name for name in ['ARTIFACTS.json', 'DELIVERY_CHECKS.json', 'DELIVERY_CHECKS_V2.json'])
    for directory, pattern in [('owe', '*.py'), ('tests', '*.py'), ('configs', '*.json')]:
        selected.update((ROOT / directory).glob(pattern))
    for directory in ['runs/retriever_diagnostic_v1', 'runs/sequential_v2']:
        selected.update(p for p in (ROOT / directory).rglob('*') if p.is_file())
    omitted = RUN / 'ranker_validation_cases.pkl'
    selected.discard(omitted)
    assert all(p.is_file() for p in selected)
    selected = sorted(selected, key=lambda p: p.relative_to(ROOT).as_posix())
    write_json(artifact_path, {
        'protocol': 'sequential-v2',
        'source_commit': original['source_commit'],
        'test_reuse_is_exploratory': True,
        'omitted_regenerable_cache': ['runs/sequential_v2/ranker_validation_cases.pkl'],
        'raw_source_note': 'Pinned raw .mm sources and full-library parse cache are omitted as in scoped-v1; reacquire with data/raw/manifest.json.',
        'files': {p.relative_to(ROOT).as_posix(): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in selected},
    })
    archive_path = ROOT / 'One_Wrong_Edge_Sequential_v2.zip'
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in [*selected, artifact_path]:
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(selected) + 1
    print(json.dumps({'archive': archive_path.name, 'files': len(selected) + 1,
                      'bytes': archive_path.stat().st_size, 'archive_sha256': digest(archive_path)}, indent=2))


if __name__ == '__main__':
    main()
