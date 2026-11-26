"""Validate and package the completed restart study without changing prior runs.

Invoke only after the final experiment audit, analysis and delivery prose are
complete. The archive carries its pinned raw source and the reparsed proof
frames, so the complete v3 audit and from-raw rebuild need no source download.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote
import zipfile

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
RUN = BASE / 'runs/restart_v3'
DATA = BASE / 'data/processed/restart_v3'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def workspace_path(name):
    path = (BASE / name).resolve()
    assert path.is_relative_to(BASE), name
    return path


def local_document_links(documents, planned_paths):
    checked = []
    for document in documents:
        text = document.read_text(encoding='utf-8')
        for target in re.findall(r'!?\[[^\]]*\]\((<[^>]+>|[^)]+)\)', text):
            target = target.strip()
            if target.startswith('<'):
                target = target[1:target.index('>')]
            elif ' "' in target:
                target = target.split(' "', 1)[0]
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith('#'):
                continue
            target = unquote(target.split('#', 1)[0])
            resolved = (document.parent / target).resolve()
            assert resolved.is_relative_to(BASE), (document.name, target)
            assert resolved.is_file() or resolved in planned_paths, (document.name, target)
            checked.append((document.relative_to(BASE).as_posix(), target))
    return checked


def main():
    prior_path = BASE / 'ARTIFACTS_V2.json'
    prior = read_json(prior_path)
    assert len(prior['files']) == 186, 'Prior delivery manifest unexpectedly changed in shape'
    for name, record in prior['files'].items():
        path = workspace_path(name)
        assert sha(path) == record['sha256'], f'Prior artifact changed: {name}'
        assert path.stat().st_size == record['bytes'], name

    lock = read_json(RUN / 'test_lock.json')
    assert lock['selected_before_test'] and lock['fresh_signature_test']
    for name, expected in lock['files'].items():
        assert sha(workspace_path(name)) == expected, name
    audit = read_json(RUN / 'audit.json')
    assert audit['all_passed'] and audit['audit_scope'] == 'full_experiment'
    assert audit['old_files_unchanged'] == len(prior['files'])
    assert sha(BASE / 'owe/audit_restart_v3.py') == audit['audit_script_sha256']
    analysis = read_json(RUN / 'analysis_checks.json')
    assert analysis and all(isinstance(value, bool) and value for value in analysis.values())
    data_manifest = read_json(DATA / 'manifest.json')
    for name, expected in data_manifest['inputs'].items():
        assert sha(workspace_path(name)) == expected, name
    for name, expected in data_manifest['outputs'].items():
        assert sha(DATA / name) == expected, name
    assert data_manifest['complete_raw_verification_passed']
    assert data_manifest['all_previous_index_signatures_excluded_from_test']
    assert data_manifest['boundary_derived_only_from_public_unhidden_source_proofs']

    complete = read_json(RUN / 'run_complete.json')
    results = pd.read_csv(RUN / 'results.csv')
    assert len(results) == complete['measurement_rows']
    assert results.qid.nunique() == complete['test_queries']
    assert results.condition.nunique() == complete['conditions']
    assert results.method.nunique() == complete['methods']
    assert not results.duplicated(['qid', 'condition', 'method', 'budget']).any()
    assert audit['experiment_checks']['all_metric_rows_independently_recalculated'] == len(results)
    assert complete['test_previously_indexed_signature_count'] == 0
    assert complete['all_selections_locked_before_test']

    artifact_path = BASE / 'ARTIFACTS_V3.json'
    checks_path = BASE / 'DELIVERY_CHECKS_V3.json'
    archive_path = BASE / 'One_Wrong_Edge_Restart_v3.zip'
    new_docs = [p for p in BASE.glob('*.md') if p.name not in prior['files']]
    links = local_document_links(new_docs, {artifact_path, checks_path, archive_path})
    write_json(checks_path, {
        'protocol': 'restart-v3',
        'prior_delivery_files_unchanged': len(prior['files']),
        'prior_manifest_included': prior_path.name,
        'prior_manifest_sha256': sha(prior_path),
        'frozen_model_code_config_and_selection_hashes_valid': len(lock['files']),
        'complete_data_input_and_output_hashes_valid': True,
        'full_independent_audit': audit,
        'analysis_checks': analysis,
        'measurement_rows_verified': len(results),
        'local_document_links_validated': len(links),
        'raw_set_mm_and_reparsed_frames_included': True,
        'old_archives_excluded': True,
        'archive_validation': 'The packager tests ZIP CRCs and its exact entry set before printing the final receipt.',
    })

    selected = {workspace_path(name) for name in prior['files']}
    selected.add(prior_path)
    selected.add(checks_path)
    selected.update(BASE.glob('*.md'))
    selected.update(BASE.glob('requirements*.txt'))
    for directory, pattern in [('owe', '*.py'), ('tests', '*.py'), ('configs', '*.json')]:
        selected.update((BASE / directory).glob(pattern))
    for directory in [DATA, RUN]:
        selected.update(path for path in directory.rglob('*') if path.is_file())
    selected.update(BASE / 'data/raw' / name for name in ['set.mm', 'manifest.json', 'LICENSE'])
    selected = {path.resolve() for path in selected
                if path.suffix.lower() != '.zip' and '__pycache__' not in path.parts}
    selected.discard(artifact_path)
    assert all(path.is_file() and path.is_relative_to(BASE) for path in selected)
    assert {workspace_path(name) for name in prior['files']}.issubset(selected)
    assert (DATA / 'parsed.pkl') in selected
    assert (BASE / 'data/raw/set.mm') in selected
    selected = sorted(selected, key=lambda path: path.relative_to(BASE).as_posix())
    entries = {path.relative_to(BASE).as_posix(): {'bytes': path.stat().st_size, 'sha256': sha(path)}
               for path in selected}
    write_json(artifact_path, {
        'protocol': 'restart-v3',
        'source_commit': prior['source_commit'],
        'source_file': 'data/raw/set.mm',
        'source_sha256': entries['data/raw/set.mm']['sha256'],
        'fresh_test_statement_signature_overlap_with_prior_index': 0,
        'retained_prior_delivery_files': len(prior['files']),
        'all_new_experiment_data_and_frames_included': True,
        'excluded': ['Earlier ZIP archives', '__pycache__ directories'],
        'self_hash_note': 'This manifest is carried in the ZIP but does not hash itself.',
        'files': entries,
    })
    expected_names = set(entries) | {artifact_path.name}
    temporary = archive_path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in [*selected, artifact_path]:
            archive.write(path, path.relative_to(BASE).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert len(names) == len(expected_names) and set(names) == expected_names
        assert json.loads(archive.read(artifact_path.name))['files'] == entries
    temporary.replace(archive_path)
    for document, target in links:
        assert (workspace_path(document).parent / target).is_file(), (document, target)
    print(json.dumps({'archive': archive_path.name, 'files': len(expected_names),
                      'bytes': archive_path.stat().st_size, 'archive_sha256': sha(archive_path),
                      'zip_crc_and_entry_checks_passed': True,
                      'prior_files_verified': len(prior['files']), 'measurement_rows': len(results)}, indent=2))


if __name__ == '__main__':
    main()
