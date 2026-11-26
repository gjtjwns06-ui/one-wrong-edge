# Reproducing One Wrong Edge

The Git repository is a readable research publication. The original completed experiment is mirrored under `study/`, with authored code, unchanged reports, protocols, selected tables and figures. Large files are restored from a fixed release archive. Original files are kept at the same relative paths inside `study/` so the scientific manifests and audit code continue to work.

## 1. Restore the complete snapshot

Python 3.13 was used for the completed runs. The bootstrap uses only Python's standard library:

```bash
git clone https://github.com/gjtjwns06-ui/one-wrong-edge.git
cd one-wrong-edge
python scripts/restore_artifacts.py
```

The archive is `One_Wrong_Edge_Restart_v3.zip`, 153,037,639 bytes. Its SHA-256 is:

```text
3eb09e60eb70d0ca9cb6fd47e7b75a030ef6e51aff56c2244ae4cc244454ad50
```

The bootstrap validates the full archive, its entry paths and every original manifest digest before restoring files into `study/`. It skips already identical files and refuses to overwrite changed ones. To use an archive downloaded separately:

```bash
python scripts/restore_artifacts.py --archive /path/to/One_Wrong_Edge_Restart_v3.zip
```

The source archive includes the pinned Metamath source, reparsed frames, original v1/v2 delivery files, v3 datasets, models, original measurement rows, rankings, intervention cases and audit receipts. The Git snapshot alone is intentionally insufficient for a full data-dependent audit. Some original report links become available locally after restoration.

## 2. Install the pinned environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r study/requirements_v2.txt
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r study/requirements_v2.txt
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
```

The original runs used CPU only on Windows with Python 3.13.5 and approximately 16GB system RAM. Timing varies by hardware. The full scientific experiment has not been asserted to have identical timing or floating-point behavior across operating systems.

## 3. Inspect and audit a completed run

```bash
cd study
python -m pytest -q tests
python -m owe.restart_v3 infer --qid trrelssd --condition clean
python -m owe.restart_v3 infer --qid trrelssd --condition random20_s401 --policy utility
python -m owe.audit_restart_v3
```

The audit reads completed artifacts and may refresh its audit receipt. Its full scope reconstructs metric rows and checks the frozen selections and original file hashes. Read `owe/audit_restart_v3.py` and the existing `runs/restart_v3/audit.json` for its exact coverage. Formal-query inference is a stored theorem-ID interface; it is not a general natural-language question-answering service.

## 4. Rebuild from the fixed source

Use a separate copy when retraining. Completed run/selection directories are intentionally protected against replacement. Copy `study/` into a new working folder, retain the fixed raw source manifest and prior scoped dataset used to exclude old statement signatures, and choose a fresh working copy without completed `data/processed/restart_v3` and `runs/restart_v3` directories. Do not remove the preserved delivery in place.

Follow the original [v3 execution instructions](../study/README_V3.md) and [frozen protocol](../study/RESTART_V3_PROTOCOL.md):

```bash
python -m owe.prepare_v3
python -m owe.restart_v3 index
python -m owe.restart_v3 candidates
python -m owe.restart_v3 rank
python -m owe.restart_v3 policy
python -m owe.restart_v3 test
python -m owe.analyze_restart_v3
python -m owe.audit_restart_v3
python -m pytest -q tests
python -m owe.analyze_restart_v3 --reports-only
```

## Publication checks and CI

From the repository root:

```bash
python scripts/check_publication.py
```

This validates the tracked original research-file hashes, figure provenance, landing-page links, release metadata and headline numbers against committed result tables. GitHub Actions also runs data-independent research tests. Dataset-dependent tests skip when the release data is absent; a green lightweight CI run does not stand in for the full archived research audit.

The banner can be regenerated with `python scripts/build_banner.py` after installing the study requirements. The result plots shown in the README are unchanged copies from the completed runs, with provenance in `PUBLICATION.json`.
