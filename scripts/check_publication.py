"""Check frozen research bytes, README evidence, figure provenance and links."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_links(path):
    text = path.read_text(encoding="utf-8")
    targets = re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text)
    targets += re.findall(r'<img\b[^>]*\bsrc="([^"]+)"', text)
    count = 0
    for target in targets:
        target = target.strip().strip("<>")
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
            continue
        name = unquote(target.split("#", 1)[0])
        resolved = (path.parent / name).resolve()
        assert resolved.is_relative_to(ROOT), target
        assert resolved.exists(), f"Broken local link in {path.name}: {target}"
        count += 1
    return count


def main():
    config = json.loads((ROOT / "PUBLICATION.json").read_text(encoding="utf-8"))
    for name, expected in config["study_files"].items():
        path = ROOT / name
        assert path.is_file(), f"Missing research file: {name}"
        assert path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"], f"Original research artifact changed: {name}"
    for asset, source in config["figure_sources"].items():
        assert sha(ROOT / asset) == sha(ROOT / source), f"Figure is no longer identical to {source}"
    summary = list(csv.DictReader((ROOT / "study/runs/restart_v3/summary.csv").open(encoding="utf-8")))
    comparison = list(csv.DictReader((ROOT / "study/runs/restart_v3/comparisons.csv").open(encoding="utf-8")))
    clean = {r["method"]: float(r["recall"]) for r in summary if r["condition"] == "clean" and r["budget"] == "20"}
    assert f"{clean['proof_ranker']*100:.2f}%" == "26.41%"
    assert f"{clean['lexical']*100:.2f}%" == "19.56%"
    assert f"{clean['old_local']*100:.2f}%" == "14.46%"
    effect = next(r for r in comparison if r["condition"] == "global20_mean" and r["budget"] == "20" and r["comparison"] == "utility minus proof_ranker")
    assert float(effect["ci_low"]) <= 0 <= float(effect["ci_high"])
    assert round(float(effect["difference"])*100, 3) == .006
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for literal in ("26.41%", "19.56%", "14.46%", "+0.006pp", "582", "29,862", "6,866"):
        assert literal in readme, f"README headline missing: {literal}"
    complete = json.loads((ROOT / "study/runs/restart_v3/run_complete.json").read_text())
    assert complete["test_queries"] == 582 and complete["measurement_rows"] == 167616
    audit = json.loads((ROOT / "study/runs/restart_v3/audit.json").read_text())
    assert audit["all_passed"] and audit["audit_scope"] == "full_experiment"
    links = sum(check_links(ROOT / name) for name in ("README.md", "NOTICE.md", "docs/REPRODUCING.md"))
    receipt = dict(status="passed", frozen_research_files=len(config["study_files"]),
                   unchanged_result_figures=len(config["figure_sources"]), local_links=links,
                   headline_values_match_source_tables=True, full_data_available_via_release=True)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
