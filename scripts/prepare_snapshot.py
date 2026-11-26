"""Build the lightweight Git snapshot from the untouched research archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    archive_hash = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(args.archive) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("ARTIFACTS_V3.json"))
        packaged = {}
        for name, record in manifest["files"].items():
            # Preserve author code, original documentation, small evidence and figures.
            if name.endswith((".pkl", ".gz", ".npz", ".mm")) or name.startswith("data/processed/") or record["bytes"] >= 750_000:
                continue
            content = archive.read(name)
            assert hashlib.sha256(content).hexdigest() == record["sha256"], name
            destination = (ROOT / "study" / name).resolve()
            assert destination.is_relative_to(ROOT / "study")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            packaged["study/" + name] = record
        content = archive.read("ARTIFACTS_V3.json")
        (ROOT / "study/ARTIFACTS_V3.json").write_bytes(content)
        packaged["study/ARTIFACTS_V3.json"] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    copies = {
        "restart_stages.png": "runs/restart_v3/figures/restart_stages.png",
        "topic_transfer.png": "runs/restart_v3/figures/topic_transfer.png",
        "signed_effects_v1.png": "runs/scoped_v1/figures/signed_effects.png",
    }
    asset_dir = ROOT / "assets/figures"
    asset_dir.mkdir(parents=True, exist_ok=True)
    for destination, original in copies.items():
        shutil.copyfile(ROOT / "study" / original, asset_dir / destination)
    config = {
        "repository": "gjtjwns06-ui/one-wrong-edge", "release": "v0.3.0",
        "archive": args.archive.name, "archive_sha256": archive_hash,
        "archive_bytes": args.archive.stat().st_size,
        "source_commit": manifest["source_commit"], "experiment_date": "2026-11-25",
        "publication_date": "2026-11-26", "snapshot": "restart-v3",
        "note": "study/ mirrors original archive paths. Research artifacts remain byte-for-byte unchanged.",
        "study_files": packaged,
        "figure_sources": {"assets/figures/" + dest: "study/" + src for dest, src in copies.items()},
    }
    (ROOT / "PUBLICATION.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Prepared {len(packaged)} frozen research files; {sum(v['bytes'] for v in packaged.values()):,} bytes.")
    print(f"Archive SHA256: {archive_hash}")


if __name__ == "__main__":
    main()
