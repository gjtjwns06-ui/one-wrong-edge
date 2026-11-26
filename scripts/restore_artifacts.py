"""Restore the original research release after archive and per-file validation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore(archive_path, config):
    if sha(archive_path) != config["archive_sha256"]:
        raise ValueError("Archive SHA-256 differs from the pinned release")
    destination = ROOT / "study"
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("ARTIFACTS_V3.json"))
        expected = set(manifest["files"]) | {"ARTIFACTS_V3.json"}
        if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
            raise ValueError("Unexpected archive entries")
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC check failed")
        # Preflight every entry before writing or replacing anything.
        for name in sorted(expected):
            path = (destination / name).resolve()
            if not path.is_relative_to(destination.resolve()):
                raise ValueError(f"Path outside study directory: {name}")
            content = archive.read(name)
            digest = hashlib.sha256(content).hexdigest()
            if name != "ARTIFACTS_V3.json" and digest != manifest["files"][name]["sha256"]:
                raise ValueError(f"Manifest mismatch: {name}")
            if path.exists() and (not path.is_file() or sha(path) != digest):
                raise ValueError(f"Existing file changed; keeping it untouched: {name}")
        restored = 0
        for name in sorted(expected):
            path = destination / name
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(name))
            restored += 1
    print(f"Verified complete archive; restored {restored} missing files into study/.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a local copy of the original ZIP")
    args = parser.parse_args()
    config = json.loads((ROOT / "PUBLICATION.json").read_text(encoding="utf-8"))
    archive_path = args.archive
    if archive_path is None:
        cache = ROOT / ".artifacts"
        cache.mkdir(exist_ok=True)
        archive_path = cache / config["archive"]
        if not archive_path.exists():
            url = f"https://github.com/{config['repository']}/releases/download/{config['release']}/{config['archive']}"
            temporary = archive_path.with_suffix(".zip.tmp")
            print(f"Downloading {config['archive']} ({config['archive_bytes']:,} bytes)...", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "OneWrongEdge-reproduction"})
            with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
            if sha(temporary) != config["archive_sha256"]:
                raise ValueError("Downloaded archive failed SHA-256 validation")
            temporary.replace(archive_path)
    restore(archive_path, config)


if __name__ == "__main__":
    main()
