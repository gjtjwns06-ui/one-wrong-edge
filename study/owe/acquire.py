"""Download an immutable, date-bounded Metamath snapshot and its licence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from datetime import datetime, timezone


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "OneWrongEdge-research/0.1"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--until", default="2026-11-23T14:59:59Z")
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sha = manifest["commit"]
    else:
        api = "https://api.github.com/repos/metamath/set.mm/commits?sha=develop&per_page=1&until=" + args.until
        commits = json.loads(download(api))
        if not commits:
            raise RuntimeError("No source commit before the requested cutoff")
        sha = commits[0]["sha"]
        manifest = {"repository": "https://github.com/metamath/set.mm", "commit": sha,
                    "commit_date": commits[0]["commit"]["committer"]["date"],
                    "requested_cutoff": args.until, "licence": "CC0-1.0", "files": {}}
    for name in ["set.mm", "iset.mm", "hol.mm", "LICENSE", "README.md", "verifiers.md"]:
        url = f"https://raw.githubusercontent.com/metamath/set.mm/{sha}/{name}"
        path = root / name
        if path.exists() and name in manifest["files"]:
            body = path.read_bytes()
            if hashlib.sha256(body).hexdigest() != manifest["files"][name]["sha256"]:
                raise RuntimeError(f"Checksum mismatch: {path}")
        else:
            body = download(url)
            path.write_bytes(body)
        manifest["files"][name] = {"url": url, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {len(body):,} bytes {manifest['files'][name]['sha256'][:12]}", flush=True)
    print(json.dumps({"commit": sha, "commit_date": manifest["commit_date"]}))


if __name__ == "__main__":
    main()
