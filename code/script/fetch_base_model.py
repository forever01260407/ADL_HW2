#!/usr/bin/env python3
"""Download only pinned official BGE-base assets; verify Hub blob/LFS hashes."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    metadata = json.loads((ROOT / "results/h3_base_remote_metadata.json").read_text())
    target = ROOT / "artifacts/models/bge-base-en-v1.5"
    manifest = ROOT / "results/h3_base_download.json"
    lock_path = ROOT / "code/script/configs/h3_base_model.lock.json"
    if manifest.exists() or lock_path.exists():
        raise ValueError("download already recorded; do not overwrite pinned model history")
    target.mkdir(parents=True, exist_ok=True)
    report = {"repo_id": metadata["repo_id"], "revision": metadata["revision"],
              "path": target.relative_to(ROOT).as_posix(), "files": {}}
    for name, info in metadata["files"].items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            url = f'https://huggingface.co/{metadata["repo_id"]}/resolve/{metadata["revision"]}/{name}?download=true'
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=90) as source, path.with_suffix(path.suffix + ".part").open("wb") as out:
                        while chunk := source.read(1024 * 1024):
                            out.write(chunk)
                    path.with_suffix(path.suffix + ".part").replace(path)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)
        size = path.stat().st_size
        digest = hashlib.sha256()
        blob = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                blob.update(chunk)
        if size != info["size"]:
            raise ValueError(f"download size mismatch: {name}")
        if "lfs" in info:
            if digest.hexdigest() != info["lfs"]["sha256"]:
                raise ValueError(f"LFS checksum mismatch: {name}")
        elif blob.hexdigest() != info["blobId"]:
            raise ValueError(f"Git blob checksum mismatch: {name}")
        report["files"][name] = {"bytes": size, "sha256": digest.hexdigest()}
        print(json.dumps({"file": name, "bytes": size, "verified": True}), flush=True)
    lock = {key: report[key] for key in ["repo_id", "revision", "path"]}
    lock.update({"dimension": 768, "max_seq_length": 512, "download_manifest": "results/h3_base_download.json"})
    for path, value in [(manifest, report), (lock_path, lock)]:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"bytes": sum(row["bytes"] for row in report["files"].values()), "files": len(report["files"])}))


if __name__ == "__main__":
    main()
