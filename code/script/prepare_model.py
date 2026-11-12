#!/usr/bin/env python3
"""Download an allowlisted model at a recorded immutable Hub revision."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "embedding": ("BAAI/bge-small-en-v1.5", "bge-small-en-v1.5"),
    "generator": ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B-Instruct"),
    "reranker": ("cross-encoder/ms-marco-MiniLM-L6-v2", "ms-marco-MiniLM-L6-v2"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=MODELS, required=True)
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    lock_path = ROOT / "code/script/configs/models.lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    repo_id, directory = MODELS[args.role]
    entry = lock.get(args.role)
    if entry and entry["repo_id"] != repo_id:
        raise ValueError("Existing model lock does not match the allowlisted repository")
    revision = entry["revision"] if entry else HfApi(token=False).model_info(repo_id).sha
    if not revision or len(revision) != 40:
        raise ValueError("Hub did not return an immutable 40-character commit")
    model_path = ROOT / "artifacts/models" / directory
    lock[args.role] = {
        "repo_id": repo_id,
        "revision": revision,
        "path": model_path.relative_to(ROOT).as_posix(),
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    info = HfApi(token=False).model_info(repo_id, revision=revision)
    has_safe_weights = any(s.rfilename.endswith(".safetensors") for s in info.siblings)
    patterns = ["*.json", "*.txt", "tokenizer.model"]
    patterns.append("*.safetensors" if has_safe_weights else "pytorch_model*.bin")
    expected = {
        item.rfilename for item in info.siblings
        if any(fnmatch.fnmatch(item.rfilename, pattern) for pattern in patterns)
    }
    existing = {
        path.relative_to(model_path).as_posix() for path in model_path.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(model_path).parts
    }
    if existing - expected:
        raise ValueError(f"Model directory has files outside the pinned snapshot: {sorted(existing - expected)}")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(model_path),
        cache_dir=str(ROOT / "artifacts/hub-cache"),
        allow_patterns=patterns,
        token=False,
        max_workers=2,
    )
    if not any(name.endswith((".safetensors", ".bin")) for name in expected):
        raise ValueError("Pinned repository has no supported model weights")
    if "config.json" not in expected or not {"tokenizer.json", "vocab.txt", "tokenizer.model"} & expected:
        raise ValueError("Pinned repository is missing model or tokenizer configuration")
    missing = [name for name in expected if not (model_path / name).is_file() or not (model_path / name).stat().st_size]
    if missing:
        raise ValueError(f"Downloaded snapshot is incomplete: {missing}")
    files = {
        name: {
            "bytes": (model_path / name).stat().st_size,
            "sha256": sha256(model_path / name),
        }
        for name in sorted(expected)
    }
    manifest = {**lock[args.role], "files": files}
    output = ROOT / "results" / f"{args.role}_download.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"role": args.role, "revision": revision, "files": len(files),
                      "bytes": sum(row["bytes"] for row in files.values())}))


if __name__ == "__main__":
    main()
