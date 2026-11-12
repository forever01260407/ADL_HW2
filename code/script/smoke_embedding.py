#!/usr/bin/env python3
"""Small real-model compatibility check; this is not a retrieval benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTRUCTION = "Represent this sentence for searching relevant passages: "


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=128)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    import faiss
    import numpy as np
    import torch
    import transformers
    import sentence_transformers
    from sentence_transformers import SentenceTransformer

    lock = json.loads((ROOT / "code/script/configs/models.lock.json").read_text())
    entry = lock["embedding"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    model = SentenceTransformer(str(ROOT / entry["path"]), device=device,
                                local_files_only=True, trust_remote_code=False)
    model.max_seq_length = 512
    products, queries = [], []
    with (ROOT / "data/train/products.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            products.append(row["title"] + "\n" + row["description"])
            if len(products) == args.sample_size:
                break
    with (ROOT / "data/train/dialogues.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            queries.append(INSTRUCTION + "\n".join(
                f"{turn['role']}: {turn['content']}" for turn in row["conversation"]))
            if len(queries) == args.sample_size:
                break
    lengths = {
        key: [len(ids) for ids in model.tokenizer(texts, truncation=False)["input_ids"]]
        for key, texts in [("products", products), ("queries", queries)]
    }
    vectors = model.encode(products, batch_size=32, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    qvec = model.encode(queries, batch_size=32, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)
    if device == "cuda":
        torch.cuda.synchronize()
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    qvec = np.ascontiguousarray(qvec, dtype=np.float32)
    assert vectors.shape == (len(products), 384), vectors.shape
    assert qvec.shape == (len(queries), 384), qvec.shape
    assert np.isfinite(vectors).all() and np.isfinite(qvec).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    assert np.allclose(np.linalg.norm(qvec, axis=1), 1, atol=1e-5)
    index = faiss.IndexFlatIP(384)
    index.add(vectors)
    scores, rows = index.search(qvec[:3], min(10, len(products)))
    assert (rows >= 0).all() and (rows < len(products)).all()
    reference = qvec[:3] @ vectors.T
    expected = np.sort(reference, axis=1)[:, -scores.shape[1]:][:, ::-1]
    assert np.allclose(scores, expected, atol=2e-6), "FAISS scores differ from exact dot product"
    assert np.allclose(scores, np.take_along_axis(reference, rows, axis=1), atol=2e-6)
    report = {
        "kind": "compatibility_smoke_not_retrieval_quality",
        "model": entry, "device": device,
        "sample_policy": "first N train products and train dialogues; no gold used",
        "samples": {key: len(value) for key, value in lengths.items()},
        "token_lengths": {key: {"min": min(value), "max": max(value),
            "p50": float(np.quantile(value, .5)), "p95": float(np.quantile(value, .95)),
            "over_512": sum(n > 512 for n in value)} for key, value in lengths.items()},
        "vector_dimension": 384, "faiss_exact_score_check": True,
        "elapsed_seconds_including_model_load": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
        "versions": {"torch": torch.__version__, "faiss": faiss.__version__, "numpy": np.__version__,
                     "transformers": transformers.__version__, "sentence_transformers": sentence_transformers.__version__},
    }
    output = ROOT / "results/w01_embedding_smoke.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
