#!/usr/bin/env python3
"""Student-editable retrieval and reranking implementation."""

from __future__ import annotations

from typing import Any


DEFAULT_TOP_K = 10


def retrieve_and_rerank(
    queries: list[dict[str, Any]],
    passages: dict[str, str],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Return one final ranked PID list per query."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if len(passages) < top_k:
        raise ValueError(
            f"product collection has {len(passages)} items, fewer than top_k={top_k}"
        )

    # ==================== STUDENT CODE START ====================
    # Only modify the code inside this block.
    #
    # Implement retrieval and optional reranking here. Return one dictionary
    # per query in the following form:
    # {
    #     "qid": query["qid"],
    #     "ranked_pids": [pid_1, pid_2, ..., pid_top_k],
    # }
    import os
    import sys
    import time
    from pathlib import Path

    import numpy as np
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("top_k must be an integer")
    if any(not isinstance(pid, str) or not pid.strip() or not isinstance(text, str)
           or not text.strip() for pid, text in passages.items()):
        raise ValueError("passages must contain non-empty string PIDs and texts")
    seen_qids = set()
    for query in queries:
        qid, text = query.get("qid"), query.get("query")
        if not isinstance(qid, str) or not qid.strip() or qid in seen_qids:
            raise ValueError("query qids must be unique non-empty strings")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"query {qid} must contain non-empty text")
        seen_qids.add(qid)
    if not queries:
        return []

    model_path = Path(os.environ.get("HW2_MODEL_PATH", str(
        Path(__file__).resolve().parent / "artifacts/models/bge-small-en-v1.5")))
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local embedding model not found: {model_path}; run download.sh first")
    batch_size = int(os.environ.get("HW2_BATCH_SIZE", "32"))
    if batch_size < 1:
        raise ValueError("HW2_BATCH_SIZE must be positive")
    model = SentenceTransformer(str(model_path), local_files_only=True, trust_remote_code=False)
    model.max_seq_length = 512

    def encode(texts):
        vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts) or not np.isfinite(vectors).all():
            raise ValueError("encoder returned invalid or non-finite vectors")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise ValueError("encoder returned zero-length vectors")
        return np.ascontiguousarray(vectors / norms, dtype=np.float32)

    # Sorting makes row-to-PID mapping independent of the input dictionary order.
    pids = sorted(passages)
    product_vectors = encode([passages[pid] for pid in pids])
    product_seconds = time.perf_counter() - started
    try:
        import faiss
    except ImportError:
        faiss = None
    index = None
    if faiss is not None:
        index = faiss.IndexFlatIP(product_vectors.shape[1])
        index.add(product_vectors)

    instruction = "Represent this sentence for searching relevant passages: "
    tokenizer = model.tokenizer
    truncated_queries = 0

    def prepare_query(raw):
        nonlocal truncated_queries
        text = instruction + raw
        if len(tokenizer.encode(text, add_special_tokens=True)) <= 512:
            return text  # preserve short transcripts byte for byte, including roles
        truncated_queries += 1
        ids = tokenizer.encode(raw, add_special_tokens=False)
        budget = 512 - tokenizer.num_special_tokens_to_add(pair=False) - len(
            tokenizer.encode(instruction, add_special_tokens=False))
        # Decoding and re-tokenization can change length. Recheck the actual input.
        while budget > 0:
            head = min(128, budget // 2)
            content = tokenizer.decode(ids[:head] + ids[-(budget - head):], skip_special_tokens=True)
            text = instruction + content
            overflow = len(tokenizer.encode(text, add_special_tokens=True)) - 512
            if overflow <= 0:
                return text
            budget -= max(1, overflow)
        raise ValueError("query cannot fit the encoder token budget")

    query_vectors = encode([prepare_query(query["query"]) for query in queries])
    if query_vectors.shape[1] != product_vectors.shape[1]:
        raise ValueError("query and product vector dimensions differ")
    results: list[dict[str, Any]] = []
    for query, vector in zip(queries, query_vectors):
        if index is not None:
            # Search one row at a time for consistent accumulation and tie behavior.
            probe_k = min(top_k + 1, len(pids))
            scores, rows = index.search(vector[None, :], probe_k)
            scores, rows = scores[0], rows[0]
            if probe_k > top_k and scores[top_k - 1] == scores[top_k]:
                # Include ALL boundary ties before applying the PID secondary key.
                scores, rows = index.search(vector[None, :], len(pids))
                scores, rows = scores[0], rows[0]
        else:
            # Exact fallback: only a single query's score array is materialized.
            all_scores = product_vectors @ vector
            threshold = np.partition(all_scores, len(pids) - top_k)[len(pids) - top_k]
            rows = np.flatnonzero(all_scores >= threshold)
            scores = all_scores[rows]
        if not np.isfinite(scores).all() or (rows < 0).any() or (rows >= len(pids)).any():
            raise ValueError("index returned invalid scores or row IDs")
        order = np.lexsort((rows, -scores))[:top_k]
        ranked = [pids[int(rows[i])] for i in order]
        if len(ranked) != top_k or len(set(ranked)) != top_k:
            raise ValueError("retrieval did not produce unique top-k PIDs")
        results.append({"qid": query["qid"], "ranked_pids": ranked})
    print(f"retrieval: products={len(pids)} queries={len(queries)} model={model_path} "
          f"batch={batch_size} backend={'faiss' if index is not None else 'numpy'} "
          f"truncated_queries={truncated_queries} "
          f"load_and_products_s={product_seconds:.3f} total_function_s={time.perf_counter()-started:.3f}",
          file=sys.stderr)
    # ===================== STUDENT CODE END =====================

    return results
