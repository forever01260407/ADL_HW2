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

    retriever = os.environ.get("HW2_RETRIEVER", "dense")
    if retriever not in {"dense", "bm25", "hybrid"}:
        raise ValueError("HW2_RETRIEVER must be dense, bm25 or hybrid")
    rerank_mode = os.environ.get("HW2_RERANK", "0")
    if rerank_mode not in {"0", "1"}:
        raise ValueError("HW2_RERANK must be 0 or 1")
    rerank_enabled = rerank_mode == "1"
    if rerank_enabled and retriever != "dense":
        raise ValueError("reranking is supported only with dense retrieval in this experiment")
    pids = sorted(passages)
    lexical_results = []
    if retriever in {"bm25", "hybrid"}:
        import re
        from array import array
        from collections import Counter

        # Keep decimal quantities and hyphenated model names as distinct terms.
        term_pattern = re.compile(r"[^\W_]+(?:[.-][^\W_]+)*", re.UNICODE)

        def terms(text):
            return term_pattern.findall(text.casefold())

        fusion_candidates = int(os.environ.get("HW2_RRF_CANDIDATES", "100"))
        if fusion_candidates < 1:
            raise ValueError("HW2_RRF_CANDIDATES must be positive")
        lexical_k = min(len(pids), max(top_k, fusion_candidates)) if retriever == "hybrid" else top_k
        lengths = np.zeros(len(pids), dtype=np.float32)
        postings = {}
        for row, pid in enumerate(pids):
            counts = Counter(terms(passages[pid]))
            lengths[row] = sum(counts.values())
            for term, frequency in counts.items():
                if term not in postings:
                    postings[term] = (array("I"), array("I"))
                postings[term][0].append(row)
                postings[term][1].append(frequency)
        mean_length = max(float(lengths.mean()), 1.0)
        norm = 1.5 * (0.25 + 0.75 * lengths / mean_length)
        for term, (row_buffer, tf_buffer) in postings.items():
            rows = np.frombuffer(row_buffer, dtype=np.uint32)
            tf = np.asarray(tf_buffer, dtype=np.float32)
            idf = np.log1p((len(pids) - len(rows) + 0.5) / (len(rows) + 0.5))
            postings[term] = (rows, np.asarray(idf * tf * 2.5 / (tf + norm[rows]), dtype=np.float32))
        for query in queries:
            scores = np.zeros(len(pids), dtype=np.float32)
            # Each distinct query term contributes once; do not amplify repeated roles/history.
            for term in sorted(set(terms(query["query"]))):
                if term in postings:
                    rows, weights = postings[term]
                    scores[rows] += weights
            threshold = np.partition(scores, len(pids) - lexical_k)[len(pids) - lexical_k]
            rows = np.flatnonzero(scores >= threshold)
            order = rows[np.lexsort((rows, -scores[rows]))][:lexical_k]
            if retriever == "hybrid":
                order = order[scores[order] > 0]  # zero matches must not earn fusion votes
            lexical_results.append([pids[int(row)] for row in order])
        print(f"bm25: products={len(pids)} queries={len(queries)} terms={len(postings)} "
              f"k1=1.5 b=0.75 query_tf=unique lexical_s={time.perf_counter() - started:.3f}", file=sys.stderr)
        del postings, lengths, norm
        if retriever == "bm25":
            return [{"qid": query["qid"], "ranked_pids": ranked}
                    for query, ranked in zip(queries, lexical_results)]

    from sentence_transformers import SentenceTransformer
    candidate_k = top_k
    if retriever == "hybrid":
        candidate_k = min(len(pids), max(top_k, fusion_candidates))
    if rerank_enabled:
        requested_candidates = int(os.environ.get("HW2_RERANK_CANDIDATES", "50"))
        rerank_batch = int(os.environ.get("HW2_RERANK_BATCH_SIZE", "16"))
        if requested_candidates < 1 or rerank_batch < 1:
            raise ValueError("rerank candidate count and batch size must be positive")
        candidate_k = min(len(passages), max(top_k, requested_candidates))
        reranker_path = Path(os.environ.get("HW2_RERANK_MODEL_PATH", str(
            Path(__file__).resolve().parent / "artifacts/models/ms-marco-MiniLM-L6-v2")))
        if not reranker_path.is_dir():
            raise FileNotFoundError(f"Local reranker model not found: {reranker_path}")

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
            probe_k = min(candidate_k + 1, len(pids))
            scores, rows = index.search(vector[None, :], probe_k)
            scores, rows = scores[0], rows[0]
            if probe_k > candidate_k and scores[candidate_k - 1] == scores[candidate_k]:
                # Include ALL boundary ties before applying the PID secondary key.
                scores, rows = index.search(vector[None, :], len(pids))
                scores, rows = scores[0], rows[0]
        else:
            # Exact fallback: only a single query's score array is materialized.
            all_scores = product_vectors @ vector
            threshold = np.partition(all_scores, len(pids) - candidate_k)[len(pids) - candidate_k]
            rows = np.flatnonzero(all_scores >= threshold)
            scores = all_scores[rows]
        if not np.isfinite(scores).all() or (rows < 0).any() or (rows >= len(pids)).any():
            raise ValueError("index returned invalid scores or row IDs")
        order = np.lexsort((rows, -scores))[:candidate_k]
        ranked = [pids[int(rows[i])] for i in order]
        if len(ranked) != candidate_k or len(set(ranked)) != candidate_k:
            raise ValueError("retrieval did not produce unique top-k PIDs")
        results.append({"qid": query["qid"], "ranked_pids": ranked})

    dense_seconds = time.perf_counter() - started
    if retriever == "hybrid":
        for row, lexical in zip(results, lexical_results):
            fused = {}
            for ranked in [row["ranked_pids"], lexical]:
                for rank, pid in enumerate(ranked, 1):
                    fused[pid] = fused.get(pid, 0.0) + 1.0 / (60 + rank)
            row["ranked_pids"] = sorted(fused, key=lambda pid: (-fused[pid], pid))[:top_k]

    if rerank_enabled:
        import gc
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Cross-encode raw dialogue/product pairs only after freeing the encoder.
        device = model.device
        model.to("cpu")
        del model, tokenizer, product_vectors, query_vectors, index
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rr_tokenizer = AutoTokenizer.from_pretrained(
            str(reranker_path), local_files_only=True, trust_remote_code=False)
        rr_model = AutoModelForSequenceClassification.from_pretrained(
            str(reranker_path), local_files_only=True, trust_remote_code=False).to(device)
        rr_model.eval()
        pair_specials = rr_tokenizer.num_special_tokens_to_add(pair=True)
        if pair_specials >= 512 - 192:
            raise ValueError("reranker tokenizer leaves no product token budget")
        rerank_scores = [np.empty(len(row["ranked_pids"]), dtype=np.float32) for row in results]
        pending_inputs, pending_positions = [], []
        rerank_truncated_queries = 0

        def score_pending():
            if not pending_inputs:
                return
            batch = rr_tokenizer.pad(pending_inputs, padding=True, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.inference_mode():
                logits = rr_model(**batch).logits.detach().float().cpu().numpy()
            if logits.shape != (len(pending_positions), 1) or not np.isfinite(logits).all():
                raise ValueError("reranker must return one finite score per candidate")
            for (query_index, candidate_index), value in zip(pending_positions, logits[:, 0]):
                rerank_scores[query_index][candidate_index] = value
            pending_inputs.clear()
            pending_positions.clear()

        for query_index, (query, row) in enumerate(zip(queries, results)):
            query_ids = rr_tokenizer.encode(query["query"], add_special_tokens=False)
            if len(query_ids) > 192:
                query_ids = query_ids[:64] + query_ids[-128:]
                rerank_truncated_queries += 1
            product_budget = 512 - pair_specials - len(query_ids)
            for candidate_index, pid in enumerate(row["ranked_pids"]):
                product_ids = rr_tokenizer.encode(passages[pid], add_special_tokens=False,
                                                  truncation=True, max_length=product_budget)
                pair = rr_tokenizer.prepare_for_model(
                    query_ids, pair_ids=product_ids, add_special_tokens=True,
                    truncation=False, return_attention_mask=True)
                if len(pair["input_ids"]) > 512:
                    raise ValueError("reranker pair exceeds 512 tokens")
                pending_inputs.append(pair)
                pending_positions.append((query_index, candidate_index))
                if len(pending_inputs) == rerank_batch:
                    score_pending()
        score_pending()
        for row, scores in zip(results, rerank_scores):
            # Dense order is the tie-breaker; candidates cannot enter from outside.
            order = np.lexsort((np.arange(len(scores)), -scores))[:top_k]
            row["ranked_pids"] = [row["ranked_pids"][int(i)] for i in order]
        print(f"rerank: model={reranker_path} candidates={candidate_k} "
              f"pairs={len(queries) * candidate_k} batch={rerank_batch} "
              f"truncated_queries={rerank_truncated_queries} "
              f"rerank_s={time.perf_counter() - started - dense_seconds:.3f}", file=sys.stderr)

    print(f"retrieval: products={len(pids)} queries={len(queries)} model={model_path} "
          f"batch={batch_size} backend={'faiss' if faiss is not None else 'numpy'} "
          f"retriever={retriever} rerank={rerank_enabled} dense_s={dense_seconds:.3f} "
          f"truncated_queries={truncated_queries} "
          f"load_and_products_s={product_seconds:.3f} total_function_s={time.perf_counter()-started:.3f}",
          file=sys.stderr)
    # ===================== STUDENT CODE END =====================

    return results
