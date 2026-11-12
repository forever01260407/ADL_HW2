#!/usr/bin/env python3
"""H5: one FP32 continuation from the selected base; no Public supervision."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import resource
import time
from datetime import datetime, timezone

from analyze_results import sha256
from hard_negative_data import ROOT, batch_negatives, frozen, read, select_verified, verify_hashes
from hard_negative_loss import own_negative_loss
from train import load_data, make_batches, prepare_query, write_json


def training_loop(model, rows, products, selected, batches, query_texts, config, output):
    import torch
    from transformers import get_linear_schedule_with_warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    warmup = math.ceil(len(batches) * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, len(batches))
    trained_negatives = masked = 0
    model.train()
    started = time.perf_counter()
    with (output / "loss.jsonl").open("x", encoding="utf-8", buffering=1) as handle:
        for step, batch in enumerate(batches, 1):
            optimizer.zero_grad(set_to_none=True)
            part = [rows[i] for i in batch]
            negative_rows, negative_texts, duplicate_count = batch_negatives(part, products, selected)
            texts = [[query_texts[i] for i in batch], [products[r["gold_pid"]] for r in part]]
            if negative_texts:
                texts.append(negative_texts)
            embeddings = []
            for stream in texts:
                features = {k: v.to(model.device) for k, v in model.tokenize(stream).items()}
                embeddings.append(model(features)["sentence_embedding"])
            loss = own_negative_loss(embeddings[0], embeddings[1],
                embeddings[2] if negative_texts else None, negative_rows, config["scale"])
            if not torch.isfinite(loss):
                raise ValueError("non-finite hard-negative loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
            if not torch.isfinite(norm) or norm.item() <= 0:
                raise ValueError("invalid gradient norm")
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            trained_negatives += len(negative_rows)
            masked += duplicate_count
            row = {"step": step, "batch_size": len(batch), "loss": loss.item(), "learning_rate": lr,
                "grad_norm_before_clip": norm.item(), "own_verified_negatives": len(negative_rows),
                "duplicate_stream_negatives_masked": duplicate_count}
            handle.write(json.dumps(row) + "\n")
            if step == 1 or step % 20 == 0 or step == len(batches):
                print(json.dumps(row), flush=True)
    torch.cuda.synchronize()
    return {"training_wall_seconds": time.perf_counter() - started, "warmup_steps": warmup,
        "trained_own_negative_exposures": trained_negatives, "duplicate_stream_negatives_masked": masked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "train"], default="train")
    args = parser.parse_args()
    plan = frozen()
    config = plan["configuration"]
    verified = read("results/h5_verified_negatives.json")
    verify_hashes(verified["input_sha256"])
    rows, products, data_hashes = load_data()
    candidates = read("results/h5_negative_candidates.json")["records"]
    decisions = [json.loads(line) for line in (ROOT / "results/h5_negative_teacher.jsonl").read_text().splitlines()]
    selected = select_verified(rows, products, candidates, decisions)
    if selected != verified["negatives"] or not set(selected) <= {r["qid"] for r in rows}:
        raise ValueError("negative selection or train-only boundary changed")
    output = ROOT / config["smoke_output" if args.mode == "smoke" else "output"]
    if not output.resolve().is_relative_to(ROOT / "artifacts/training"):
        raise ValueError("training output must remain inside workspace artifacts/training")
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="4")
    started = time.perf_counter()
    report = {"status": "running", "mode": args.mode, "configuration": config,
        "started_utc": datetime.now(timezone.utc).isoformat(), "dev_used_for_gradient": False,
        "dev_used_for_mining": False, "public_used": False, "official_hardware_validation": False}
    write_json(output / "training_manifest.json", report)
    try:
        import numpy as np
        import torch
        import transformers
        import sentence_transformers
        from sentence_transformers import SentenceTransformer
        if not torch.cuda.is_available():
            raise ValueError("CUDA required")
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(config["cuda_allocator_budget_gib"] * 2**30 /
            torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        model = SentenceTransformer(str(ROOT / config["source_checkpoint"]), device="cuda",
            local_files_only=True, trust_remote_code=False)
        model.max_seq_length = 512
        model.float()
        model[0].auto_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if not model[0].auto_model.is_gradient_checkpointing or model.get_sentence_embedding_dimension() != 768:
            raise ValueError("gradient checkpointing or base dimension incorrect")
        batches = make_batches(rows, products, config["batch_size"], config["seed"])
        if args.mode == "smoke":
            first = next(b for b in batches if batch_negatives([rows[i] for i in b], products, selected)[0])
            batches = [first] * 4
        elif (ROOT / config["smoke_output"] / "training_manifest.json").is_file():
            smoke = read(f'{config["smoke_output"]}/training_manifest.json')
            if smoke["status"] != "completed" or smoke["trained_own_negative_exposures"] <= 0:
                raise ValueError("engineering smoke did not pass")
        else:
            raise ValueError("complete the engineering smoke before quality training")
        qids = [[rows[i]["qid"] for i in batch] for batch in batches]
        query_texts = {i: prepare_query(rows[i]["query"], model.tokenizer) for batch in batches for i in batch}
        write_json(output / "batches.json", qids)
        report.update({"train_count": len({qid for batch in qids for qid in batch}), "optimizer_steps": len(batches),
            "gradient_qids": sorted({qid for batch in qids for qid in batch}), "precision": "float32",
            "gradient_checkpointing": True, "cuda_allocator_budget_gib": config["cuda_allocator_budget_gib"],
            "batch_sizes": [len(b) for b in batches], "batches_sha256": sha256(output / "batches.json"),
            "selected_negative_queries": len(selected), "deterministic_algorithms": True,
            "gpu": torch.cuda.get_device_name(), "versions": {"torch": torch.__version__,
                "transformers": transformers.__version__, "sentence_transformers": sentence_transformers.__version__},
            "input_sha256": {**data_hashes, **{p: sha256(ROOT / p) for p in [
                "results/h5_training_plan.json", "results/h5_verified_negatives.json", "code/script/train_hard_negative.py"]}}})
        tracked = next(p for p in model.parameters() if p.requires_grad)
        before = tracked.detach().clone()
        report.update(training_loop(model, rows, products, selected, batches, query_texts, config, output))
        report["tracked_parameter_max_abs_change"] = (tracked.detach() - before).abs().max().item()
        if report["tracked_parameter_max_abs_change"] <= 0 or report["trained_own_negative_exposures"] <= 0:
            raise ValueError("no weight update or verified negatives trained")
        model.eval()
        probes = [query_texts[batches[0][0]], products[rows[batches[0][0]]["gold_pid"]]]
        expected = model.encode(probes, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        checkpoint = output / "checkpoint"
        model.save(str(checkpoint), safe_serialization=True, create_model_card=False)
        del model, tracked, before
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = SentenceTransformer(str(checkpoint), device="cuda", local_files_only=True, trust_remote_code=False)
        actual = reloaded.encode(probes, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        if actual.shape != (2, 768):
            raise ValueError("reloaded base vectors have wrong dimensions")
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        report.update({"status": "completed", "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
            "checkpoint_files": {p.relative_to(checkpoint).as_posix(): sha256(p) for p in sorted(checkpoint.rglob("*")) if p.is_file()},
            "offline_reload_max_abs_difference": float(np.max(np.abs(actual - expected))),
            "loss_sha256": sha256(output / "loss.jsonl"), "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "wall_seconds": time.perf_counter() - started})
        if max(report["peak_cuda_allocated_bytes"], report["peak_cuda_reserved_bytes"]) > 8 * 2**30:
            raise ValueError("training exceeded allocator budget")
        write_json(output / "training_manifest.json", report)
        print(json.dumps({k: report[k] for k in ["status", "train_count", "optimizer_steps", "trained_own_negative_exposures",
            "training_wall_seconds", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "offline_reload_max_abs_difference"]}), flush=True)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", wall_seconds=time.perf_counter() - started)
        write_json(output / "training_manifest.json", report)
        raise


if __name__ == "__main__":
    main()
