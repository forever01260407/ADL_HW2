#!/usr/bin/env python3
"""H3 encoder substitution using the unchanged W08 data/sampler/query helpers."""
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
from pathlib import Path

from analyze_results import sha256
from train import ROOT, load_data, make_batches, prepare_query, write_json


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def reference_batches(rows, products, config):
    batches = make_batches(rows, products, config["batch_size"], config["seed"])
    qids = [[rows[i]["qid"] for i in batch] for batch in batches]
    if qids != read("artifacts/training/w08_epoch1/batches.json"):
        raise ValueError("H3 batches differ from the actual W08 qid/negative grouping")
    return batches, qids


def verify_checkpoint_shape(actual, expected, dimension, max_seq_length):
    import numpy as np
    if max_seq_length != 512 or actual.shape != (2, dimension) or not np.isfinite(actual).all():
        raise ValueError("invalid encoder checkpoint shape/values")
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "train"], default="train")
    args = parser.parse_args()
    experiment = read("code/script/configs/h3_experiment.json")
    config = read(experiment["reference_training_config"])
    lock = read(experiment["model_lock"])
    plan = read("results/h3_experiment_plan.json")
    for path, expected in plan["input_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"changed frozen H3 source/input: {path}")
    download = read(lock["download_manifest"])
    for path, row in download["files"].items():
        if sha256(ROOT / lock["path"] / path) != row["sha256"]:
            raise ValueError(f"base pretrained asset changed: {path}")
    rows, products, hashes = load_data()
    batches, batch_qids = reference_batches(rows, products, config)
    if args.mode == "smoke":
        batches, batch_qids = [batches[0]] * 4, [batch_qids[0]] * 4
    output = ROOT / experiment["smoke_output" if args.mode == "smoke" else "output"]
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    started = time.perf_counter()
    report = {"status": "running", "mode": args.mode, "experiment": experiment["experiment"],
        "config": config, "pretrained_model": lock, "started_utc": datetime.now(timezone.utc).isoformat(),
        "reference_w08_batches_identical": True, "official_hardware_validation": False}
    write_json(output / "training_manifest.json", report)
    try:
        import numpy as np
        import torch
        import transformers
        import sentence_transformers
        from sentence_transformers import SentenceTransformer, losses
        from transformers import get_linear_schedule_with_warmup
        if not torch.cuda.is_available():
            raise ValueError("CUDA is required for this fixed FP32 training experiment")
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = SentenceTransformer(str(ROOT / lock["path"]), device="cuda", local_files_only=True, trust_remote_code=False)
        model.max_seq_length = 512
        model.float()
        if model.get_sentence_embedding_dimension() != lock["dimension"]:
            raise ValueError("base embedding dimension differs from the pinned model")
        query_texts = {i: prepare_query(rows[i]["query"], model.tokenizer) for batch in batches for i in batch}
        write_json(output / "batches.json", batch_qids)
        report.update({"train_count": len({i for batch in batches for i in batch}),
            "candidate_pool_count": len(products), "gradient_qids": sorted({rows[i]["qid"] for batch in batches for i in batch}),
            "optimizer_steps": len(batches), "batch_sizes": [len(batch) for batch in batches],
            "batches_sha256": sha256(output / "batches.json"), "input_sha256": {**hashes,
                "results/h3_experiment_plan.json": sha256(ROOT / "results/h3_experiment_plan.json"),
                "code/script/train_base.py": sha256(Path(__file__).resolve())},
            "dev_used_for_gradient": False, "public_used": False, "precision": "float32",
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "sentence_transformers": sentence_transformers.__version__},
            "gpu": torch.cuda.get_device_name(0), "deterministic_algorithms": True,
            "negative_policy": "same W08 batches; in-batch positives, same-PID/text excluded; no mined negatives"})
        write_json(output / "training_manifest.json", report)
        loss_fn = losses.MultipleNegativesRankingLoss(model, scale=config["scale"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        warmup = math.ceil(len(batches) * config["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, len(batches))
        report["warmup_steps"] = warmup
        tracked = next(p for p in model.parameters() if p.requires_grad)
        before = tracked.detach().clone()
        torch.cuda.reset_peak_memory_stats()
        step_started = time.perf_counter()
        model.train()
        with (output / "loss.jsonl").open("x", encoding="utf-8", buffering=1) as handle:
            for step, batch in enumerate(batches, 1):
                optimizer.zero_grad(set_to_none=True)
                texts = [[query_texts[i] for i in batch], [products[rows[i]["gold_pid"]] for i in batch]]
                features = [{k: v.to(model.device) for k, v in model.tokenize(part).items()} for part in texts]
                loss = loss_fn(features, None)
                if not torch.isfinite(loss):
                    raise ValueError(f"non-finite loss at step {step}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
                if not torch.isfinite(norm) or norm.item() <= 0:
                    raise ValueError(f"invalid gradient at step {step}")
                lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                row = {"step": step, "loss": loss.item(), "grad_norm_before_clip": norm.item(),
                    "learning_rate": lr, "batch_size": len(batch), "in_batch_negatives_per_query": len(batch) - 1}
                handle.write(json.dumps(row) + "\n")
                if step == 1 or step % 20 == 0 or step == len(batches):
                    print(json.dumps(row), flush=True)
        torch.cuda.synchronize()
        report["training_wall_seconds"] = time.perf_counter() - step_started
        report["tracked_parameter_max_abs_change"] = (tracked.detach() - before).abs().max().item()
        if report["tracked_parameter_max_abs_change"] <= 0:
            raise ValueError("model weights did not change")
        del before
        model.eval()
        probes = [query_texts[batches[0][0]], products[rows[batches[0][0]]["gold_pid"]]]
        expected = model.encode(probes, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        checkpoint = output / "checkpoint"
        model.save(str(checkpoint), create_model_card=False, safe_serialization=True)
        del loss_fn, optimizer, scheduler, features, loss, model, tracked
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = SentenceTransformer(str(checkpoint), device="cuda", local_files_only=True, trust_remote_code=False)
        actual = reloaded.encode(probes, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        verify_checkpoint_shape(actual, expected, lock["dimension"], reloaded.max_seq_length)
        report.update({"status": "completed", "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
            "checkpoint_files": {p.relative_to(checkpoint).as_posix(): sha256(p) for p in sorted(checkpoint.rglob("*")) if p.is_file()},
            "offline_reload_max_abs_difference": float(np.max(np.abs(actual - expected))),
            "loss_sha256": sha256(output / "loss.jsonl"), "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "wall_seconds": time.perf_counter() - started})
        write_json(output / "training_manifest.json", report)
        print(json.dumps({key: report[key] for key in ["status", "mode", "optimizer_steps", "training_wall_seconds",
            "peak_cuda_allocated_bytes", "offline_reload_max_abs_difference"]}), flush=True)
    except BaseException as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "wall_seconds": time.perf_counter() - started})
        write_json(output / "training_manifest.json", report)
        raise


if __name__ == "__main__":
    main()
