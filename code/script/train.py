#!/usr/bin/env python3
"""One offline BGE fine-tuning experiment; no dev gradients or automatic promotion."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyze_results import sha256
from eval_utils import load_eval_queries, load_products
from prepare_splits import normalize

INSTRUCTION = "Represent this sentence for searching relevant passages: "


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_query(raw, tokenizer):
    """Match the formal student block, including its head/tail token budget."""
    text = INSTRUCTION + raw
    if len(tokenizer.encode(text, add_special_tokens=True)) <= 512:
        return text
    ids = tokenizer.encode(raw, add_special_tokens=False)
    budget = 512 - tokenizer.num_special_tokens_to_add(pair=False) - len(
        tokenizer.encode(INSTRUCTION, add_special_tokens=False))
    while budget > 0:
        head = min(128, budget // 2)
        content = tokenizer.decode(ids[:head] + ids[-(budget - head):], skip_special_tokens=True)
        text = INSTRUCTION + content
        overflow = len(tokenizer.encode(text, add_special_tokens=True)) - 512
        if overflow <= 0:
            return text
        budget -= max(1, overflow)
    raise ValueError("query cannot fit the encoder token budget")


def validate_partitions(train, dev, products, manifest):
    for name, rows in [("train", train), ("dev", dev)]:
        expected = manifest["partitions"][name]
        qids = [row["qid"] for row in rows]
        if len(qids) != expected["count"] or qids != expected["qids"]:
            raise ValueError(f"{name} qids/order differ from split manifest")
        if len(set(qids)) != len(qids) or any(r["gold_pid"] not in products for r in rows):
            raise ValueError(f"invalid {name} queries/targets")
    for field in ["qid", "gold_pid"]:
        if {r[field] for r in train} & {r[field] for r in dev}:
            raise ValueError(f"train/dev {field} leakage")
    if ({normalize(products[r["gold_pid"]]) for r in train} &
            {normalize(products[r["gold_pid"]]) for r in dev}):
        raise ValueError("train/dev normalized product text leakage")
    if {normalize(r["query"]) for r in train} & {normalize(r["query"]) for r in dev}:
        raise ValueError("train/dev normalized dialogue leakage")


def make_batches(rows, products, batch_size, seed):
    """Use every training row once, without same-PID/text in-batch negatives."""
    if batch_size < 2:
        raise ValueError("contrastive training requires a batch size of at least two")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    pending, batches = deque(order), []
    identities = [(r["gold_pid"], normalize(products[r["gold_pid"]])) for r in rows]

    def compatible(index, batch):
        pid, text = identities[index]
        return all(pid != identities[i][0] and text != identities[i][1] for i in batch)

    while pending:
        batch = []
        for _ in range(len(pending)):
            index = pending.popleft()
            if len(batch) < batch_size and compatible(index, batch):
                batch.append(index)
            else:
                pending.append(index)
        if len(batch) < 2:
            # Attach singleton to a compatible batch or borrow a distinct item.
            index = batch[0]
            for previous in reversed(batches):
                if len(previous) < batch_size and compatible(index, previous):
                    previous.append(index)
                    batch = []
                    break
                if len(previous) >= 3:
                    moved = next((i for i in previous if compatible(i, batch)), None)
                    if moved is not None:
                        previous.remove(moved)
                        batch.append(moved)
                        break
            if batch and len(batch) < 2:
                raise ValueError("cannot form valid batches without false duplicate negatives")
        if batch:
            batches.append(batch)
    if sorted(i for batch in batches for i in batch) != list(range(len(rows))):
        raise ValueError("training batches lost or repeated rows")
    return batches


def load_data():
    manifest_path = ROOT / "code/script/configs/split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {"train": ROOT / "data/derived/train.jsonl", "dev": ROOT / "data/derived/dev.jsonl",
             "products": ROOT / "data/train/products.jsonl"}
    for name in ["train", "dev"]:
        if sha256(paths[name]) != manifest["partitions"][name]["sha256"]:
            raise ValueError(f"changed {name} split")
    for path, expected in manifest["source_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"changed source: {path}")
    train, dev = (load_eval_queries(paths[name]) for name in ["train", "dev"])
    products = load_products(paths["products"])
    if len(products) != manifest["candidate_pool"]["count"]:
        raise ValueError("candidate pool count changed")
    validate_partitions(train, dev, products, manifest)
    return train, products, {str(p.relative_to(ROOT)): sha256(p) for p in [manifest_path, *paths.values()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "code/script/configs/w08_train.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "train"], default="train")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if (config["epochs"] != 1 or config["batch_size"] not in [8, 16] or
            config["max_seq_length"] != 512 or config["precision"] != "float32" or
            config["loss"] != "MultipleNegativesRankingLoss" or config["query_instruction"] != INSTRUCTION):
        raise ValueError("configuration differs from the bounded W08 design")
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / "artifacts"):
        raise ValueError("write training artifacts inside this project's artifacts directory")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    started = time.perf_counter()
    report = {"status": "running", "mode": args.mode, "config": config,
              "started_utc": datetime.now(timezone.utc).isoformat(), "official_hardware_validation": False}
    write_json(args.output / "training_manifest.json", report)
    try:
        import resource
        import numpy as np
        import torch
        import sentence_transformers
        import transformers
        from sentence_transformers import SentenceTransformer, losses
        from transformers import get_linear_schedule_with_warmup

        if not torch.cuda.is_available():
            raise ValueError("W08 GPU training requires CUDA")
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        train, products, hashes = load_data()
        lock_path = ROOT / "code/script/configs/models.lock.json"
        model_lock = json.loads(lock_path.read_text())["embedding"]
        download_path = ROOT / "results/embedding_download.json"
        download = json.loads(download_path.read_text())
        for name, details in download["files"].items():
            if sha256(ROOT / download["path"] / name) != details["sha256"]:
                raise ValueError(f"pretrained model changed: {name}")
        model = SentenceTransformer(str(ROOT / model_lock["path"]), device="cuda",
                                    local_files_only=True, trust_remote_code=False)
        model.max_seq_length = 512
        model.float()
        batches = make_batches(train, products, config["batch_size"], config["seed"])
        if args.mode == "smoke":
            batches = [batches[0]] * 4  # engineering sanity, not a second quality experiment
        query_texts = {i: prepare_query(train[i]["query"], model.tokenizer) for b in batches for i in b}
        batch_qids = [[train[i]["qid"] for i in b] for b in batches]
        write_json(args.output / "batches.json", batch_qids)
        hashes.update({str(p.relative_to(ROOT)): sha256(p) for p in
                       [args.config.resolve(), Path(__file__).resolve(), lock_path, download_path,
                        ROOT / "retrieval_utils.py", ROOT / "eval_utils.py", ROOT / "code/script/prepare_splits.py"]})
        report.update({"input_sha256": hashes, "pretrained_model": model_lock,
            "train_count": len(train), "candidate_pool_count": len(products),
            "gradient_qids": sorted({qid for batch in batch_qids for qid in batch}),
            "optimizer_steps": len(batches), "batch_sizes": [len(b) for b in batches],
            "batches_sha256": sha256(args.output / "batches.json"),
            "versions": {"python": sys.version, "torch": torch.__version__,
                         "sentence_transformers": sentence_transformers.__version__, "transformers": transformers.__version__},
            "gpu": torch.cuda.get_device_name(0), "precision": "float32", "deterministic_algorithms": True,
            "negative_policy": "other training positives in batch; exclude identical PID/normalized product text; alternatives can remain false negatives",
            "dev_used_for_gradient": False, "public_used": False})
        write_json(args.output / "training_manifest.json", report)
        loss_fn = losses.MultipleNegativesRankingLoss(model, scale=config["scale"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        warmup = math.ceil(len(batches) * config["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, len(batches))
        report["warmup_steps"] = warmup
        # A small parameter snapshot verifies that actual weights, not just loss, change.
        tracked = next(p for p in model.parameters() if p.requires_grad)
        before = tracked.detach().clone()
        torch.cuda.reset_peak_memory_stats()
        step_started = time.perf_counter()
        model.train()
        with (args.output / "loss.jsonl").open("x", encoding="utf-8", buffering=1) as handle:
            for step, batch in enumerate(batches, 1):
                optimizer.zero_grad(set_to_none=True)
                texts = [[query_texts[i] for i in batch], [products[train[i]["gold_pid"]] for i in batch]]
                features = [{k: v.to(model.device) for k, v in model.tokenize(part).items()} for part in texts]
                loss = loss_fn(features, None)
                if not torch.isfinite(loss):
                    raise ValueError(f"non-finite loss at step {step}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
                if not torch.isfinite(norm) or norm.item() <= 0:
                    raise ValueError(f"invalid or zero gradient at step {step}")
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
        probe_texts = [query_texts[batches[0][0]], products[train[batches[0][0]]["gold_pid"]]]
        expected = model.encode(probe_texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        checkpoint = args.output / "checkpoint"
        model.save(str(checkpoint), create_model_card=False, safe_serialization=True)
        del loss_fn, optimizer, scheduler, features, loss, model, tracked
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = SentenceTransformer(str(checkpoint), device="cuda", local_files_only=True, trust_remote_code=False)
        actual = reloaded.encode(probe_texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        if reloaded.max_seq_length != 512 or actual.shape != (2, 384) or not np.isfinite(actual).all():
            raise ValueError("invalid reloaded embedding checkpoint")
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        report.update({"status": "completed", "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_files": {str(p.relative_to(checkpoint)): sha256(p) for p in sorted(checkpoint.rglob("*")) if p.is_file()},
            "offline_reload_max_abs_difference": float(np.max(np.abs(actual - expected))),
            "loss_sha256": sha256(args.output / "loss.jsonl"),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "wall_seconds": time.perf_counter() - started})
        write_json(args.output / "training_manifest.json", report)
        print(json.dumps({k: report[k] for k in ["status", "mode", "optimizer_steps", "training_wall_seconds",
              "peak_cuda_allocated_bytes", "offline_reload_max_abs_difference"]}), flush=True)
    except BaseException as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                       "wall_seconds": time.perf_counter() - started})
        write_json(args.output / "training_manifest.json", report)
        raise


if __name__ == "__main__":
    main()
