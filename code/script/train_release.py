#!/usr/bin/env python3
"""Portable two-stage reproduction of H3 -> H5; never promotes a checkpoint.

Uses the original sampler/query/loss helpers and frozen Train-only negatives.
Stage-one non-reentrant checkpointing is a memory implementation change.
Public data/results are never read. Dev is read only for split integrity.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import resource
import time
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
    TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="4")
from train import ROOT, load_data, make_batches, prepare_query, write_json
from analyze_results import sha256
from hard_negative_data import batch_negatives
from hard_negative_loss import own_negative_loss


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_stage(stage, source, output, rows, products, selected, config, smoke):
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer, losses
    from transformers import get_linear_schedule_with_warmup
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for FP32 training reproduction")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "running", "stage": stage, "source": str(source), "smoke": smoke,
        "public_used": False, "dev_used_for_gradient": False, "dev_used_for_mining": False,
        "gradient_checkpointing": True, "precision": "float32", "seed": 42,
        "official_hardware_validation": False}
    write_json(output / "training_manifest.json", report)
    try:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(min(1.0,
            8 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
        torch.cuda.reset_peak_memory_stats()
        model = SentenceTransformer(str(source), device="cuda", local_files_only=True, trust_remote_code=False)
        model.float()
        model.max_seq_length = 512
        model[0].auto_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if model.get_sentence_embedding_dimension() != 768:
            raise ValueError("Expected BGE-base 768-dimensional encoder")
        batches = make_batches(rows, products, 16, 42)
        if smoke:
            batches = [batches[0]] * 4
        query_texts = {i: prepare_query(rows[i]["query"], model.tokenizer) for b in batches for i in b}
        write_json(output / "batches.json", [[rows[i]["qid"] for i in b] for b in batches])
        lr = 2e-5 if stage == "base" else 5e-6
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
        warmup = math.ceil(len(batches) * .1)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, len(batches))
        loss_fn = losses.MultipleNegativesRankingLoss(model, scale=20) if stage == "base" else None
        tracked = next(p for p in model.parameters() if p.requires_grad)
        before = tracked.detach().clone()
        model.train()
        exposures = masked = 0
        step_started = time.perf_counter()
        with (output / "loss.jsonl").open("x", encoding="utf-8", buffering=1) as handle:
            for step, batch in enumerate(batches, 1):
                optimizer.zero_grad(set_to_none=True)
                part = [rows[i] for i in batch]
                streams = [[query_texts[i] for i in batch], [products[r["gold_pid"]] for r in part]]
                negative_rows, negative_texts, duplicate_count = ([], [], 0)
                if stage == "hard":
                    negative_rows, negative_texts, duplicate_count = batch_negatives(part, products, selected)
                    if negative_texts:
                        streams.append(negative_texts)
                features = [{k: v.to(model.device) for k,v in model.tokenize(texts).items()} for texts in streams]
                if stage == "base":
                    loss = loss_fn(features, None)
                else:
                    vectors = [model(f)["sentence_embedding"] for f in features]
                    loss = own_negative_loss(vectors[0], vectors[1], vectors[2] if negative_texts else None,
                        negative_rows, 20)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                if norm.item() <= 0:
                    raise ValueError("zero gradient")
                current_lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                exposures += len(negative_rows)
                masked += duplicate_count
                row = {"step": step, "loss": loss.item(), "batch_size": len(batch),
                    "learning_rate": current_lr, "own_verified_negatives": len(negative_rows),
                    "grad_norm_before_clip": norm.item()}
                handle.write(json.dumps(row)+"\n")
                if step == 1 or step % 40 == 0 or step == len(batches):
                    print(json.dumps({"stage": stage, **row}), flush=True)
        torch.cuda.synchronize()
        report.update(train_count=len({i for b in batches for i in b}), optimizer_steps=len(batches),
            training_wall_seconds=time.perf_counter()-step_started, warmup_steps=warmup,
            learning_rate=lr, trained_own_negative_exposures=exposures,
            duplicate_stream_negatives_masked=masked,
            tracked_parameter_max_abs_change=(tracked.detach()-before).abs().max().item())
        if report["tracked_parameter_max_abs_change"] <= 0:
            raise ValueError("weights did not update")
        model.eval()
        probes = [query_texts[batches[0][0]], products[rows[batches[0][0]]["gold_pid"]]]
        expected = model.encode(probes, normalize_embeddings=True, show_progress_bar=False)
        checkpoint = output / "checkpoint"
        model.save(str(checkpoint), safe_serialization=True, create_model_card=False)
        del model, loss_fn, optimizer, scheduler, tracked, before, features, loss
        if stage == "hard":
            del vectors
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = SentenceTransformer(str(checkpoint), device="cuda", local_files_only=True, trust_remote_code=False)
        actual = reloaded.encode(probes, normalize_embeddings=True, show_progress_bar=False)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        report.update(status="completed", checkpoint=str(checkpoint.relative_to(ROOT)),
            offline_reload_max_abs_difference=float(np.max(np.abs(actual-expected))),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            wall_seconds=time.perf_counter()-started,
            checkpoint_files={p.relative_to(checkpoint).as_posix():sha256(p) for p in sorted(checkpoint.rglob('*')) if p.is_file()},
            input_sha256={"code/script/train_release.py":sha256(Path(__file__)),
                "code/script/configs/release_negatives.json":sha256(ROOT/'code/script/configs/release_negatives.json')})
        if max(report['peak_cuda_allocated_bytes'],report['peak_cuda_reserved_bytes'])>8*2**30:
            raise ValueError("allocator budget exceeded")
        write_json(output / "training_manifest.json",report)
        print(json.dumps({k:report[k] for k in ['status','stage','train_count','optimizer_steps',
            'training_wall_seconds','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','trained_own_negative_exposures']}),flush=True)
        del reloaded
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output/'training_manifest.json',report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all','base','hard'],default='all')
    parser.add_argument('--output-root',type=Path,default=ROOT/'artifacts/reproduced')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    output=args.output_root.resolve()
    if not output.is_relative_to(ROOT/'artifacts'):
        parser.error('output must remain inside this project artifacts/')
    config=read(ROOT/'code/script/configs/release_assets.json')
    rows,products,_=load_data()
    neg_path=ROOT/'code/script/configs/release_negatives.json'
    if sha256(neg_path)!=config['training_negatives_sha256']:
        raise ValueError('fixed train-only negatives changed')
    selected=read(neg_path)['negatives']
    by_qid={r['qid']:r for r in rows}
    for qid,n in selected.items():
        if qid not in by_qid or n['gold_pid']!=by_qid[qid]['gold_pid'] or n['candidate_pid'] not in products:
            raise ValueError('negative training boundary mismatch')
    base=config['auxiliary_models']['base']
    if args.stage in ['all','base']:
        for name,h in base['files'].items():
            if sha256(ROOT/base['path']/name)!=h:
                raise ValueError(f'pretrained base asset differs: {name}')
        run_stage('base',ROOT/base['path'],output/'base',rows,products,{},config,args.smoke)
    if args.stage in ['all','hard']:
        run_stage('hard',output/'base/checkpoint',output/'hard',rows,products,selected,config,args.smoke)


if __name__=='__main__':
    main()
