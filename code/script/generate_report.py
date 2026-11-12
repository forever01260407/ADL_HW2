#!/usr/bin/env python3
"""Q2: freeze shared evidence, then generate the fixed five-by-three grid offline.

This module never loads gold labels or retrieval Judge outputs. Its name refers
to the assignment's generation analysis, not automatic authorship of report.pdf.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyze_results import sha256
from eval_utils import load_products, load_retrieval_queries
from evaluate import load_submission
from smoke_retrieval import PROTECTED


def json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def data_json(value):
    # Preserve data through JSON decoding while preventing literal chat-control tags.
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")


def render_request(tokenizer, query, evidence, prompts, variant):
    user = (variant["instruction"] + "\n\nCONVERSATION_DATA_JSON:\n" + data_json(query) +
            "\n\nRETRIEVED_PRODUCT_DATA_JSON:\n" + data_json(evidence))
    messages = [{"role": "system", "content": prompts["system"]}, {"role": "user", "content": user}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    return messages, rendered, ids


def build_context(tokenizer, query, pids, products, prompts, config):
    if len(pids) != config["top_n"] or len(set(pids)) != len(pids):
        raise ValueError("context requires three distinct ranked products")
    if any(pid not in products for pid in pids):
        raise ValueError("context references an unknown product")
    originals = {pid: tokenizer.encode(products[pid], add_special_tokens=False) for pid in pids}
    cap = config["product_token_cap"]
    while cap > 0:
        evidence = [{"rank": rank, "pid": pid, "text": products[pid] if len(originals[pid]) <= cap else
                     tokenizer.decode(originals[pid][:cap], skip_special_tokens=True)} for rank, pid in enumerate(pids, 1)]
        rendered = [render_request(tokenizer, query, evidence, prompts, variant) for variant in prompts["prompts"]]
        overflow = max(len(item[2]) for item in rendered) - config["input_token_limit"]
        if overflow <= 0:
            diagnostics = [{"pid": pid, "original_tokens": len(originals[pid]),
                "supplied_tokens": len(tokenizer.encode(row["text"], add_special_tokens=False)),
                "truncated": row["text"] != products[pid]} for pid, row in zip(pids, evidence)]
            return evidence, rendered, {"shared_product_token_cap": cap, "products": diagnostics}
        cap -= max(1, math.ceil(overflow / len(pids)))
    raise ValueError("conversation/instructions cannot fit without removing evidence; do not silently truncate")


def load_samples(path, expected_count):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    allowed = {"dialogue_id", "conversation", "selection_tag"}
    if len(rows) != expected_count or any(set(row) - allowed for row in rows):
        raise ValueError("generation samples must have the fixed count and no target/gold/extra fields")
    queries = load_retrieval_queries(path)
    if any(turn.get("role") not in {"user", "assistant"} for row in rows for turn in row["conversation"]):
        raise ValueError("unsupported conversation role")
    return [(row, query) for row, query in zip(rows, queries)]


def output_diagnostics(text, evidence):
    formatted = re.findall(r"\[PID:\s*([^\]\s]+)\s*\]", text)
    mentioned = set(formatted) | set(re.findall(r"\bB[A-Z0-9]{9}\b", text))
    return {"formatted_pids": formatted, "mentioned_pids": sorted(mentioned),
            "out_of_evidence_pids": sorted(mentioned - {row["pid"] for row in evidence}),
            "empty_output": not bool(text.strip()), "word_count": len(text.split())}


def validate_completed(records, requests):
    if len(records) > len(requests):
        raise ValueError("too many generated records")
    for row, request in zip(records, requests):
        for key in ["qid", "prompt_id", "prompt_sha256", "evidence_sha256"]:
            if row[key] != request[key]:
                raise ValueError("generated records do not match the frozen request prefix")


def verify_inputs(manifest):
    for name, expected in manifest["input_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"generation input/source changed: {name}")
    model_path = ROOT / manifest["model"]["path"]
    for name, expected in manifest["model_files"].items():
        if sha256(model_path / name) != expected:
            raise ValueError(f"generation model changed: {name}")


def prepare(config_path, output):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config["sample_count"] != 5 or config["top_n"] != 3 or
            config["input_token_limit"] + config["generation"]["max_new_tokens"] > config["total_token_limit"]):
        raise ValueError("configuration violates the W09 grid/token contract")
    for key in ["queries", "products", "rankings"]:
        if sha256(ROOT / config[key]) != config[key + "_sha256"]:
            raise ValueError(f"frozen W09 source differs: {key}")
    for name, expected in PROTECTED.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"official source changed: {name}")
    samples = load_samples(ROOT / config["queries"], 5)
    products = load_products(ROOT / config["products"])
    rankings_path = ROOT / config["rankings"]
    payload = json.loads(rankings_path.read_text())
    rankings = load_submission(rankings_path, [{"qid": r["qid"]} for r in payload["results"]], set(products), top_k=50)
    prompts = json.loads((ROOT / config["prompts"]).read_text(encoding="utf-8"))
    if [r["id"] for r in prompts["prompts"]] != ["G-P1", "G-P2", "G-P3"]:
        raise ValueError("expected the three fixed prompts")
    lock_path = ROOT / "code/script/configs/models.lock.json"
    model = json.loads(lock_path.read_text())["generator"]
    if model["repo_id"] != "Qwen/Qwen2.5-1.5B-Instruct":
        raise ValueError("Q2 requires the specified Qwen model")
    download_path = ROOT / "results/generator_download.json"
    download = json.loads(download_path.read_text())
    if download["path"] != model["path"] or download["revision"] != model["revision"]:
        raise ValueError("model download and lock disagree")
    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / model["path"]), local_files_only=True, trust_remote_code=False)
    model_config = AutoConfig.from_pretrained(str(ROOT / model["path"]), local_files_only=True, trust_remote_code=False)
    if config["total_token_limit"] > model_config.max_position_embeddings:
        raise ValueError("planned generation exceeds the model context limit")
    requests, contexts = [], []
    for raw, query in samples:
        if query["qid"] not in rankings:
            raise ValueError("fixed sample is missing from frozen rankings")
        evidence, rendered, diagnostics = build_context(tokenizer, query["query"], rankings[query["qid"]][:3], products, prompts, config)
        contexts.append({"qid": query["qid"], "query": query["query"], "selection_tag": raw["selection_tag"],
                         "message_count": len(raw["conversation"]), "evidence": evidence, **diagnostics})
        for variant, (messages, text, ids) in zip(prompts["prompts"], rendered):
            requests.append({"qid": query["qid"], "prompt_id": variant["id"], "prompt_label": variant["label"],
                "messages": messages, "rendered_prompt": text, "input_ids": ids, "input_tokens": len(ids),
                "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(), "evidence_sha256": json_hash(evidence),
                "evidence": evidence})
    output.mkdir(parents=True, exist_ok=False)
    bundle = {"experiment": config["experiment"], "contexts": contexts, "requests": requests}
    write_json(output / "plan.json", bundle)
    paths = [config_path.resolve(), ROOT / config["queries"], ROOT / config["products"], rankings_path,
             ROOT / config["prompts"], lock_path, download_path, Path(__file__).resolve(),
             ROOT / "retrieval_utils.py", *[ROOT / n for n in PROTECTED]]
    manifest = {"status": "prepared", "config": config, "model": model,
        "created_utc": datetime.now(timezone.utc).isoformat(), "planned_outputs": 15,
        "model_files": {name: details["sha256"] for name, details in download["files"].items()},
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
        "plan_sha256": sha256(output / "plan.json"), "official_hardware_validation": False,
        "generation_reads_gold_or_judge": False,
        "diagnostic_note": "PID/token checks are structural; they do not establish factual or recommendation quality"}
    verify_inputs(manifest)
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"status": "prepared", "requests": len(requests),
        "max_input_tokens": max(r["input_tokens"] for r in requests),
        "truncated_products": sum(p["truncated"] for c in contexts for p in c["products"])}), flush=True)


def generate(output, resume=False):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    verify_inputs(manifest)
    if sha256(output / "plan.json") != manifest["plan_sha256"]:
        raise ValueError("frozen prompt plan changed")
    bundle = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    path = output / "outputs.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    validate_completed(records, bundle["requests"])
    existing_count = len(records)
    if len(records) == 15 or (path.exists() and not resume):
        raise ValueError("outputs already exist; completed outputs are immutable; use --resume only for an interrupted prefix")
    settings = manifest["config"]["generation"]
    if settings != {"seed": 0, "batch_size": 1, "dtype": "bfloat16", "do_sample": False, "num_beams": 1, "max_new_tokens": 256}:
        raise ValueError("generation settings differ from the frozen W09 design")
    started = time.perf_counter()
    manifest["status"] = "running"
    manifest.pop("error", None)
    write_json(output / "manifest.json", manifest)
    try:
        import resource
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(settings["seed"])
        torch.cuda.manual_seed_all(settings["seed"])
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        model_path = str(ROOT / manifest["model"]["path"])
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
            local_files_only=True, trust_remote_code=False, low_cpu_mem_usage=True).to("cuda").eval()
        for key in ["temperature", "top_p", "top_k"]:
            setattr(model.generation_config, key, None)
        model.generation_config.do_sample = False
        eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else eos
        torch.cuda.reset_peak_memory_stats()
        with path.open("a" if resume else "x", encoding="utf-8", buffering=1) as handle:
            for request in bundle["requests"][len(records):]:
                actual_ids = tokenizer.encode(request["rendered_prompt"], add_special_tokens=False)
                if actual_ids != request["input_ids"]:
                    raise ValueError("tokenizer inputs differ from frozen plan")
                ids = torch.tensor([actual_ids], dtype=torch.long, device="cuda")
                torch.cuda.synchronize()
                tick = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                        do_sample=False, num_beams=1, max_new_tokens=256, pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=eos)
                torch.cuda.synchronize()
                new_ids = generated[0, ids.shape[1]:].tolist()
                text = tokenizer.decode(new_ids, skip_special_tokens=True)
                row = {k: request[k] for k in ["qid", "prompt_id", "prompt_sha256", "evidence_sha256", "input_tokens"]}
                row.update({"raw_output": text, "generated_ids": new_ids, "generated_tokens": len(new_ids),
                    "stop_reason": "eos" if new_ids and new_ids[-1] in eos_ids else
                                   ("length" if len(new_ids) == 256 else "other"),
                    "generation_seconds": time.perf_counter() - tick, "settings": settings,
                    "diagnostics": output_diagnostics(text, request["evidence"])})
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                records.append(row)
                print(json.dumps({"completed": len(records), "qid": row["qid"], "prompt": row["prompt_id"],
                                  "tokens": len(new_ids), "stop": row["stop_reason"]}), flush=True)
        manifest.update({"status": "completed", "completed_outputs": len(records),
            "outputs_sha256": sha256(path), "gpu": torch.cuda.get_device_name(0),
            "versions": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
            "eos_token_ids": eos_ids, "decoded_with_skip_special_tokens": True})
    except BaseException as exc:
        manifest.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        torch_module = sys.modules.get("torch")
        # Each process has its own resource peaks; do not sum CUDA peaks on resume.
        attempt = {"wall_seconds": time.perf_counter() - started, "started_after_outputs": existing_count,
                   "completed_outputs": len(records), "status": manifest["status"]}
        if "error" in manifest:
            attempt["error"] = manifest["error"]
        if torch_module is not None and torch_module.cuda.is_initialized():
            attempt.update({"peak_cuda_allocated_bytes": torch_module.cuda.max_memory_allocated(),
                            "peak_cuda_reserved_bytes": torch_module.cuda.max_memory_reserved()})
        if "resource" in sys.modules:
            attempt["peak_process_rss_bytes"] = sys.modules["resource"].getrusage(0).ru_maxrss * 1024
        manifest.setdefault("attempts", []).append(attempt)
        write_json(output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "generate"], required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "code/script/configs/w09_generation.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/w09_q2")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    if args.stage == "prepare":
        prepare(args.config, args.output)
    else:
        generate(args.output, args.resume)


if __name__ == "__main__":
    main()
