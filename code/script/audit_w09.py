#!/usr/bin/env python3
"""Audit Q2 provenance, token IDs and a complete fixed grid; export raw text."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from generate_report import (ROOT, build_context, json_hash, load_samples,
    output_diagnostics, sha256, validate_completed, verify_inputs, write_json)
from eval_utils import load_products


def fence(text):
    # Only the Markdown wrapper changes; the enclosed model text is unchanged.
    longest = max((len(part) for part in re.findall(r"`+", text)), default=0)
    mark = "`" * max(3, longest + 1)
    return f"{mark}text\n{text}\n{mark}\n"


def audit(directory):
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or manifest["completed_outputs"] != 15:
        raise ValueError("generation grid is not complete")
    verify_inputs(manifest)
    for filename, key in [("plan.json", "plan_sha256"), ("outputs.jsonl", "outputs_sha256")]:
        if sha256(directory / filename) != manifest[key]:
            raise ValueError(f"changed saved generation artifact: {filename}")
    config = manifest["config"]
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    outputs = [json.loads(line) for line in (directory / "outputs.jsonl").read_text(encoding="utf-8").splitlines()]
    validate_completed(outputs, plan["requests"])
    if len(outputs) != 15 or len(plan["requests"]) != 15 or len(plan["contexts"]) != 5:
        raise ValueError("expected exactly fifteen outputs and requests")
    samples = load_samples(ROOT / config["queries"], 5)
    products = load_products(ROOT / config["products"])
    rankings = {r["qid"]: r["ranked_pids"] for r in json.loads((ROOT / config["rankings"]).read_text())["results"]}
    prompts = json.loads((ROOT / config["prompts"]).read_text())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / manifest["model"]["path"]), local_files_only=True, trust_remote_code=False)
    expected_grid = [(q["qid"], variant["id"]) for _, q in samples for variant in prompts["prompts"]]
    if [(r["qid"], r["prompt_id"]) for r in outputs] != expected_grid:
        raise ValueError("sample/prompt order differs from the frozen grid")
    for index, ((raw, query), context) in enumerate(zip(samples, plan["contexts"])):
        evidence, rendered, diag = build_context(tokenizer, query["query"], rankings[query["qid"]][:3], products, prompts, config)
        expected_context = {"qid": query["qid"], "query": query["query"], "selection_tag": raw["selection_tag"],
                            "message_count": len(raw["conversation"]), "evidence": evidence, **diag}
        if context != expected_context:
            raise ValueError("saved evidence differs from the frozen top-three products")
        for request, (messages, text, ids) in zip(plan["requests"][index * 3:index * 3 + 3], rendered):
            if (request["messages"] != messages or request["rendered_prompt"] != text or
                request["input_ids"] != ids or request["input_tokens"] != len(ids) or
                request["evidence"] != evidence or request["evidence_sha256"] != json_hash(evidence) or
                request["prompt_sha256"] != hashlib.sha256(text.encode()).hexdigest()):
                raise ValueError("saved prompt/token IDs differ from independently rebuilt request")
            if len(ids) > config["input_token_limit"]:
                raise ValueError("input token budget exceeded")
    for output, request in zip(outputs, plan["requests"]):
        ids = output["generated_ids"]
        stop = "eos" if ids and ids[-1] in manifest["eos_token_ids"] else ("length" if len(ids) == 256 else "other")
        if (tokenizer.decode(ids, skip_special_tokens=True) != output["raw_output"] or
                len(ids) != output["generated_tokens"] or output["settings"] != config["generation"] or
                output["stop_reason"] != stop or output["diagnostics"] != output_diagnostics(output["raw_output"], request["evidence"])):
            raise ValueError("raw text/output metadata differ from saved generated token IDs")
        if len(ids) > 256 or request["input_tokens"] + len(ids) > config["total_token_limit"]:
            raise ValueError("generation length exceeds frozen limits")
    stats = {}
    for variant in prompts["prompts"]:
        rows = [r for r in outputs if r["prompt_id"] == variant["id"]]
        stats[variant["id"]] = {"outputs": len(rows), "stop_reasons": dict(Counter(r["stop_reason"] for r in rows)),
            "total_generation_seconds": sum(r["generation_seconds"] for r in rows),
            "generated_tokens": sum(r["generated_tokens"] for r in rows),
            "out_of_evidence_pid_outputs": sum(bool(r["diagnostics"]["out_of_evidence_pids"]) for r in rows),
            "empty_outputs": sum(r["diagnostics"]["empty_output"] for r in rows),
            "over_120_words": sum(r["diagnostics"]["word_count"] > 120 for r in rows)}
    report = {"status": "passed", "kind": "structural_generation_audit_not_quality_score", "sample_count": 5,
        "prompt_count": 3, "outputs": 15, "frozen_grid_verified": True, "shared_top3_evidence_verified": True,
        "raw_decoding_verified": True, "max_input_tokens": max(r["input_tokens"] for r in plan["requests"]),
        "max_generated_tokens": max(r["generated_tokens"] for r in outputs),
        "truncated_evidence_products": sum(p["truncated"] for c in plan["contexts"] for p in c["products"]),
        "by_prompt": stats, "attempts": manifest["attempts"], "official_hardware_validation": False,
        "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
            [directory / "manifest.json", directory / "plan.json", directory / "outputs.jsonl", Path(__file__).resolve()]}}
    return report, plan, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "results/w09_q2")
    args = parser.parse_args()
    report, plan, outputs = audit(args.directory)
    if (args.directory / "audit.json").exists():
        raise ValueError("audit already exists; preserve the original result")
    raw = ["# W09 原始生成回覆\n\n以下文字按保存的 token IDs 解碼，沒有潤飾；不代表回答正確。\n"]
    evidence_doc = ["# W09 固定商品證據\n\n商品文字是資料集提供的文案，不是本次驗證其功效的證據。\n"]
    for context in plan["contexts"]:
        raw += [f"\n## {context['qid']}\n", fence(context["query"]), "\n檢索順序：\n"]
        evidence_doc += [f"\n## {context['qid']}\n", fence(context["query"])]
        for item in context["evidence"]:
            raw += [f"- {item['rank']}. `{item['pid']}`：{item['text'].splitlines()[0]}\n"]
            evidence_doc += [f"\n### {item['rank']}. {item['pid']}\n", fence(item["text"])]
        for row in (r for r in outputs if r["qid"] == context["qid"]):
            raw += [f"\n### {row['prompt_id']}\n\n{row['generated_tokens']} tokens；停止原因 `{row['stop_reason']}`。\n\n", fence(row["raw_output"])]
    (args.directory / "raw_outputs.md").write_text("".join(raw), encoding="utf-8")
    (args.directory / "evidence.md").write_text("".join(evidence_doc), encoding="utf-8")
    write_json(args.directory / "audit.json", report)
    print(json.dumps({k: report[k] for k in ["status", "sample_count", "prompt_count", "outputs", "max_input_tokens",
                                           "max_generated_tokens", "truncated_evidence_products", "by_prompt"]}, indent=2))


if __name__ == "__main__":
    main()
