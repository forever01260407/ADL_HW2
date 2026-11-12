#!/usr/bin/env python3
"""Summarize explicitly subjective Q2 review separately from measured diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_report import ROOT, sha256, write_json


def summarize(review, outputs):
    expected = [(r["qid"], r["prompt_id"]) for r in outputs]
    actual = [(r["qid"], r["prompt_id"]) for r in review["records"]]
    if actual != expected or len(set(actual)) != len(actual):
        raise ValueError("review must cover each generated output exactly once in frozen order")
    dimensions = review["dimensions"]
    for row in review["records"]:
        if (len(row["scores"]) != len(dimensions) or
                any(type(score) is not int or not 0 <= score <= 2 for score in row["scores"]) or
                not row["assessment"].strip()):
            raise ValueError("invalid subjective score or missing assessment")
    by_prompt = {}
    for prompt in dict.fromkeys(r["prompt_id"] for r in outputs):
        ratings = [r for r in review["records"] if r["prompt_id"] == prompt]
        rows = [r for r in outputs if r["prompt_id"] == prompt]
        by_prompt[prompt] = {
            "samples": len(ratings),
            "subjective_mean_scores": {key: sum(r["scores"][i] for r in ratings) / len(ratings)
                                       for i, key in enumerate(dimensions)},
            "subjective_mean_total_out_of_10": sum(sum(r["scores"]) for r in ratings) / len(ratings),
            "measured_diagnostics": {
                "outputs_mentioning_any_pid": sum(bool(r["diagnostics"]["mentioned_pids"]) for r in rows),
                "outputs_with_requested_pid_format": sum(bool(r["diagnostics"]["formatted_pids"]) for r in rows),
                "outputs_with_out_of_evidence_pid": sum(bool(r["diagnostics"]["out_of_evidence_pids"]) for r in rows),
                "over_120_words": sum(r["diagnostics"]["word_count"] > 120 for r in rows),
                "length_capped": sum(r["stop_reason"] == "length" for r in rows)}}
    return {"kind": "assistant_subjective_review_not_official_quality_score", "reviewer": review["reviewer"],
            "outputs_reviewed": len(outputs), "dimensions": dimensions, "by_prompt": by_prompt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "results/w09_q2")
    args = parser.parse_args()
    directory = args.directory
    review = json.loads((directory / "review.json").read_text(encoding="utf-8"))
    for name, expected in review["source_sha256"].items():
        if sha256(directory / name) != expected:
            raise ValueError("review was written for different raw outputs/evidence")
    outputs = [json.loads(line) for line in (directory / "outputs.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = summarize(review, outputs)
    summary["input_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in
        [directory / "review.json", directory / "outputs.jsonl", directory / "plan.json", Path(__file__).resolve()]}
    path = directory / "review_summary.json"
    if path.exists():
        raise ValueError("review summary already exists; preserve recorded analysis")
    write_json(path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
