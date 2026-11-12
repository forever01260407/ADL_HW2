#!/usr/bin/env python3
"""Count actual BGE tokens over the full train product pool and fixed dev."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval_utils import load_products, load_retrieval_queries
from smoke_judge import length_summary


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    lock = json.loads((ROOT / "code/script/configs/models.lock.json").read_text())["embedding"]
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / lock["path"]), local_files_only=True)
    products = load_products(str(ROOT / "data/train/products.jsonl"))
    queries = load_retrieval_queries(str(ROOT / "data/derived/dev.jsonl"))
    data = {"products": list(products.items()), "dev_queries_with_instruction": [
        (q["qid"], "Represent this sentence for searching relevant passages: " + q["query"]) for q in queries]}
    report = {"model": lock, "policy": "product head truncation at 512 tokens; query raw text if it fits, otherwise head/tail",
              "source": {"products": "data/train/products.jsonl", "queries": "data/derived/dev.jsonl"}}
    for name, rows in data.items():
        lengths = []
        for start in range(0, len(rows), 128):
            ids = tokenizer([text for _, text in rows[start:start + 128]], truncation=False)["input_ids"]
            lengths.extend(len(seq) for seq in ids)
        report[name] = {**length_summary(lengths, 512),
                        "truncated_fraction": sum(n > 512 for n in lengths) / len(lengths),
                        "first_10_over_limit_ids": [rows[i][0] for i, n in enumerate(lengths) if n > 512][:10]}
    (ROOT / "results/w06_token_statistics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
