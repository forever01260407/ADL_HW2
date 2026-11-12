# ADL HW2: RAG

Use Python 3.12.9.

1. Download the dataset and place it under `data/` so that each split
   contains `dialogues.jsonl` and `products.jsonl`.

2. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Implement retrieval and reranking only inside
   `retrieval_utils.py`, then run `retrieval.py` to generate the ranked results:

   ```bash
   python retrieval.py --dialogues "dialogues_path" --products "products_path" --output "output_path"
   ```

4. Evaluate the results:

   ```bash
   python evaluate.py --retrieval-results "path_to_retrieval_result" --dialogues "dialogues_path" --products "products_path" --output "output_path"
   ```

## W08 development training

The current retrieval default is W08 (fine-tuned BGE-small + H1 BM25 + RRF),
selected after the complete dev Gold gate and one official Judge comparison.
The selected checkpoint is `artifacts/training/w08_epoch1/checkpoint`.
The assignment is not yet fully packaged. See `PROGRESS.md` for remaining work.

Development was verified in WSL Ubuntu 22.04, Python 3.12.12, torch 2.8.0+cu128,
sentence-transformers 5.1.0, transformers 4.56.1, NumPy 1.26.4 and CPU FAISS 1.12.0,
on an RTX 4090. This is not validation of the official Python 3.12.9 / RTX 3070.

Run from the project root in Linux, after preparing `data/train/` and the local
model. `prepare_model.py` uses the immutable revision in `models.lock.json` and
requires network access only for the initial download:

```bash
python code/script/prepare_splits.py --seed 42
python code/script/prepare_model.py --role embedding
python -m unittest discover -s code/script/tests -p 'test_*.py' -v
bash run.sh
```

For this workspace's existing environment use `HW2_PYTHON=.venv/bin/python bash
run.sh`. The frozen settings are in `code/script/configs/w08_train.json`: 3,864
training pairs, one epoch, actual batch 16, AdamW learning rate 2e-5, warmup 10%,
weight decay 0.01, FP32, cosine MultipleNegativesRankingLoss scale 20, seed 42.
Training excludes dev supervision and same-PID / identical-text in-batch negatives.

The default output is `artifacts/training/w08_epoch1/` and existing output folders
are refused. To reproduce without overwriting the recorded experiment:

```bash
bash run.sh --output artifacts/training/w08_reproduction
```

`checkpoint/` is an offline-loadable SentenceTransformer; `training_manifest.json`,
`batches.json` and `loss.jsonl` record the actual run. To evaluate the W08 checkpoint
explicitly, rebuild the product index with the existing official entrypoint:

```bash
HW2_MODEL_PATH=artifacts/training/w08_epoch1/checkpoint \
HW2_RETRIEVER=hybrid HW2_RERANK=0 HW2_RRF_CANDIDATES=100 \
HW2_BM25_QUERY_MODE=full HW2_BM25_LAST_WEIGHT=1 HW2_BM25_TITLE_WEIGHT=1 \
python retrieval.py --dialogues data/derived/dev.jsonl \
  --products data/train/products.jsonl --top-k 50 --output results/w08_manual_top50.json
```

Gold diagnostics and the gated official comparison are documented in
`W08_RESULTS.md`. Public data is reserved for the later frozen-system check.

## W09 Q2 generation analysis

`W09_RESULTS.md` contains the fixed five-sample, three-prompt comparison. The
retriever remains frozen at W08. All 15 raw outputs, token IDs, complete prompts,
shared top-three product evidence and analysis are retained in `results/w09_q2/`.
Structural audit success does not mean that the answers are factually correct;
the report documents unsupported claims, attribution errors and format failures.

The generator uses the locked local Qwen2.5-1.5B-Instruct, bf16, greedy decoding,
batch 1 and at most 256 new tokens. It does not read gold labels or Judge outputs.
The exact source hashes and settings are in `code/script/configs/w09_generation.json`;
all three prompt strings are in `code/script/configs/w09_prompts.json`.
Use the existing WSL Python, or replace `.venv/bin/python` with the matching runtime:

```bash
.venv/bin/python code/script/prepare_model.py --role generator
.venv/bin/python code/script/generate_report.py --stage prepare --output results/w09_reproduction
.venv/bin/python code/script/generate_report.py --stage generate --output results/w09_reproduction
.venv/bin/python code/script/audit_w09.py --directory results/w09_reproduction
```

The fixed W02 samples, train product pool and frozen W08 ranking file must be
present; mismatched hashes are refused. Existing output directories/complete
outputs are preserved. An interrupted generated prefix can be continued with
`--stage generate --resume` in the same directory after validation. A completed
grid is not regenerated. The recorded `review.json` is a non-blind Codex assistant
qualitative review, not human labels or official scores; new outputs need their
own review. `generate_report.py` does not create the final `report.pdf`.
