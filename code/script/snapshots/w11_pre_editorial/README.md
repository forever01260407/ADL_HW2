# ADL HW2 - r15922a14

Final system: **H5 continued BGE-base + BM25, weighted RRF 5:1**.
The tested final weights are downloaded to
`artifacts/training/h5_hard_negative/checkpoint`. Inference does not retrain.
`run.sh` reproduces both training stages into a separate directory.

## 1. Environment

The assignment specifies Ubuntu 22.04, Python 3.12.9, 32 GB RAM and RTX 3070
8 GB. Development used WSL Ubuntu 22.04, Python 3.12.12 and RTX 4090 24 GB.
Real RTX 3070 / Python 3.12.9 validation has not been performed.

The training/inference versions used here are torch 2.8.0+cu128,
transformers 4.56.1, sentence-transformers 5.1.0, NumPy 1.26.4 and CPU
FAISS 1.12.0. Use the supplied course environment first. If setting up a
new environment yourself, install `requirements.txt` with a compatible CUDA
PyTorch build. Do not run installation inside `download.sh`.

```bash
python --version
python -m pip install -r requirements.txt  # environment setup only, if needed
```

The implementation uses CPU FAISS `IndexFlatIP`; GPU FAISS is unnecessary.
An already available compatible `faiss` module also provides this interface.
The original unused `datasets` requirement is not needed by the final scripts.
Report-building dependencies are separate in
`code/script/configs/requirements-dev.txt`; graders need not install them.

## 2. Data

Place the official, unmodified training files at:

```text
data/train/dialogues.jsonl
data/train/products.jsonl
```

The supplied `data/derived/` contains the fixed grouped Train/Dev split and
five Q2 queries. It can be rebuilt from the original Train with:

```bash
python code/script/prepare_splits.py --seed 42
```

The frozen split has **3,864 Train / 966 Dev**, with all **50,000 products**
available as candidates. PID, normalized dialogue and target-text groups stay
in one partition. Loaders verify the source and split SHA-256 values.
Dev is read by training only for split integrity, never for gradients or mining.
Public/Private data is not needed by `run.sh`.

## 3. Download the submitted model

```bash
bash download.sh
```

This downloads the exact H5 archive and pinned raw BGE-base / Qwen assets,
verifies SHA-256, and extracts the H5 checkpoint. It performs no installation,
training or evaluation. Already matching files are verified and reused.
The downloader has a 30-minute transfer budget. Retain the final URL for at
least three weeks and do not replace assets after the assignment deadline.

For inference only, skip the auxiliary raw training / generation models:

```bash
bash download.sh --skip-auxiliary
```

If the archive was downloaded manually:

```bash
bash download.sh --final-model-archive /path/to/h5_final_r15922a14.zip --skip-auxiliary
```

The hosting URL is recorded in `code/script/configs/release_assets.json`.
If it is still null, the release is pending URL configuration; use the local
archive command only for local validation. An HTML sharing page is not a direct
download URL. Incorrect files fail hash verification.

## 4. Official inference and scoring

From this directory, without model/retriever environment overrides:

```bash
python retrieval.py --dialogues /path/to/dialogues.jsonl --products /path/to/products.jsonl --output ranked.json
python evaluate.py --retrieval-results ranked.json --dialogues /path/to/dialogues.jsonl --products /path/to/products.jsonl --model artifacts/models/Qwen2.5-1.5B-Instruct --batch-size 8 --max-new-tokens 12 --top-k 10 --device cuda --output graded.json
```

The official three source files are unchanged. All formal retrieval logic is
inside the permitted `retrieval_utils.py` student block. No persistent index,
precomputed Public answer or custom inference helper is required. The complete
input product pool is indexed on each run. Default top-k is ten.

The encoder uses full role transcripts with the BGE retrieval instruction,
512-token head/tail protection for long queries, one title+description vector
per product, FP32 normalized 768-dimensional embeddings and exact inner-product
search. BM25 reads full transcripts/products, k1=1.5, b=.75. Each route takes
100 candidates; weighted RRF uses 5/(60+dense_rank)+1/(60+bm25_rank).
No cross-encoder reranker is enabled. Avoid `HW2_*` overrides for the submitted
configuration. The measured inference batch size was 32; the default is 32.

## 5. Reproduce final training

First complete the downloads and provide the original Train files above.
Then run:

```bash
bash run.sh
# or choose a new separate output directory:
bash run.sh --output-root artifacts/reproduced_run2
```

Outputs are `artifacts/reproduced/base/checkpoint` and
`artifacts/reproduced/hard/checkpoint`, plus batch qids, per-step losses and
resource/reload manifests. Existing output directories are never overwritten.
The downloaded formal H5 checkpoint is not changed or automatically replaced.

The two stages use seed42, batch16, FP32, 512 tokens, cosine scale20,
AdamW weight decay .01, gradient clipping1, 10% warmup (25 steps), linear
learning-rate decay and one epoch / 242 steps / all3,864 queries each.
Stage 1 starts from pinned raw BGE-base with lr2e-5 and in-batch negatives.
Stage 2 continues from its output with lr5e-6 and fixed own-query hard negatives.
Both stages use non-reentrant gradient checkpointing and an 8 GiB PyTorch
allocator cap. That cap excludes CUDA context and other processes.

`release_negatives.json` is the frozen Train-only derived training set: 3,796
queries have one teacher-filtered negative, 68 retain only in-batch training.
The original H3 Train Dense top100, fixed rank5/15 candidate mining, all7,728
original Qwen judgments and filtering code are supplied as provenance. Running
`run.sh` reuses this fixed dataset; it does not rerun the teacher or read
Public labels. A repeated batch-positive product is masked, yielding3,795
actual own-negative exposures. Qwen judgments are pseudo-labels, not guaranteed
human irrelevance. Other queries' mined negatives are not broadcast.

W11 executed both complete stages, checked all batch qids and compared all
199 parameter tensors per stage: **maximum parameter difference0**, and both
`model.safetensors` SHA-256 values equal their original H3/H5 counterparts.
Non-reentrant checkpointing fixes the original H3 memory issue on this
4090 allocator test. This is measured reproducibility in the stated development
environment, not a guarantee across hardware/library versions.
See `results/w11_reproduction_comparison.json` for actual measurements.

Engineering smoke, if desired, uses a separate directory and is not a quality
experiment:

```bash
bash run.sh --smoke --output-root artifacts/reproduction_smoke
```

## 6. Q2 generation analysis

Q2 is the saved W09 experiment: **five fixed samples x three prompts =15**
raw answers using Qwen2.5-1.5B-Instruct. Its evidence is the historical W08
small+1:1 top3, explicitly not H5. Prompts/evidence/model settings were fixed
before generation; no target or Judge outputs were passed to the generator.

Files: `results/w09_q2/{plan,manifest,audit,review,review_summary}.json`,
`outputs.jsonl`, `raw_outputs.md`, `evidence.md`, plus
`code/script/configs/w09_{generation,prompts}.json`.
The report includes every answer verbatim and discusses hallucinations,
wrong specification attribution and unjustified refusal. Qualitative ratings
are non-blinded Codex-assistant judgments, not human labels or official scores.

To reproduce generation into a new directory using the saved W08 rankings:

```bash
python code/script/generate_report.py --stage prepare --output results/q2_reproduction
python code/script/generate_report.py --stage generate --output results/q2_reproduction
python code/script/audit_w09.py --directory results/q2_reproduction
```

This uses the downloaded Qwen, official Train products, fixed Q2 queries and
supplied `results/w08_default_dev_top50.json`. It does not require retraining
W08 or rerunning Public. Preserve the original W09 outputs.

## 7. Results and limitations

| Fixed configuration | Split | Graded Recall@10 | Graded MRR@10 |
|---|---|---:|---:|
| H3 base +5:1 | Dev966 |0.8706004141 |0.6582980052 |
| H5 continued base +5:1 | Dev966 |0.8933747412 |0.7071909198 |
| H4 original base +5:1 | Public1000 |0.9330000000 |0.7479154762 |
| H6 fixed H5 +5:1 | Public1000 |0.9520000000 |0.8024043651 |

Public has been seen in prior confirmations; no Public tuning, gradients,
mining or rollback was performed. It is not an untouched holdout. Gold metrics
are diagnostics and differ from Graded metrics, which accept alternatives.
The final Public run had9,096 comparisons, zero parse failures and no Judge
input truncation. Shared candidate judgments can differ across candidate
batches; no Judge rerun was used to select a higher score.

H6 retrieval cold time272.102s, allocated1.036/reserved1.309GiB, RSS2.190GiB,
under the development8GiB allocator cap. The original Judge was uncapped and
reserved9.096GiB; no8GB Judge validation is claimed. Private performance and
actual TA-environment validation are unknown. About27% of Train products
exceed the Dense512-token limit. Training improvement combines continuation
and curated negatives without a matched no-negative continuation control.

## 8. Checks, report and file roles

```bash
python -m unittest discover -s code/script/tests -p 'test_*.py'
bash -n run.sh download.sh
```

Some historical audits intentionally require development checkpoints or frozen
snapshots; they document prior experiments. Use `train_release.py`,
`download_release.py`, `run.sh` and `download.sh` for the portable final workflow.
Historical scripts, raw observations and snapshots are retained, not silently
rewritten to appear portable. Small/model comparison experiments are not
rerun by `run.sh`.

`report.pdf` is the formal Q1/Q2 report. Editable `report.md` and
`code/script/build_final_report.py` are supplied. To rebuild the PDF, install
`code/script/configs/requirements-dev.txt` in an authoring environment and run
the builder. It defaults to Microsoft JhengHei on Windows; set
`HW2_REPORT_FONT_DIR` to a directory containing the corresponding TrueType
collections when authoring elsewhere. Fonts are embedded as PDF subsets,
not distributed as separate font files. Poppler is used only for visual QA.

AI assistance: Codex assisted implementation, tests, logs and report drafting;
all execution results are actual saved outputs. Q2 assistant-review limitations
are explicitly disclosed. No code is automatically published and no NTU COOL
submission is performed by these scripts.
