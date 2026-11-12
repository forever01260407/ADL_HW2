# W08 — BGE-small 單一配置微調實驗

日期：2026-10-24（Asia/Taipei）。狀態：**W08 完成；Gold 與唯一一次官方 Judge 均改善，已採用微調 checkpoint，完整預設排名驗證通過。**

## 實驗問題與控制

檢驗只更新 BGE-small 權重，是否能改善商品對話檢索。對照為現行 H1；不合併 H1U 的 U2、不更換 embedding 模型家族、不增加 reranker、chunking 或需求改寫。這是小規模微調既有預訓練模型，不是從頭預訓練。

Dense 與 BM25 都沿用完整角色對話；商品仍為 title＋description、一商品一向量。Dense 上限 512 tokens、超長 query 保留首尾、query instruction 與 normalization 均保持一致。BM25 full／last weight 1／title weight 1、k1=1.5、b=0.75；兩路各取 100，等權 RRF constant 60。正式 `retrieval.py`、`evaluate.py`、`eval_utils.py` 保護 hash 不變。

先固定配置及門檻於 `results/w08_experiment_plan.json`：整體與 beauty-train 的 Gold Hit@10／MRR@10／Hit@50 全不下降，且每組至少一項提高，才執行唯一一次原版 Judge。正式 graded Recall／MRR 全不下降且至少一項提高才採用。無第二 epoch、第二配置或第二次 Judge。

## 訓練設計與資料隔離

| 項目 | 實際設定 |
|---|---|
| 初始模型 | BAAI/bge-small-en-v1.5，revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| 監督資料 | 分組 train 3,864 筆完整對話 → gold 商品；dev 966 筆不參與梯度 |
| Loss | sentence-transformers 5.1.0 `MultipleNegativesRankingLoss`，cosine，scale 20 |
| 負例 | 同批其他 train 正樣本商品；排除相同 PID／正規化後相同商品文字 |
| Optimizer | AdamW，learning rate 2e-5，weight decay 0.01，max grad norm 1 |
| 排程 | 線性 warmup 25 steps（ceil(242×0.1)），再線性降低學習率 |
| 訓練量 | 1 epoch、242 steps；241 個 batch 16、末批 8；每筆資料恰好一次 |
| 精度與 seed | FP32、seed 42、PyTorch deterministic algorithms，TF32 關閉 |
| 推論候選池 | 完整 50,000 商品；重新使用新模型編碼，沒有沿用舊向量 |

資料與模型檔案均核對 SHA-256。Dev gold 商品仍留在檢索池，但不作訓練正例或額外負例；public 未用於訓練或選型。同批其他商品仍可能是合理替代品，去重不能完全排除 false negatives，因此初版未加入 hard-negative mining。

## 工程驗證與資源

- **43 個單元測試通過**（5.816 秒，`results/w08_tests_initial.log`）；包含資料隔離、去重 batch、全資料一次覆蓋、末批 singleton、無合法負例時拒絕，以及訓練 query 與正式推論輸入一致。
- 真模型 smoke 為固定第一個 training batch 的 4 個步驟，只驗證有限 loss、非零梯度、權重更新與離線重載。它不是第二個品質實驗；其 checkpoint 不參與 dev 比較。正式 Hybrid 入口也通過 12 商品／8 題合法 top-10 驗證。
- 全量訓練的 242 個 loss／梯度均為有限值，參數確實改變；checkpoint 存檔後離線重載的測試向量最大差為 **0**。不同 batch 的 loss 不直接當成品質提升證據。

| 量測 | 開發機實測 |
|---|---:|
| 純訓練迴圈 | 30.018 秒 |
| 訓練腳本總時間（含載入、資料驗證、存檔、重載） | 53.555 秒 |
| 訓練 CUDA peak allocated | 4,244,037,120 bytes（3.953 GiB） |
| 訓練 CUDA peak reserved | 4,838,129,664 bytes（4.506 GiB） |
| 訓練 process peak RSS | 2,094,206,976 bytes（1.950 GiB） |
| 完整 dev／50k 商品檢索冷啟動 | 110.848 秒 |

環境：WSL Ubuntu 22.04、Python 3.12.12、RTX 4090 24 GB；torch 2.8.0+cu128、sentence-transformers 5.1.0、transformers 4.56.1、CPU FAISS 1.12.0。不是官方 Python 3.12.9／RTX 3070 8 GB 驗收。訓練時間沒有包含安裝／模型下載；檢索冷啟動含新程序、資料／模型載入、全商品編碼、建索引、搜尋與輸出。

## 完整 dev Gold 快篩

這些是只看標註 gold PID 的診斷，不能代替官方可接受替代品評分。

| 分組／配置 | Gold Hit@10 | Gold MRR@10 | Gold Hit@50 |
|---|---:|---:|---:|
| 全部 966 題，H1 | 0.557971 | 0.354759 | 0.703934 |
| 全部 966 題，W08 | **0.665631** | **0.481551** | **0.832298** |
| beauty-train 437 題，H1 | 0.350114 | 0.193782 | 0.505721 |
| beauty-train 437 題，W08 | **0.450801** | **0.270179** | **0.707094** |

整體 Gold Hit@10：112 題改善、8 題退步，淨增加 104 題（539 → 643）；MRR@10 317 題改善、56 題退步；Hit@50 130 題改善、6 題退步。Beauty-train Hit@10 51／7、MRR@10 114／25、Hit@50 94／6。逐題 ID、其他來源分組與排名保留於 `results/w08_screening.json`／`w08_epoch1_dev_gold.json`。

| 來源分組 | 題數 | H1 Hit@10 → W08 | H1 MRR@10 → W08 |
|---|---:|---:|---:|
| beauty-train | 437 | 0.350114 → 0.450801 | 0.193782 → 0.270179 |
| train-beauty-official | 74 | 0.797297 → 0.891892 | 0.521606 → 0.697817 |
| train-home-and-kitchen | 104 | 0.673077 → 0.817308 | 0.488366 → 0.632772 |
| train-pet-supplies | 102 | 0.735294 → 0.872549 | 0.438558 → 0.651264 |
| train-sports-and-outdoors | 130 | 0.838462 → 0.923077 | 0.582515 → 0.746966 |
| train-toys-and-games | 119 | 0.613445 → 0.722689 | 0.404755 → 0.555706 |

六項門檻均通過，已授權唯一一次官方 Judge。這只能支持此 dev 上的權重微調效果，尚未證明具體語言能力的因果改善，也不代表 public／private 成績。

固定選出 beauty-train 的前兩個 Hit10 改善／退步案例（依 qid 排序，非挑選最佳案例），保存於 `results/w08_cases.json`：

- `beauty-train-000078`：顧客以前買過 Burt's Bees 足部商品，現在要 SLS-free／美國製洗髮精。Gold 由第 45 名升至第 2 名；H1 前三名皆為狗用洗髮精，新結果前兩名為人用洗髮產品，但第三名仍有貓用洗髮精。因此只是此案例排名改善，不能宣稱歷史干擾完全消失。
- `beauty-train-000090`：敏感肌、日用、抗老保濕需求，gold 從 top-50 外升至第 2 名；前列也由爽膚水／眼霜轉向臉部保濕產品。
- `beauty-train-000495`：先前維他命 C 精華無效的情境，gold 抗老乳霜第 6 → 22 名；新前三名仍為維他命 C 精華，歷史使用經驗與當前需求的混淆未消失。
- `beauty-train-001542`：Giovanni 護髮素 gold 第 2 → 50 名，但另一款同品牌護髮素仍為第一名。這說明 gold 排名下降不必然等於沒有可接受替代品，須與官方評分分開解讀。

## 官方結果與採用

固定完整 966 題順序、原版 evaluator、本地固定 Qwen revision、batch 8、greedy／12 new tokens、不截斷。只執行此配置的一次官方評分；9,017 個非 gold 比較、**0 parse failures**。Raw output、逐候選 gain、逐題與 aggregate metrics、資料／來源 hash 已通過稽核。

| 分組／配置 | 官方 graded Recall@10 | 官方 graded MRR@10 |
|---|---:|---:|
| 全部 966 題，H1 | 0.739130 | 0.465780 |
| 全部 966 題，W08 | **0.829193** | **0.611836** |
| beauty-train 437 題，H1 | 0.633867 | 0.359200 |
| beauty-train 437 題，W08 | **0.741419** | **0.481396** |

整體可接受商品命中 **714 → 801 題**，淨增加 87 題，Recall +9.006 個百分點；110 題改善、23 題退步。MRR +0.146056，382 題改善、94 題退步。Beauty-train 命中 277 → 324 題，兩項指標也提高。W08 同時通過 Gold 與正式採用政策，因此選用微調後 BGE，H1 的 BM25／RRF 設定保持不變。

Judge 冷啟動 577.955 秒，peak CUDA allocated 4.569 GiB／reserved 11.178 GiB。Reserved 含 allocator cache，不能據此斷言官方 8 GB 一定 OOM，也不能宣稱已通過驗收。檢索 peak allocated 0.469 GiB／reserved 0.668 GiB。

舊來源已保存至 `code/script/snapshots/h1u/`；只變更正式 student block 的模型預設路徑為 `artifacts/training/w08_epoch1/checkpoint`，其他預設維持 Hybrid、BM25 full／1／1、MiniLM 關閉。切換後 **43 tests／5.625 秒通過**；不帶任何模型／檢索設定覆寫的正式入口已完成全 966 題，冷啟動 **104.861 秒**，top-50 逐題與被評估的 W08 完全一致。Checkpoint、預設來源與無覆寫環境均已核對；B0／A1／H1／H1U 的歷史來源與輸入 hashes 仍可成功稽核。

選型當下的紀錄保留於 `results/w08_summary.json`（`promotion_pending` 是當時狀態）；**最終完成紀錄為 `results/w08_final.json`，`adopt=true`、`official_judge_runs=1`、`default_verified=true`、`default_action=promoted_and_verified`**。未覆寫或追加原始單次 Judge。

## 程式與重現

- `code/script/train.py`：訓練與 smoke，各次使用獨立輸出資料夾，拒絕覆寫。
- `code/script/configs/w08_train.json`：本輪固定設定。
- `run.sh`：只重現訓練，不自動跑評分或提升預設。
- `code/script/audit_w08.py`：核對資料／模型／loss／排名，檢查 Gold 門檻與單次 Judge。
- `code/script/finalize_w08.py`：只讀核對預設無覆寫、checkpoint 與完整排名一致，保存最終驗證紀錄；不啟動模型或 Judge。
- `artifacts/training/w08_epoch1/`：`checkpoint/`、`training_manifest.json`、`batches.json`、完整 `loss.jsonl`。
- `results/w08_*`：測試、正式入口 smoke、檢索冷啟動、Gold 與官方評分原始產物。

在目前 WSL 環境從專案根目錄執行；已有產物時使用新的 output 路徑：

```bash
HW2_PYTHON=.venv/bin/python bash run.sh --output artifacts/training/w08_reproduction

HW2_MODEL_PATH=artifacts/training/w08_epoch1/checkpoint \
HW2_RETRIEVER=hybrid HW2_RERANK=0 HW2_RRF_CANDIDATES=100 \
HW2_BM25_QUERY_MODE=full HW2_BM25_LAST_WEIGHT=1 HW2_BM25_TITLE_WEIGHT=1 \
.venv/bin/python code/script/run_measured.py --output results/w08_repro_timing.json -- \
  retrieval.py --dialogues data/derived/dev.jsonl --products data/train/products.jsonl \
  --top-k 50 --output results/w08_repro_top50.json
```

精確重現 H1／B0／A1 舊實驗時，需顯式設定 `HW2_MODEL_PATH=artifacts/models/bge-small-en-v1.5`，再依該實驗指定 retriever／reranker；微調 checkpoint 與舊模型不可混用。相同 seed 不等於跨硬體、不同套件版本逐位元一致。

## 限制與剩餘工作

同一份 dev 已用於多輪策略選型，並非獨立測試集。不同 PID 的合理替代品可能成為 false negatives；完整商品超過 512 tokens 的既有限制未解決。官方 Judge 先前有組批敏感性，本輪固定順序與 batch 8，不重跑挑選較高分。Q2、public／官方硬體、下載腳本與報告封裝仍屬後續工作；完成 W08 不代表整份作業完成。
