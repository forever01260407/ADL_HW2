# H2 — W08 微調 Dense／BM25 的 RRF 權重消融

日期：2026-10-24（Asia/Taipei）。**完整 dev 966 題／50,000 商品，八組固定配置已完成；正式預設仍是 W08 的 1:1。**本輪沒有訓練、重排模型、官方 Judge、Public 重跑、W11、report.pdf 或封裝腳本。

這次證據支持「提高微調後 Dense 的權重」，但尚不能宣稱官方分數提高或達到 Boss。固定八組中，依預先約定的 Gold MRR@10 → Hit@10 → Hit@50 排序，最佳為 **Dense:BM25＝4:1**。這是本次有限比較的最佳，不是所有可能比例的最優解。

## 1. 先保存來源，再確認公平比較

- 新補跑 `results/h2_w08_dense_top100.json`，明確指定 **W08 微調 checkpoint**：`artifacts/training/w08_epoch1/checkpoint`，不是 models.lock 中的原版 BGE。checkpoint 的 10 個檔案逐一核對訓練 manifest 的 SHA-256。
- `results/h1u_dense_top100.json` 是微調前的 Dense；**本輪完全未讀取或挪用**。融合 top-50 不能逆推出兩路 top-100。
- BM25 沿用 `h1u_u0_full_lexical_top100.json` 與 `h1u_u0_full_trace.json`。歷史 SHA-256、完整 dev／商品資料、full／末輪 1／title 1、兩路候選 100 均已核對；整段 BM25 程式與歷史來源一致。只計正分 BM25 候選，零分不獲得融合票。
- **先用新 Dense 與既有 BM25 重組 1:1，966 題的每一個 top-50 排序全部重現 W08，差異 0 題，才進行其他配置分析。**證據：`results/h2_rrf_baseline_check.json`。
- Dense 與 BM25 均維持完整原始角色對話；商品、模型、512-token 設定、RRF 常數 60、兩路各 100、同分 PID 次排序均固定。唯一變因是全域路徑權重，不依品類／qid 設定權重。

公式：`score(p) = w_dense / (60 + dense_rank(p)) + w_bm25 / (60 + bm25_rank(p))`；未出現在該路候選的商品，該項為 0。Dense-only 使用權重 1:0。

計畫在模型檢索前已保存於 `results/h2_rrf_plan.json`，完整設定及來源／模型檔案 hashes 均在其中。沒有看完結果再追加比例或依 Public 標籤選型。

## 2. 結果與判斷

4:1 相對現行 1:1：

- **Gold Hit@10：0.665631 → 0.721532，增加 5.590 個百分點。**命中 643 → 697／966；77 題改善、23 題退步，淨增加 54 題。
- **Gold MRR@10：0.481551 → 0.517324，增加 0.035773。**223 題改善、95 題退步。
- **Gold Hit@50：0.832298 → 0.862319，增加 3.002 個百分點。**命中 804 → 833；38 題改善、9 題退步。

純微調 Dense 也已優於 1:1 的三項整體 Gold 指標。4:1 再比 Dense-only 的 Hit10 多 12 題、Hit50 多 10 題，MRR10 高約 0.008125，說明 BM25 的排序訊號仍有價值。不能把結論寫成「BM25 沒用」或「所有 Hybrid 都比 Dense 差」。

**分組有取捨：**4:1 對 beauty、home-and-kitchen、toys-and-games 三指標均提高；pet-supplies 的 Hit10／Hit50 提高，但 MRR10 0.651264 → 0.621168；sports-and-outdoors 的 MRR10／Hit50 提高，但 Hit10 0.923077 → 0.892308（少 4 題）。beauty-train 三項皆提高，但 beauty-official 的 MRR 微降；表格保留兩個來源，避免合併後看不出差異。3:1 的整體 Hit50 比 4:1 多 2 題，整體 Hit10／MRR10 則較低。

較高權重不是輕微的調整：因 `2/110 > 1/61`，2:1 以上時，即使 BM25-only 商品排第 1，也無法超過 Dense 第 50 名的最低票數。本次 2:1、3:1、4:1 的 top-50 均沒有 Dense top-100 以外的商品；BM25 主要是在 Dense top-100 內重排，而不是完全失去作用。1.1:1／1.5:1 也已納入，不只測整數比例。

## 3. 驗證與限制

- 新增 7 個測試，完整 **66 tests／5.983 秒／OK**，紀錄 `results/h2_rrf_tests.log`；涵蓋公式、零分／零權重、同分、邊界、錯誤輸入、基準不一致拒絕、分組與逐題統計。
- `code/script/audit_rrf_ablation.py` 不使用融合函式，另以兩路 rank-map 公式獨立重建 **8×966＝7,728** 份排名；整體、五品類、六來源的 Gold 指標與逐題得失全部核對一致。證據：`results/h2_rrf_audit.json`。
- 補跑 Dense top-100 冷啟動 **105.101 秒**，CUDA allocated 0.469 GiB／reserved 0.668 GiB、RSS 約 1.665 GiB。這是 RTX 4090／Python 3.12.12 的開發結果，不是官方 RTX 3070／Python 3.12.9 驗收。其餘比例為重用排名的 CPU 消融，不能當作完整 Hybrid 冷啟動速度。
- Gold 只判斷唯一標註 PID；官方 Judge 還能認可其他合理商品。因此這些變化不能直接換算成官方 Recall／MRR、Boss 是否通過或可得幾分。**目前唯一已跑過的 Public 分數仍是凍結 W08 的 Recall 0.902000／MRR 0.704348。**
- 同一 dev 已被多次用於選型；本輪不是獨立未見測試，也沒有顯著性或泛化保證。Public 已在 W10 看過，本輪不讀取其標籤或結果檔進行調參。

建議下一個獨立步驟：針對 **4:1 單一候選**實作可切換的正式 student-block 權重，先確認完整 dev top-50 與本次快篩一致，再執行一次完整 dev 原版 Judge；同時核對整體與 pet／sports 的 graded 得失，通過後才考慮改預設。本輪依授權只回報 Gold 數據，尚未執行這一步，也不追加 5:1、6:1 等比例或進入 W11。

## 4. 保存內容與重現方式

| 路徑 | 用途 |
|---|---|
| `results/h2_rrf_plan.json` | 預先凍結的八組設定、W08 checkpoint 每檔與輸入來源 hashes、BM25 快取稽核 |
| `results/h2_w08_dense_top100.json` | 新計算的微調後 Dense 原始排名；未來可直接重用，須驗證指紋 |
| `results/h2_w08_dense_timing.{json,log,worker.json}` | 官方 retrieval.py 入口命令、來源／資料指紋、計時及資源 |
| `results/h1u_u0_full_lexical_top100.json`／`h1u_u0_full_trace.json` | 既有、經核對的 BM25 原始排名及正分數量 |
| `results/h2_rrf_baseline_check.json` | 1:1 重現 966 題完整 top-50、Dense 來源與新檔 hashes |
| `results/h2_{variant}_{top50,top10,gold}.json` | 每組排名及 analyze_results.py 的逐題／分組 Gold 分析 |
| `results/h2_rrf_summary.json` | 全部指標、品類與來源分組、逐題改善／退步 qid、描述性最佳、每組檔案 hashes |
| `results/h2_rrf_audit.{json,log}` | 獨立公式／統計／檔案核對 |
| `code/script/run_rrf_ablation.py` | prepare → screen；拒絕覆寫既有計畫或結果、不呼叫模型訓練／Judge |
| `code/script/audit_rrf_ablation.py` | 不載入模型的獨立核對 |

以下為本次實際流程，於 WSL 專案根目錄使用既有 `.venv`；既有紀錄拒絕覆寫，重新實驗須使用新的實驗名稱／路徑，不能直接蓋掉本次產物。

```bash
.venv/bin/python -m unittest discover -s code/script/tests -v
.venv/bin/python code/script/run_rrf_ablation.py --stage prepare

HW2_RETRIEVER=dense HW2_RERANK=0 \
HW2_MODEL_PATH=artifacts/training/w08_epoch1/checkpoint \
HW2_RRF_CANDIDATES=100 HW2_BM25_QUERY_MODE=full \
HW2_BM25_LAST_WEIGHT=1 HW2_BM25_TITLE_WEIGHT=1 \
.venv/bin/python code/script/run_measured.py \
  --output results/h2_w08_dense_timing.json -- retrieval.py \
  --dialogues data/derived/dev.jsonl --products data/train/products.jsonl \
  --top-k 100 --output results/h2_w08_dense_top100.json

.venv/bin/python code/script/run_rrf_ablation.py --stage screen
.venv/bin/python code/script/audit_rrf_ablation.py
```

`screen` 對八組逐一呼叫 `analyze_results.py`；實際參數／輸入 hashes 均保存在每組 Gold JSON。所有結果保留，不因分組退步刪除配置。`PLAN.md`、`DECISIONS.md`、`PROGRESS.md` 記錄本輪範圍與完成狀態；本文件是實驗紀錄，不是 report.pdf。

## 5. 完整數據表

以下表格從已通過獨立稽核的 `results/h2_rrf_summary.json` 產生。所有指標均為 **Gold-only**。

### 5.1 Overall (n=966)

| Dense:BM25 | Gold Hit@10 | Gold MRR@10 | Gold Hit@50 | Hit@10 + / - vs 1:1 | BM25-only in top-50 |
|---|---:|---:|---:|---:|---:|
| Dense-only (1:0) | 0.709110 | 0.509199 | 0.851967 | +94 / -52 | 0 |
| 0.75:1 | 0.620083 | 0.471612 | 0.798137 | +3 / -47 | 22232 |
| 1:1 (W08) | 0.665631 | 0.481551 | 0.832298 | +0 / -0 | 14022 |
| 1.1:1 | 0.678054 | 0.485509 | 0.843685 | +14 / -2 | 11334 |
| 1.5:1 | 0.690476 | 0.496447 | 0.858178 | +29 / -5 | 3071 |
| 2:1 | 0.700828 | 0.506253 | 0.864389 | +44 / -10 | 0 |
| 3:1 | 0.715321 | 0.515907 | 0.864389 | +66 / -18 | 0 |
| 4:1 | 0.721532 | 0.517324 | 0.862319 | +77 / -23 | 0 |

### 5.2 All five categories, all eight settings

| Category | n | Dense:BM25 | Gold Hit@10 | Gold MRR@10 | Gold Hit@50 |
|---|---:|---|---:|---:|---:|
| beauty | 511 | Dense-only (1:0) | 0.610568 | 0.418454 | 0.788650 |
| beauty | 511 | 0.75:1 | 0.442270 | 0.315778 | 0.688845 |
| beauty | 511 | 1:1 (W08) | 0.514677 | 0.332107 | 0.741683 |
| beauty | 511 | 1.1:1 | 0.532290 | 0.335671 | 0.763209 |
| beauty | 511 | 1.5:1 | 0.551859 | 0.350568 | 0.786693 |
| beauty | 511 | 2:1 | 0.571429 | 0.365444 | 0.792564 |
| beauty | 511 | 3:1 | 0.600783 | 0.385658 | 0.792564 |
| beauty | 511 | 4:1 | 0.610568 | 0.397020 | 0.790607 |
| home-and-kitchen | 104 | Dense-only (1:0) | 0.855769 | 0.607025 | 0.971154 |
| home-and-kitchen | 104 | 0.75:1 | 0.798077 | 0.632700 | 0.951923 |
| home-and-kitchen | 104 | 1:1 (W08) | 0.817308 | 0.632772 | 0.980769 |
| home-and-kitchen | 104 | 1.1:1 | 0.836538 | 0.647073 | 0.980769 |
| home-and-kitchen | 104 | 1.5:1 | 0.836538 | 0.652831 | 0.980769 |
| home-and-kitchen | 104 | 2:1 | 0.846154 | 0.649599 | 0.990385 |
| home-and-kitchen | 104 | 3:1 | 0.865385 | 0.665060 | 0.990385 |
| home-and-kitchen | 104 | 4:1 | 0.875000 | 0.655975 | 0.990385 |
| pet-supplies | 102 | Dense-only (1:0) | 0.833333 | 0.528933 | 0.911765 |
| pet-supplies | 102 | 0.75:1 | 0.882353 | 0.657376 | 0.931373 |
| pet-supplies | 102 | 1:1 (W08) | 0.872549 | 0.651264 | 0.931373 |
| pet-supplies | 102 | 1.1:1 | 0.872549 | 0.647179 | 0.931373 |
| pet-supplies | 102 | 1.5:1 | 0.882353 | 0.649537 | 0.931373 |
| pet-supplies | 102 | 2:1 | 0.882353 | 0.647386 | 0.941176 |
| pet-supplies | 102 | 3:1 | 0.872549 | 0.624358 | 0.941176 |
| pet-supplies | 102 | 4:1 | 0.882353 | 0.621168 | 0.941176 |
| sports-and-outdoors | 130 | Dense-only (1:0) | 0.869231 | 0.709881 | 0.969231 |
| sports-and-outdoors | 130 | 0.75:1 | 0.915385 | 0.732714 | 0.953846 |
| sports-and-outdoors | 130 | 1:1 (W08) | 0.923077 | 0.746966 | 0.969231 |
| sports-and-outdoors | 130 | 1.1:1 | 0.923077 | 0.752552 | 0.969231 |
| sports-and-outdoors | 130 | 1.5:1 | 0.915385 | 0.761056 | 0.976923 |
| sports-and-outdoors | 130 | 2:1 | 0.907692 | 0.772094 | 0.976923 |
| sports-and-outdoors | 130 | 3:1 | 0.900000 | 0.772509 | 0.976923 |
| sports-and-outdoors | 130 | 4:1 | 0.892308 | 0.756996 | 0.976923 |
| toys-and-games | 119 | Dense-only (1:0) | 0.722689 | 0.577228 | 0.840336 |
| toys-and-games | 119 | 0.75:1 | 0.680672 | 0.555532 | 0.848739 |
| toys-and-games | 119 | 1:1 (W08) | 0.722689 | 0.555706 | 0.857143 |
| toys-and-games | 119 | 1.1:1 | 0.731092 | 0.557433 | 0.857143 |
| toys-and-games | 119 | 1.5:1 | 0.747899 | 0.565906 | 0.865546 |
| toys-and-games | 119 | 2:1 | 0.747899 | 0.574240 | 0.873950 |
| toys-and-games | 119 | 3:1 | 0.739496 | 0.571579 | 0.873950 |
| toys-and-games | 119 | 4:1 | 0.739496 | 0.561915 | 0.865546 |

### 5.3 Beauty source groups kept separate

| Source | n | Dense:BM25 | Gold Hit@10 | Gold MRR@10 | Gold Hit@50 |
|---|---:|---|---:|---:|---:|
| beauty-train | 437 | Dense-only (1:0) | 0.574371 | 0.389575 | 0.759725 |
| beauty-train | 437 | 0.75:1 | 0.368421 | 0.250274 | 0.645309 |
| beauty-train | 437 | 1:1 (W08) | 0.450801 | 0.270179 | 0.707094 |
| beauty-train | 437 | 1.1:1 | 0.469108 | 0.274706 | 0.729977 |
| beauty-train | 437 | 1.5:1 | 0.491991 | 0.290898 | 0.757437 |
| beauty-train | 437 | 2:1 | 0.517162 | 0.309456 | 0.764302 |
| beauty-train | 437 | 3:1 | 0.549199 | 0.332512 | 0.764302 |
| beauty-train | 437 | 4:1 | 0.560641 | 0.348353 | 0.762014 |
| train-beauty-official | 74 | Dense-only (1:0) | 0.824324 | 0.588996 | 0.959459 |
| train-beauty-official | 74 | 0.75:1 | 0.878378 | 0.702606 | 0.945946 |
| train-beauty-official | 74 | 1:1 (W08) | 0.891892 | 0.697817 | 0.945946 |
| train-beauty-official | 74 | 1.1:1 | 0.905405 | 0.695694 | 0.959459 |
| train-beauty-official | 74 | 1.5:1 | 0.905405 | 0.702944 | 0.959459 |
| train-beauty-official | 74 | 2:1 | 0.891892 | 0.696075 | 0.959459 |
| train-beauty-official | 74 | 3:1 | 0.905405 | 0.699507 | 0.959459 |
| train-beauty-official | 74 | 4:1 | 0.905405 | 0.684417 | 0.959459 |

### 5.4 Category gains/losses: 4:1 vs 1:1

| Category | Hit@10 + / - | MRR@10 + / - | Hit@50 + / - |
|---|---:|---:|---:|
| beauty | +56 / -7 | +142 / -32 | +31 / -6 |
| home-and-kitchen | +10 / -4 | +25 / -15 | +1 / -0 |
| pet-supplies | +3 / -2 | +17 / -18 | +2 / -1 |
| sports-and-outdoors | +1 / -5 | +20 / -15 | +1 / -0 |
| toys-and-games | +7 / -5 | +19 / -15 | +3 / -2 |
