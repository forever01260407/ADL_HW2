# H2B — 高 Dense 權重的轉折點與單次 dev 官方評分

日期：2026-10-24（Asia/Taipei）。**四組追加 Gold 消融與唯一候選 6:1 的官方 dev Judge 已完成並通過稽核。**6:1 原版 dev Recall@10 **0.845756**／MRR@10 **0.629401**，均高於 W08 1:1 的 0.829193／0.611836。原 H2 產物全部保留，正式預設仍為 W08 1:1；本輪不訓練、不重跑 Public、不進 W11、不產生 report.pdf 或打包腳本。

整體改善有代價：pet MRR 明顯下降，sports 兩項下降，toys Recall 提高但 MRR 幾乎不變。本輪只驗證 6:1 的官方 dev 表現，沒有證明其官方分數勝過 5:1 或已達 Boss。完整數據與限制見後面的官方評分結果。

## 實驗範圍與選型

使用者明確要求補測 Dense:BM25＝5:1、6:1、8:1、10:1，並對最佳設定執行一次完整 dev 原版 evaluate.py。此次授權取代上一輪不追加比例／不跑 Judge 的限制，另存 H2B，不覆寫歷史。

沿用已核對的 **W08 微調後 Dense top-100** 與 BM25 full／末輪 1／title 1 top-100；完整 dev 966 題、完整 50,000 商品、RRF 常數 60、完整角色對話與商品表示均不變。沒有使用微調前 Dense、Public 標籤、Judge 分數或品類分流權重。

原八組＋新四組共 12 組，只依預先固定的 **Gold MRR@10 → Gold Hit@10 → Gold Hit@50** 選出唯一候選。四組設定先凍結於 `results/h2b_rrf_plan.json`，在任何本輪 Judge 之前選定 **6:1**，排名及來源再凍結於 `results/h2b_judge_plan.json`。4／5／8／10:1 不送第二輪 Judge。

## Gold 結果的解讀

**5:1～6:1 是這批測試點中的平台區間，8:1、10:1 開始回落。**5:1 的 Gold Hit@10 最高（699／966＝0.723602），6:1 的 Gold MRR@10 最高（0.518419），因此按既定 MRR 優先規則選 6:1。不能說某個比例在所有指標都最高，也不能據這些離散測試點證明連續權重的全域最高峰。

- 6:1 相對 4:1：Hit10 697 → 698 題，MRR10 +0.001095，Hit50 都是 833 題。改善幅度很小。
- 6:1 相對 5:1：Hit10 少 1 題，MRR10 僅高 0.000499；不宣稱顯著勝出。
- 6:1 相對現行 1:1：Gold Hit10 0.665631 → 0.722567、MRR10 0.481551 → 0.518419、Hit50 0.832298 → 0.862319。
- 分組有取捨：6:1 對 beauty、home、toys 的三項整體品類 Gold 均提高；pet 的 MRR10 0.651264 → 0.604505、Hit10 0.872549 → 0.862745；sports 的 Hit10 0.923077 → 0.892308、MRR10 0.746966 → 0.745800。不能把整體 Gold 最佳解讀為各品類都改善。

## 測試與來源核對

新增三個測試，完整 **69 tests／6.003 秒／OK**，涵蓋選型優先順序與同分、非法指標、四種新權重的獨立公式／排名損壞拒絕、Recall／MRR 取捨的逐題統計；見 `results/h2b_tests.log`。

所有 H2 凍結產物、微調 checkpoint、資料及官方三檔指紋重新核對；1:1 再次完整重現 W08 的 966 題 top-50。四組新配置合計 **3,864 份 top-50** 另用 rank-map 公式獨立重建，與累加版融合完全一致。每組 top-10 是 top-50 的前綴，Gold 使用既有 `analyze_results.py`。

## 官方 Judge 設定與範圍

- 唯一候選：6:1，`results/h2b_d6_b1_top10.json`；SHA-256 `748ae810646621ab39de61fe2f59ec6651483eb659b444e3ae7048eebea6cb7c`。
- 完整 dev 966 題、8,962 個非 gold 比較；無抽樣、無額外 Judge smoke、無重跑挑分。
- 原版 `evaluate.py`／`eval_utils.py`／`retrieval.py` 未修改；使用 `Qwen/Qwen2.5-1.5B-Instruct` revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，本地權重逐檔核對下載 manifest。
- bf16、greedy、batch 8、12 new tokens；所有原版 render 輸入先計算長度，最大 **2,285 tokens**，context 32,768，**不截斷**。
- 本輪正式 evaluator 直接讀取保存的合法排名；`retrieval_utils.py`／模型／推論預設未改。若後續採用新權重，仍須補正式 student-block 權重支援與完整管線排名一致性驗證。
- 本機 RTX 4090／Python 3.12.12 不等於官方 RTX 3070 8 GB／Python 3.12.9 驗收；重用排名的快篩時間不能當作正式 Hybrid 冷啟動。

## 保存內容與重現

| 檔案 | 用途 |
|---|---|
| `code/script/extend_rrf_ablation.py` | prepare → screen → prepare-judge → judge → final；完整 dev 消融、選型與單次原版 Judge／原始 gain 核對 |
| `code/script/tests/test_rrf_extension.py` | 選型、獨立融合與逐題 graded 統計測試 |
| `results/h2b_rrf_plan.json` | 看新結果前固定的四比例、選型次序及來源 hashes |
| `results/h2b_rrf_screening.json` | 原八組＋新四組 Gold、各品類／來源、選定候選與排名 hashes |
| `results/h2b_{d5,d6,d8,d10}_b1_{top50,top10,gold}.json` | 四組完整排名及逐題 Gold 分析 |
| `results/h2b_judge_plan.json` | 唯一勝出排名、固定 Qwen 權重、原版參數與 token preflight |
| `results/h2b_winner_graded.json` | 官方 evaluator 的完整原始回答、gain、逐題及 aggregate 指標 |
| `results/h2b_winner_judge_timing.{json,worker.json,log}` | 全程序時間、來源及資源 |
| `results/h2b_rrf_final.json` | 完成後的原始結果稽核、對 W08 的逐題／品類／來源 graded 比較 |
| `results/h2b_gold_audit.json` | 四組新 Gold、五品類／六來源獨立核對及 6:1 對 4／5:1 的逐題差異 |

在 WSL 專案根目錄與既有 `.venv` 執行；各階段拒絕覆寫原有實驗／Judge 紀錄：

```bash
.venv/bin/python -m unittest discover -s code/script/tests -v
.venv/bin/python code/script/extend_rrf_ablation.py --stage prepare
.venv/bin/python code/script/extend_rrf_ablation.py --stage screen
.venv/bin/python code/script/extend_rrf_ablation.py --stage prepare-judge
.venv/bin/python code/script/extend_rrf_ablation.py --stage judge
.venv/bin/python code/script/extend_rrf_ablation.py --stage final
```

`judge` 以原版 `evaluate.py`、`--dialogues data/derived/dev.jsonl --products data/train/products.jsonl --model artifacts/models/Qwen2.5-1.5B-Instruct --batch-size 8 --max-new-tokens 12 --top-k 10 --device cuda` 執行；完整命令保存於計時 JSON。

## Gold 完整數據

以下數字均為 Gold-only；不是官方 graded 分數。五品類中的 beauty 合併 beauty-train 437 與 beauty-official 74；六來源分組另外保存於 screening JSON，避免混淆。

### Gold overall (n=966)

| Dense:BM25 | Gold Hit@10 | Gold MRR@10 | Gold Hit@50 |
|---|---:|---:|---:|
| Dense-only (1:0) | 0.709110 | 0.509199 | 0.851967 |
| 0.75:1 | 0.620083 | 0.471612 | 0.798137 |
| 1:1 (W08) | 0.665631 | 0.481551 | 0.832298 |
| 1.1:1 | 0.678054 | 0.485509 | 0.843685 |
| 1.5:1 | 0.690476 | 0.496447 | 0.858178 |
| 2:1 | 0.700828 | 0.506253 | 0.864389 |
| 3:1 | 0.715321 | 0.515907 | 0.864389 |
| 4:1 | 0.721532 | 0.517324 | 0.862319 |
| 5:1 | 0.723602 | 0.517920 | 0.862319 |
| 6:1 | 0.722567 | 0.518419 | 0.862319 |
| 8:1 | 0.719462 | 0.514959 | 0.859213 |
| 10:1 | 0.717391 | 0.514727 | 0.859213 |

### Gold by category: baseline, 4:1, and extended weights

| Category | n | Dense:BM25 | Gold Hit@10 | Gold MRR@10 | Gold Hit@50 |
|---|---:|---|---:|---:|---:|
| beauty | 511 | 1:1 (W08) | 0.514677 | 0.332107 | 0.741683 |
| beauty | 511 | 4:1 | 0.610568 | 0.397020 | 0.790607 |
| beauty | 511 | 5:1 | 0.618395 | 0.403699 | 0.786693 |
| beauty | 511 | 6:1 | 0.618395 | 0.407114 | 0.786693 |
| beauty | 511 | 8:1 | 0.614481 | 0.409080 | 0.786693 |
| beauty | 511 | 10:1 | 0.614481 | 0.411654 | 0.786693 |
| home-and-kitchen | 104 | 1:1 (W08) | 0.817308 | 0.632772 | 0.980769 |
| home-and-kitchen | 104 | 4:1 | 0.875000 | 0.655975 | 0.990385 |
| home-and-kitchen | 104 | 5:1 | 0.865385 | 0.638282 | 0.990385 |
| home-and-kitchen | 104 | 6:1 | 0.855769 | 0.640087 | 0.990385 |
| home-and-kitchen | 104 | 8:1 | 0.855769 | 0.630407 | 0.980769 |
| home-and-kitchen | 104 | 10:1 | 0.855769 | 0.629605 | 0.980769 |
| pet-supplies | 102 | 1:1 (W08) | 0.872549 | 0.651264 | 0.931373 |
| pet-supplies | 102 | 4:1 | 0.882353 | 0.621168 | 0.941176 |
| pet-supplies | 102 | 5:1 | 0.862745 | 0.609827 | 0.950980 |
| pet-supplies | 102 | 6:1 | 0.862745 | 0.604505 | 0.950980 |
| pet-supplies | 102 | 8:1 | 0.862745 | 0.588204 | 0.941176 |
| pet-supplies | 102 | 10:1 | 0.862745 | 0.578957 | 0.941176 |
| sports-and-outdoors | 130 | 1:1 (W08) | 0.923077 | 0.746966 | 0.969231 |
| sports-and-outdoors | 130 | 4:1 | 0.892308 | 0.756996 | 0.976923 |
| sports-and-outdoors | 130 | 5:1 | 0.892308 | 0.755275 | 0.984615 |
| sports-and-outdoors | 130 | 6:1 | 0.892308 | 0.745800 | 0.984615 |
| sports-and-outdoors | 130 | 8:1 | 0.892308 | 0.734710 | 0.984615 |
| sports-and-outdoors | 130 | 10:1 | 0.884615 | 0.730855 | 0.984615 |
| toys-and-games | 119 | 1:1 (W08) | 0.722689 | 0.555706 | 0.857143 |
| toys-and-games | 119 | 4:1 | 0.739496 | 0.561915 | 0.865546 |
| toys-and-games | 119 | 5:1 | 0.747899 | 0.565136 | 0.865546 |
| toys-and-games | 119 | 6:1 | 0.747899 | 0.567857 | 0.865546 |
| toys-and-games | 119 | 8:1 | 0.739496 | 0.565873 | 0.857143 |
| toys-and-games | 119 | 10:1 | 0.731092 | 0.565780 | 0.857143 |

### Gold by source: 6:1 versus 1:1

| Source | n | Dense:BM25 | Gold Hit@10 | Gold MRR@10 | Gold Hit@50 |
|---|---:|---|---:|---:|---:|
| beauty-train | 437 | 1:1 (W08) | 0.450801 | 0.270179 | 0.707094 |
| beauty-train | 437 | 6:1 | 0.574371 | 0.362401 | 0.757437 |
| train-beauty-official | 74 | 1:1 (W08) | 0.891892 | 0.697817 | 0.945946 |
| train-beauty-official | 74 | 6:1 | 0.878378 | 0.671166 | 0.959459 |
| train-home-and-kitchen | 104 | 1:1 (W08) | 0.817308 | 0.632772 | 0.980769 |
| train-home-and-kitchen | 104 | 6:1 | 0.855769 | 0.640087 | 0.990385 |
| train-pet-supplies | 102 | 1:1 (W08) | 0.872549 | 0.651264 | 0.931373 |
| train-pet-supplies | 102 | 6:1 | 0.862745 | 0.604505 | 0.950980 |
| train-sports-and-outdoors | 130 | 1:1 (W08) | 0.923077 | 0.746966 | 0.969231 |
| train-sports-and-outdoors | 130 | 6:1 | 0.892308 | 0.745800 | 0.984615 |
| train-toys-and-games | 119 | 1:1 (W08) | 0.722689 | 0.555706 | 0.857143 |
| train-toys-and-games | 119 | 6:1 | 0.747899 | 0.567857 | 0.865546 |

## 官方 dev Judge 結果

| Setting | Recall@10 | MRR@10 | Hit queries |
|---|---:|---:|---:|
| W08 1:1 | 0.829192547 | 0.611835749 | 801/966 |
| H2B 6:1 | 0.845755694 | 0.629400818 | 817/966 |

Recall@10：52 題改善、36 題退步，淨增 16 題（801 → 817）。MRR@10：222 題改善、172 題退步。

| Category | n | Recall 1:1 | Recall 6:1 | MRR 1:1 | MRR 6:1 |
|---|---:|---:|---:|---:|---:|
| beauty | 511 | 0.769080 | 0.792564 | 0.517285 | 0.563581 |
| home-and-kitchen | 104 | 0.913462 | 0.903846 | 0.675080 | 0.701293 |
| pet-supplies | 102 | 0.950980 | 0.960784 | 0.768585 | 0.691678 |
| sports-and-outdoors | 130 | 0.953846 | 0.930769 | 0.792436 | 0.780415 |
| toys-and-games | 119 | 0.773109 | 0.831933 | 0.630926 | 0.630856 |

| Source | n | Recall 1:1 | Recall 6:1 | MRR 1:1 | MRR 6:1 |
|---|---:|---:|---:|---:|---:|
| beauty-train | 437 | 0.741419 | 0.764302 | 0.481396 | 0.533234 |
| train-beauty-official | 74 | 0.932432 | 0.959459 | 0.729220 | 0.742787 |
| train-home-and-kitchen | 104 | 0.913462 | 0.903846 | 0.675080 | 0.701293 |
| train-pet-supplies | 102 | 0.950980 | 0.960784 | 0.768585 | 0.691678 |
| train-sports-and-outdoors | 130 | 0.953846 | 0.930769 | 0.792436 | 0.780415 |
| train-toys-and-games | 119 | 0.773109 | 0.831933 | 0.630926 | 0.630856 |

## 官方結果稽核與限制

- 8,962 次非 gold 比較、0 次解析失敗；原始回答、gain、題目順序與 aggregate 指標均已核對。
- 698 題的前十名包含 gold，另 119 題由 Judge 認可替代品補命中，合計 817 題。
- 完整程序 564.747 秒；CUDA allocated 4.568 GiB／reserved 9.441 GiB，程序 RSS 6.033 GiB。這是 RTX 4090／Python 3.12.12 開發結果，官方 8 GiB 硬體尚未驗收。
- 與 W08 共有 4,773 個相同的非 gold 配對，其中 60 個 gain 改變。兩次候選列表與組批不同，原因尚未隔離；保留兩份原始結果，不追加 Judge 挑較好分數。
- 同一 dev 多次用於選型，沒有統計顯著或未見資料泛化保證。本輪只有 6:1 接受官方 Judge，因此尚不知道其官方分數是否勝過 5:1。
- 沒有新 Public 分數。已知 W08 Public Recall 0.902000／MRR 0.704348 保持原紀錄；不能將 dev 改善量加到這些分數，也不能宣稱 Boss 已過。
- 本輪實驗完成並停止。預設仍 W08 1:1，未訓練、重跑 Public 或進入 W11。若後續採用 6:1，需補正式 student-block 全域權重支援，並用完整 dev 確認排名與本次被評分的結果一致。
