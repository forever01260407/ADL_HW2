# H6 — 已採用 H5 的完整 Public 確認

日期：2026-10-25（Asia/Taipei）  
狀態：完整檢索、唯一一次原版 Judge 與原始結果稽核完成；本次 Public 兩項皆超過 Boss。

## 結果

本次是 **完整 Public 1,000 題／75,000 商品的原版 Graded 評分**，不是 Dev、Gold 診斷或訓練分數。

| 配置 | Graded Recall@10 | Graded MRR@10 | 命中題數 | Public 配分估計 |
|---|---:|---:|---:|---:|
| W10：微調 small＋1:1（歷史） | 0.902000 | 0.7043484126984128 | 902／1000 | 6／8 |
| H4：原微調 base＋5:1（歷史） | 0.933000 | 0.7479154761904762 | 933／1000 | 6／8 |
| **H6：H5 難負例續訓 base＋5:1** | **0.952000** | **0.8024043650793651** | **952／1000** | **8／8** |
| Public Boss 門檻 | 0.950000 | 0.800000 | — | — |

兩項以未四捨五入值判斷均超過門檻：Recall 高 0.002（2 題），MRR 高 0.0024043650793651。這是本機單次原版 Judge 的達標紀錄，餘裕有限，不保證不同環境／組批下逐項完全相同。Private 尚未評估，report 未評分；8／8 是 Public 可累加的課程百分點估計，不是整份作業已滿分或已獲助教確認。

相較 H4，Recall **+0.019**、MRR **+0.0544888888888889**，命中淨增 **19 題**。逐題 Recall 29 題改善／10 題退步；MRR 196 題改善／83 題退步。H5 第一個被接受的商品排在第一名共有 717 題，top10 沒有被接受商品共 48 題。Gold 命中 904 題，另有 48 題由 Judge 接受的替代商品補足，共 952 題。

## 授權與固定設定

使用者授權「跑 Public，只測試是否達到，不是訓練」。新一輪只讀取已採用 checkpoint，沒有 optimizer、梯度更新、負例挖掘或新增品質候選。

- checkpoint：`artifacts/training/h5_hard_negative/checkpoint`，由 Train 3,864 題續訓；H5 的完整訓練與 Dev 驗證見 H5_RESULTS.md。
- 在新 Public 檢索前核對 H5 completion 的 55 份產物／來源與 10 個 checkpoint 檔案，並凍結正式程式、權重、Public 資料與評分程式 hashes。
- 正式預設不覆寫：Dense＋BM25、RRF 5:1／常數 60、兩路各 100，無 reranker。
- Dense 保留完整角色對話；超長對話既有首尾政策不變。商品仍是一件一向量、title＋description、最多 512 tokens。
- BM25 保留完整對話、完整商品，last／title weight 都為 1。
- 原版 `retrieval.py` 完整檢索一次，保存 top50 與其 top10 前綴；沒有使用歷史 Public 排名代替本次推論。
- 原版 `evaluate.py`、`eval_utils.py` 不修改。固定本地 Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，batch8／bf16／greedy／seed0／max-new-tokens12／top10／不截斷，只評分一次。
- Public 標籤僅供評估：不進訓練／負例、不調比例／模型、不以本次結果回滾挑分。模型已於 H5 的 Train／Dev 流程採用，這次不改預設。
- Public 在 W10／H4 已看過；本次是已見過測試集上的固定配置確認，不宣稱獨立未見 holdout。
- 未開始 W11、report.pdf、download.sh 或封裝腳本。

## Gold 診斷（與正式分數分開）

| 配置 | Gold Hit@10 | Gold MRR@10 | Gold Hit@50 |
|---|---:|---:|---:|
| H4 歷史 | 0.854 | 0.6632373015873015 | 0.946 |
| H6 固定 H5 | **0.904** | **0.728475** | **0.968** |

指定 Gold 商品在 top10 淨增 50 題，在 top50 淨增 22 題。這些指標只檢查原標記商品，不能當成官方 Graded Recall／MRR。

## 品類結果（各 200 題，僅診斷）

| 品類 | H4 Recall | H6 Recall | H4 MRR | H6 MRR |
|---|---:|---:|---:|---:|
| beauty | 0.935 | 0.950 | 0.7141150794 | 0.8099166667 |
| home-and-kitchen | 0.970 | 0.975 | 0.8455317460 | 0.8484226190 |
| pet-supplies | 0.935 | 0.960 | 0.7052738095 | 0.7708650794 |
| sports-and-outdoors | 0.975 | 0.985 | 0.8064365079 | 0.8675833333 |
| toys-and-games | 0.850 | 0.890 | 0.6682202381 | 0.7152341270 |

五品類的整體兩項都高於 H4，但有逐題退步。這些結果不產生品類分流、Public 專用規則或下一輪超參數選擇。

## 檢查與資源

- 已有 grading 5 項、measurement 2 項、analysis 5 項測試通過；新測試工具 py_compile 通過。正式模型實作未改，H5 已通過的 84 項紀錄保留。
- 1,000 題順序、合法／不重複 PID、完整 75k 候選池、top10 前綴及原始輸入 hashes 通過。
- 原版 Judge 共有 **9,096 個非 Gold 比較、0 解析失敗**；完整 prompt 最長 **2,399 tokens**，context32,768／輸出12，不截斷。
- 原始 output、parse flag、每件 gain、每題 Recall／MRR、全體 aggregate 及比較數量一致。
- 與 H4 共用的 5,970 個非 Gold query–product 配對，有 54 個 gain 改變。兩次候選／組批不同，原因未隔離；保留兩次原始回答，不重跑挑較高分。單次差值不能完全視為純模型因果效果。

| 程序 | 完整冷啟動秒數 | CUDA allocated GiB | CUDA reserved GiB | RSS GiB |
|---|---:|---:|---:|---:|
| Public 檢索 | 272.101910 | 1.035655 | 1.308594 | 2.190132 |
| 原版 Public Judge | 633.660281 | 4.653175 | 9.095703 | 6.133663 |

環境：WSL Ubuntu22.04／RTX4090 24GB／Python3.12.12。檢索在 PyTorch 8GiB allocator 預算下執行，沒有模型或檢索設定覆寫，時間<1h、RSS<32GiB；此上限不含 CUDA context／其他程序，不能等同實體 RTX3070／Python3.12.9 驗收。Judge 未設 8GiB allocator cap，reserved 峰值約9.096GiB，不能宣稱 Judge 通過8GB驗收。

H5 續訓已驗過8GiB allocator，仍不代表來源H3原FP32訓練（約8.19GiB allocated）能在8GB完整重現。實體官方環境、訓練重現鏈與正式繳交檔案仍待後續獨立處理，這次沒有進入封裝。

## 重現本次測試的命令

下列命令記錄已執行程序；測量工具拒絕覆寫既有結果，請勿為了挑分重跑。

```bash
.venv/bin/python code/script/confirm_h5_public.py --stage freeze
.venv/bin/python code/script/run_validated.py --budget-gib 8 --output results/h6_h5_public_timing.json -- retrieval.py --dialogues data/public/dialogues.jsonl --products data/public/products.jsonl --top-k 50 --output results/h6_h5_public_top50.json
.venv/bin/python code/script/confirm_h5_public.py --stage prepare-judge
.venv/bin/python code/script/run_validated.py --output results/h6_h5_public_judge_timing.json -- evaluate.py --retrieval-results results/h6_h5_public_top10.json --dialogues data/public/dialogues.jsonl --products data/public/products.jsonl --model artifacts/models/Qwen2.5-1.5B-Instruct --batch-size 8 --max-new-tokens 12 --top-k 10 --device cuda --output results/h6_h5_public_graded.json
.venv/bin/python code/script/confirm_h5_public.py --stage final
```

## 保存的證據

- `results/h6_frozen_plan.json`：新 Public 前固定的設定、來源／資料與 checkpoint hashes。
- `results/h6_h5_public_top50.json`／`top10.json`／`gold.json`：完整排名與 Gold 診斷（後兩個也使用相同 `h6_h5_public_` 前綴）。
- `results/h6_judge_plan.json`：固定原版 Judge、排名及模型 hashes／不截斷檢查。
- `results/h6_h5_public_graded.json`：每題候選與原始 Judge 回答。
- `results/h6_h5_public_timing.*`／`h6_h5_public_judge_timing.*`：兩次程序的量測、worker 與完整日誌。
- `results/h6_summary.json`：稽核、逐題／品類得失、門檻逐格估算與限制。
- `results/h6_tests_{grading,measurement,analysis}.log`：12項既有測試的真實輸出。
- `results/h6_completion.json`：完成後來源、結果、報告與模型 hashes；H5既有completion保留不改。

本輪完成後停止。維持已採用 H5＋5:1，不以 Public 成績再選配置。
