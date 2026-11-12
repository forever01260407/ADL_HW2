# W10 — 凍結 W08 的 public 評分與配分估算

日期：2026-10-24（Asia/Taipei）。狀態：**完整 public 檢索、一次原版 Judge 與稽核完成；兩項達 Strong，public 累加估算 6／8 個課程百分點。官方硬體與作業總分仍未確認。**

## 本輪範圍與規則

使用者要求實際跑投影片 Grading 對應評分，判斷目前分數，若不足再評估規則允許的改進。凍結目前 W08：**自行微調的 BGE-small＋BM25＋等權 RRF**，不是 BM25-only。對話、商品處理、兩路各 100、RRF 60、BM25 full／1／1、reranker 關閉均保持不變。

先完成完整 public 1,000 題／75,000 商品，保存正式入口的 top-50 診斷及其 top-10 前綴，再用原版 evaluate.py、指定 Qwen revision、batch 8、greedy／12 new tokens、不截斷，執行此凍結配置的一次 Judge。沒有將 public target 傳給檢索函式，也沒有使用 public 作梯度、負例或逐題規則。

版本、資料／模型／來源 hash 先鎖於 `results/w10_frozen_plan.json`。W09 被更新文件的舊版也保存於 `code/script/snapshots/w09_documents/`，供其歷史摘要 hashes 核對。後續若基於 public 回饋修改策略，必須記錄 public 已看過，只用 train／dev 篩選候選，不能再把後續 public 稱為未見過的獨立測試。

## 配分如何解讀

來源是使用者提供的 Grading 圖，以及本地 `ADL2026-HW2.pptx.pdf` p.19／`PROJECT.md` §8.4。以下是**課程百分點**，本作業各項合計 20；不是直接把 Recall 百分比當成作業成績。

| 級距 | Public Recall@10 | Public MRR@10 | Private Recall@10 | Private MRR@10 |
|---|---:|---:|---:|---:|
| Simple | 0.860（2） | 0.650（2） | 0.845（1） | 0.635（1） |
| Strong | 0.885（1） | 0.680（1） | 0.880（1） | 0.675（1） |
| Boss | 0.950（1） | 0.800（1） | 0.930（1） | 0.780（1） |

Public 共 8、private 共 6、報告共 6（Q1 4＋Q2 2），模型共 14。依各格累加作估算：例如 public Recall 0.90、MRR 0.70，兩者均過 simple／strong，估計拿 public 6／8。**累加細則及恰等於門檻如何算，文件沒有額外明文；此工具列出含等號與嚴格大於的兩種估算，不保證官方最後給分。**使用原始未四捨五入的指標，Recall 與 MRR 分開判斷。

Private 不可取得，報告也尚未完成／評分；不拿 public 數字套 private 門檻、不預給報告分，也不把局部結果包裝成整份作業總分。

## 已完成的驗證

- **59 tests／6.117 秒通過**（`results/w10_tests.log`）。新增門檻獨立計分、累加、public／private 權重、相等邊界與非有限／非法指標測試；既有功能也通過。
- W08 checkpoint、官方三檔、正式推論來源與 public 資料 hash 一致，無環境設定覆寫模型／檢索策略。
- 完整 public 檢索冷啟動 **163.892 秒**（開發 4090），包含程序啟動、載入、75k 商品重新編碼、建索引、搜尋與寫檔；無預存商品索引。1,000 題 top-50 合法，其 top-10 前綴一致。
- 檢索 CUDA peak allocated **0.469 GiB**／reserved **0.668 GiB**，process peak RSS **1.867 GiB**；未把整張顯卡的其他程序占用當成本程序用量。
- Gold-only Hit@10 **0.819**／MRR@10 **0.615437**／Hit@50 **0.928**；這是診斷，不是官方 graded 配分依據。
- 原版 Judge 有 9,181 個非 gold 比較；最長實際 input 2,399 tokens，低於 context 32,768 扣除 12 output 的預算，沒有做截斷。只有一次初始 public 評分被授權。

## 官方結果與配分

使用凍結 W08、完整 1,000 題／75k 商品、本地原版 evaluator：

| 指標 | 實測 | Simple | Strong | Boss |
|---|---:|---:|---:|---:|
| graded Recall@10 | **0.902000（902／1000 題）** | 0.860，通過 | 0.885，通過 | 0.950，未達 |
| graded MRR@10 | **0.7043484126984128** | 0.650，通過 | 0.680，通過 | 0.800，未達 |

| Public 配分格 | Recall 估算 | MRR 估算 | 小計 |
|---|---:|---:|---:|
| Simple | 2 | 2 | 4 |
| Strong | 1 | 1 | 2 |
| Boss | 0 | 0 | 0 |
| **合計** | **3** | **3** | **6／8** |

本次數值都不在門檻邊界，嚴格大於／包含等號的估算均為 6。距離 Boss Recall 尚差 0.048（在 1,000 題下相當於多命中 48 題），MRR 尚差 0.0956516；不能把多命中 48 題自動當成 MRR 也會達標。

**目前可估的是 public 6／8，並非整份作業 75 分。**若以本作業 20 個課程百分點換成作業 100 分，public 這塊最多 40 分，本次相當其中 30 分；private 另占最多 30、報告另占最多 30，兩者仍未知。尚有格式／重現等規範與扣分可能，不能直接保證已獲得的總分，也不能先假設報告滿分。

這是同一個 W08 的首次 public 評測，沒有新訓練。Public 與先前 966 題 dev 的組成、商品池不同，不能直接把 public 90.2% 減 dev 82.92% 當成本輪新改進。

### 原始結果與資源核對

9,181 個非 gold 比較、**0 parse failures**，僅一次初始 public Judge。819 題命中 gold，另 83 題由合理替代品命中，共 902 題；原始文字解析、gain、逐題指標、aggregate、排名、query 順序、資料與來源 hashes 均核對通過。Judge 冷啟動 **664.868 秒**，CUDA peak allocated **4.653 GiB**／reserved **9.875 GiB**。Reserved 包含 cache，不能據此判定 3070 必定 OOM，也不能宣稱已通過實卡驗收。

| Public 來源分組 | 題數 | graded Recall@10 | graded MRR@10 |
|---|---:|---:|---:|
| beauty | 200 | 0.935 | 0.733143 |
| home-and-kitchen | 200 | 0.940 | 0.762476 |
| pet-supplies | 200 | 0.875 | 0.679976 |
| sports-and-outdoors | 200 | 0.935 | 0.777986 |
| toys-and-games | 200 | 0.825 | 0.568161 |

分組用於理解差異，不依 public qid／PID 做推論分流或例外規則，也不使用其 target 訓練／挖負例。後續 public 已屬看過的回饋資料。

### 本輪決定

**維持 W08，不啟動新模型或第二次 public。**已達 Strong，並非連基本門檻都過不了的低分情況；報告 6 個課程百分點尚未交付，先完成官方環境與報告／封裝更有優先性。若之後明確要衝 Boss，再依下一節在 train／dev 做獨立實驗；不因較大模型存在就先假設它會更準。

## 若分數不足，如何合規改善

先確保 public 的 Recall／MRR 都達 simple；若已達 strong，優先完成正式環境與繳交，避免為追求較大模型擠壓報告／封裝時間。此為開發優先序，不是老師新增的門檻。

1. 先在同一 train／dev 比較微調後 Dense-only 與現有 Hybrid，檢驗未微調時有用的 BM25 是否仍補足新的 encoder，或反而拉低排序；維持資料與模型不變。必要時下一個獨立消融才是 RRF 兩路權重，不把「已是 Hybrid」誤當成尚未做。
2. 若小型模型仍有明確瓶頸，再考慮同系列較大的 `BAAI/bge-base-en-v1.5`。它是可微調的 embedding 候選，不保證優於已微調的 small；需要重新訓練／全商品編碼，先以 train／dev 選型並實測官方 8 GB／時間／磁碟預算。[官方模型卡](https://huggingface.co/BAAI/bge-base-en-v1.5)
3. 可以改的是作業未限定的 retriever／embedding／選用 reranker；**不能為提高成績替換官方 Judge，也不能把 Q2 指定 Qwen 任意換大**。不使用被禁止的外部資料、其他同學／作業解法模型、public 梯度或 private 資料。

以上是後續若追求 Boss 的候選次序；本輪結果已達 Strong，沒有啟動新模型下載、訓練或第二個 public 實驗。整份作業仍須完成 private 的正式評分、官方硬體／版本驗收、報告與封裝，不能把這次估算當作助教最終總分。

## 資源與限制

本機是 WSL Ubuntu 22.04／Python 3.12.12／RTX 4090 24 GB；官方為 Ubuntu 22.04／Python 3.12.9／RTX 3070 8 GB。本機時間低於一小時不能冒充官方硬體完整驗收，allocated／reserved 也須分別記錄。截止後模型連結不得改內容、需維持三週有效，以及報告／download.sh／正確封裝等要求仍待 W11 完成。

## 程式與產物

- `code/script/audit_w10.py`：凍結核對、以實際 public 商品做原版 Judge token 預檢、raw output／gain／aggregate 稽核與逐格配分估算。
- `results/w10_frozen_plan.json`／`w10_judge_plan.json`：初始凍結與一次性 Judge 計畫。
- `results/w10_w08_public_top50.json`／`top10.json`／`gold.json`：完整檢索與 Gold 診斷。
- `results/w10_w08_public_timing.json`／`judge_timing.json`／`graded.json`：正式入口原始時間／資源與官方判斷。
- `results/w10_summary.json`：已完成的原始結果稽核、逐格配分、資源及限制摘要。

重現使用未占用的新輸出名稱，不覆寫本次紀錄；正式命令為 `retrieval.py --dialogues data/public/dialogues.jsonl --products data/public/products.jsonl`，Judge 使用 `evaluate.py --model artifacts/models/Qwen2.5-1.5B-Instruct --batch-size 8 --max-new-tokens 12 --top-k 10` 並傳入相同 public 檔及 top-10 排名。完整實際命令保留於 timing JSON。
