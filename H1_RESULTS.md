# H1 — BM25 與語意搜尋融合

日期：2026-10-24（Asia/Taipei）。狀態：**H1 完成；Hybrid 兩項指標改善，已設為開發版預設，完整 dev 預設一致性驗證通過。**

重現提醒：上述採用狀態為 H1 當時的決策，最新選型見 PROGRESS.md。本報告使用未微調的 BGE；執行下列歷史指令前，先設定 `export HW2_MODEL_PATH=artifacts/models/bge-small-en-v1.5`，不可混入後續 W08 checkpoint。

## 實驗目的

檢驗容量、型號等詞面資訊是否能補足 B0 的語意搜尋。W07 的 MiniLM 重排已退步，因此本輪停用重排，保持相同完整角色對話，不同時改摘要、需求改寫、embedding 或商品分段。

## 實作與固定設定

- `HW2_RETRIEVER=dense|bm25|hybrid`；評估期間預設維持 dense，評估後已選定 **hybrid 為開發版預設**，`HW2_RERANK=0`。可明確指定 dense 重現 B0。
- BM25 以標準函式庫與既有 NumPy 建立倒排索引；不安裝套件，不使用額外資料或訓練標籤。BM25-only 不載入 embedding 模型、不使用 GPU。
- 商品：完整標題＋描述。大小寫統一，用 Unicode 字母／數字分詞，保留小數與連字號型號；不移除否定詞、不做 stemming。`1.6` 與 `1.7` 為不同詞，沒有將容量作硬篩選。
- Query 使用完整角色對話；每個不同詞項計一次，避免重複出現次數直接增加權重。它仍不具備明確的否定、歷史／當前需求理解。
- BM25 固定 k1=1.5、b=0.75；IDF 為 `log(1+(N-df+0.5)/(df+0.5))`，長度正規化採實際詞數平均。此 IDF 可對照 [Lucene 官方文件](https://lucene.apache.org/core/9_12_1/core/org/apache/lucene/search/similarities/BM25Similarity.html)；k1=1.5 是本次起始值，不是聲稱 Lucene 預設值。
- Hybrid：BGE 語意搜尋與 BM25 各取前 100，RRF 為兩路 `1/(60+rank)` 相加，等權重；同分依 PID 排序。BM25 零分候選不投票，零詞面命中時保留 dense 排名。[RRF 官方說明](https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/rrf/)
- 預設候選數可用 `HW2_RRF_CANDIDATES` 覆寫；本輪固定 100、不掃參數。
- BM25 使用完整商品文字，BGE 仍有 512-token 上限。因此本輪是系統方案比較，不能把差異全部歸因於搜尋公式。

## 測試與 smoke

新增獨立 BM25 公式 oracle、RRF 排名、query 重複詞政策、PID 增刪與換序、小數／型號／否定、長文尾端、零命中、同分、小池與錯誤設定測試。另驗證歷史來源可由精確快照核對，但竄改資料或快照必須失敗。**共 28 個測試通過，3.617 秒**，見 `results/h1_tests_final.log`。

Dev-fast 100 題／完整 50k 商品池，以下僅為 gold-only 診斷：

| 方法 | Gold Hit@10 | Gold MRR@10 |
|---|---:|---:|
| B0 | 0.460000 | 0.248540 |
| BM25-only | 0.470000 | 0.352262 |
| Hybrid RRF | 0.570000 | 0.305381 |

## 完整 dev 結果與採用決策

完整 dev 966 題／50,000 商品，固定原版 Qwen judge、同一資料順序、batch 8：

| 方法 | 官方 graded Recall@10 | 官方 graded MRR@10 | 命中題數 |
|---|---:|---:|---:|
| B0 Dense | 0.652174 | 0.396446 | 630／966 |
| BM25-only | 0.650104 | 0.450887 | 628／966 |
| **Hybrid RRF** | **0.739130** | **0.465780** | **714／966** |

**採用 Hybrid 為下一開發候選與程式預設。**相較 B0，Recall 提高 **8.70 個百分點**，MRR 提高 **0.069334**，淨增加 84 題命中。BM25-only 的排序品質提高，但 Recall 未提高，依事先定義的政策不選為預設。

- Hybrid Recall：132 題改善、48 題退步、786 題不變；MRR：336 題改善、166 題退步、464 題不變。
- 3 則訊息：Recall 0.615542 → 0.703476；MRR 0.349925 → 0.411342。
- 5 則訊息：Recall 0.689727 → 0.775681；MRR 0.444137 → 0.521587。兩組皆改善，但不是對話較長造成進步的因果證明。
- Hybrid judge：9,121 次比較、0 parse failures，完整冷啟動 598.660 秒；CUDA allocated 峰值 4.569 GiB、reserved 9.719 GiB。檢索程序 peak allocated 0.469 GiB、reserved 0.668 GiB，RSS 約 1.678 GiB。
- 完整稽核已實際通過：原始 judge output、逐題 gain 與 aggregate 一致；官方檔案與 student block 邊界正確；B0 全 dev top-50 與 W06 完全一致；兩個 H1 方案的 top-50 前綴及 dev-fast／完整 dev 排名均一致；實驗輸入與目前／歷史 source hash 可核對。
- 模型、query、分詞與融合設定均未因看到評分而調參。選定後只更改預設 mode，並將舊的 `dense_s` log 名稱釐清為包含詞面索引時間的 `pre_rerank_s`；被評估的 source 精確保存在 `code/script/snapshots/h1/`。
- **預設一致性驗證通過：**不設定 `HW2_RETRIEVER`、`HW2_RRF_CANDIDATES` 的正式入口，完整 966 題排名與被評估的 Hybrid 完全相同，冷啟動 112.755 秒；28 個測試再次通過（3.722 秒）。最終稽核 `selected_default_verified=true`，見 `results/h1_audit_final.log`／`h1_tests_promoted.log`。

原始結果：[機器可讀總表](C:/Users/yuting/Desktop/HW2_ADL/results/h1_summary.json)、[Hybrid 官方原始評分](C:/Users/yuting/Desktop/HW2_ADL/results/h1_hybrid_dev_graded.json)、[固定選樣的改善／退步案例](C:/Users/yuting/Desktop/HW2_ADL/results/h1_cases.json)。

BM25-only 原版官方評分已完成：Recall@10 **0.650104**、MRR@10 **0.450887**；相較 B0，Recall 少 2 題命中、MRR 增加約 0.054441。這是取捨，尚未符合兩項皆改善的採用政策。9,182 次 LLM 比較、0 parse failures；judge 冷啟動 610.263 秒，CUDA allocated 峰值 4.568 GiB、reserved 12.943 GiB。Reserved 包含 allocator cache，不能單憑它判定 8 GB 成敗；正式環境仍需驗收。

已完成的 BM25-only 診斷：gold Hit@10 0.494824、MRR@10 0.362346、Hit@50 0.627329；冷啟動 top-10 檢索 9.576 秒。獨立 top-50 的前十名與 top-10 在 966 題全數一致。

完整 dev 的 gold-only 診斷已完成，尚未替代官方評分：

| 方法 | Gold Hit@10 | Gold MRR@10 | Gold Hit@50 | Top-10 冷啟動秒數 |
|---|---:|---:|---:|---:|
| B0 | 0.468944 | 0.292416 | 0.649068 | 105.082 |
| BM25-only | 0.494824 | 0.362346 | 0.627329 | 9.576 |
| Hybrid RRF | 0.557971 | 0.354759 | 0.703934 | 105.499 |

BM25 與 Hybrid 各自的獨立 top-50 前十名，都與其 top-10 run 在 966 題完全相同。上述耗時是開發機觀測值，其他系統負載與 OS cache 未清除，不是隔離硬體 benchmark。

採用政策：兩項完整官方 graded 指標皆高於 B0 的配置才可提升為下一開發候選；若多個符合，先比較 Recall，再比較 MRR。提升預設前保存實驗 source 快照，再用未設定 `HW2_RETRIEVER` 的正式入口驗證完整 dev 排名一致。選型依據是 dev，不能視為 public 成績。

此政策已選出 Hybrid，正式預設輸出的最後一致性驗證也已實際完成。可以不設定 mode 直接使用目前 Hybrid，或以 `HW2_RETRIEVER=dense`／`bm25` 重現其他方案。

## 已觀測的限制

Gold-only 改善並非所有來源一致。按 qid 前綴分組，`beauty-train` 的 437 題：B0 Hit@10 0.375286、BM25 0.224256、Hybrid 0.350114；其餘五個前綴的 Hybrid Hit@10 均提高。這是來源分組觀察，不能據此前綴設計推論時分流或宣稱所有商品類別都改善。

同一 `beauty-train` 組的**官方 graded** Recall 0.615561 → 0.633867、MRR 0.348768 → 0.359200，仍有改善；gold-only 的下降並不等於所有替代商品都不合適。兩套指標與全部分組都保留在總表，沒有只呈現較好的一套。

固定舊錯誤案例的檢查也顯示：`beauty-train-000078` 的歷史品牌仍讓 BM25／Hybrid 推出狗用洗毛精；`beauty-train-000193` 的 1.6 oz 香水仍未命中。分詞保留完整複合詞，`1.6-Ounce` 與 `1.6 oz` 尚未做單位／子詞正規化；不宣稱加入 BM25 就解決容量限制。這些問題留待獨立的對話表示或分詞比較，沒有在本輪看完結果後偷偷修改設定。

## 重現

在 WSL Ubuntu 22.04、專案根目錄，使用新的輸出目錄保留實驗歷史：

```bash
.venv/bin/python -m unittest discover -s code/script/tests -v

# 將 hybrid 換成 bm25 或 dense 即可重跑不同方案。
HW2_RETRIEVER=hybrid HW2_RERANK=0 \
  .venv/bin/python code/script/run_measured.py \
  --output results/recheck_h1/dev_timing.json -- retrieval.py \
  --dialogues data/derived/dev.jsonl --products data/train/products.jsonl \
  --output results/recheck_h1/dev_top10.json

.venv/bin/python code/script/run_measured.py \
  --output results/recheck_h1/dev_judge_timing.json -- evaluate.py \
  --retrieval-results results/recheck_h1/dev_top10.json \
  --dialogues data/derived/dev.jsonl --products data/train/products.jsonl \
  --model artifacts/models/Qwen2.5-1.5B-Instruct \
  --batch-size 8 --max-new-tokens 12 --device cuda \
  --output results/recheck_h1/dev_graded.json
```

本輪原始結果的稽核為 `.venv/bin/python code/script/audit_h1.py`，讀取固定 `results/h1_*`／`w06_*`；`recheck_h1` 是另一次實驗，不冒充原始執行。歷史 source 只允許對照 `code/script/snapshots/` 中指定階段的精確 hash，資料 hash 不允許繞過。

## 限制與後續

開發機為 RTX 4090／Python 3.12.12，官方 RTX 3070 8 GB／Python 3.12.9 尚未驗收。官方 judge 固定同一模型、dev 順序與 batch 8，但候選變動會改變組批；W06 曾觀察到組批敏感性。保留原始輸出，不將單次 dev 比較當成未見資料保證。

本輪未訓練、未跑 public、未加入對話改寫或分段。後續依錯誤分布決定 QRY1 或 A2；若課程確認訓練必做，A2 優先。Q2、正式硬體／public 驗收與提交封裝仍待後續完成。
