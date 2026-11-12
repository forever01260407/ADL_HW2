# W06 — B0 Dev 評估與冷啟動紀錄

日期：2026-10-24（Asia/Taipei）。狀態：**W06 開發環境評估與稽核完成；官方 3070 8 GB／public 驗收仍未完成。**

重現提醒：本報告使用未微調的 BGE。執行下列歷史指令前，先設定 `export HW2_MODEL_PATH=artifacts/models/bge-small-en-v1.5`；不可混入後續 W08 checkpoint。

本階段只量測既有 B0，沒有改 embedding 模型、query 策略或排序，也沒有訓練、使用 public 答案、啟用 reranker。所有數值均來自實際執行。機器可讀總表見 [w06_summary.json](C:/Users/yuting/Desktop/HW2_ADL/results/w06_summary.json)。

## 1. 評估設計

- 使用 W02 固定分組 split：完整 dev 966 題；dev-fast 100 題是它的子集，不是另一份獨立測試集。
- 每次檢索都使用完整 **50,000 商品**，從新程序載入模型、重新編碼商品與建立索引；不使用持久 index cache。
- B0：BGE-small、384 維、512-token 上限、完整角色對話、batch 32、正規化向量、CPU FAISS 精確內積搜尋。同分依 PID 排序。
- 另跑 top-50 作候選診斷，與獨立 top-10 run 比較前十名；正式 judge 仍只評 top-10。
- 使用原版 `evaluate.py`，指定本地 Qwen2.5-1.5B-Instruct 權重，**batch 8、bf16、greedy、max_new_tokens 12、不截斷**。
- 外層計時包含 Python 子程序啟動、imports、讀資料／權重、完整建索引、搜尋和寫檔。測量 wrapper 只用 `runpy` 執行原入口並收集記憶體，沒有 patch 官方函式。OS 檔案快取沒有清除，因此這是「新程序／重建索引」的冷啟動，不宣稱磁碟快取全冷。

## 2. 品質結果

| 資料 | 題數 | Gold Hit@10 | Gold MRR@10 | 官方 graded Recall@10 | 官方 graded MRR@10 |
|---|---:|---:|---:|---:|---:|
| Dev-fast | 100 | 0.460000 | 0.248540 | 0.680000 | 0.375484 |
| 完整 dev | 966 | 0.468944 | 0.292416 | **0.652174** | **0.396446** |

Gold-only 只對資料標註的唯一商品；官方 graded 也接受 Qwen 認可的替代品。不能把兩者混用，也不能把 dev 指標直接宣稱為 public 的及格／級距結果。

完整 dev 有 **453 題命中 gold**，另有 **177 題透過替代品命中**，合計 630／966 題至少有一個可接受商品；336 題的前十名未被判為可接受。9,207 次 LLM 比較的 **parse failures = 0**。

| Dev 訊息數切片 | 題數 | Graded Recall@10 | Graded MRR@10 |
|---|---:|---:|---:|
| 3 則訊息 | 489 | 0.615542 | 0.349925 |
| 5 則訊息 | 477 | 0.689727 | 0.444137 |

這是不同樣本群的描述統計，不能解讀成「增加對話訊息必定提高準確率」。

完整 dev 的 **Gold Hit@50 = 0.649068**，比 Hit@10 高約 **18.01 個百分點**。兩次獨立檢索的前十名在 966 題中全數一致。這說明部分 gold 在第 11–50 名，但未證明任一 reranker 能把它排進前十；也不能把 gold Hit@50 當成 graded 指標的上限。

## 3. 時間與資源

| 執行 | 新程序總時間 | GPU peak allocated | GPU peak reserved |
|---|---:|---:|---:|
| Dev-fast 檢索／50k 商品 | 112.305 秒 | 0.469 GiB | 0.668 GiB |
| 完整 dev 檢索／50k 商品 | 105.082 秒 | 0.469 GiB | 0.668 GiB |
| 完整 dev top-50 診斷 | 101.870 秒 | 詳見量測 JSON | 詳見量測 JSON |
| Dev-fast 官方 judge | 88.149 秒 | 4.261 GiB | 5.898 GiB |
| 完整 dev 官方 judge | 583.114 秒（約 9 分 43 秒） | 4.554 GiB | **8.631 GiB** |

完整 dev 檢索的程序 RSS 峰值約 **1.665 GiB**。樣本較少的程序反而稍慢，不能據此認定 query 數增加會加速；imports、檔案快取與背景負載會影響單次時間。

**8 GB 風險：**完整 judge 的 reserved 峰值超過 8 GiB。Reserved 包含 PyTorch 保留的快取，不等於活躍張量需求；allocated 峰值為 4.554 GiB。這既不能直接證明 3070 會 OOM，也不能當成已通過 8 GB 驗收。須在相容環境另測；本次沒有減 batch、截斷 prompt 或修改官方 judge 來掩蓋問題。

開發硬體是 **RTX 4090／WSL Ubuntu 22.04／Python 3.12.12**，不同於指定 RTX 3070 8 GB／Python 3.12.9。量測期間整張卡約有 19–20 GiB 用量，無法完整辨識其來源；本程序的 PyTorch 峰值另外記錄。以上是當時的實測，不是乾淨硬體 benchmark，也不是官方 3070 驗收。Public 的一小時限制仍須在 W10 驗證。

## 4. Token 與相容性發現

- **商品：**50,000 筆中 13,488 筆（26.976%）超過 512 tokens；P50 387、P95 692、最長 2,746。B0 會截掉商品文字後段，可能遺失規格，但本次沒有做消融實驗，不能斷言它造成多少失分。
- **Dev 對話：**包含 query instruction 後最長 200 tokens，966 筆全數未截斷；目前 dev 失分不能歸因於對話超過 512 tokens。
- **官方 judge prompt：**完整 dev 共有 9,207 個非 gold 比較，最長 2,265 tokens；加 12 個輸出 tokens 仍在指定 Qwen 的 context 範圍內。沒有為了測試而刪掉長題。
- **Qwen smoke：**first-8 及 dev-fast 最長 8 個比較都用原版 judge 成功執行，16 個輸出均可解析；dev-fast 與完整 dev 的正式評分也都沒有解析失敗。

### 重現性的實際界線

Dev-fast 的 100 題，獨立檢索與完整 dev 檢索的 top-10 全數相同；完整 dev 的 top-50 前綴也與 top-10 全數一致。但在 dev-fast／完整 dev 的 **不同 batch 搭配與順序**下，100 題中有 4 題的 graded Recall 或 MRR 不同。這不是 reranker 或檢索策略的改善，也不能宣稱相同問題在任意 batch 下評分完全相同；確切原因尚未做隔離實驗。兩份原始結果均保留，完整 dev run 作為後續主要比較基準，後續配置須使用相同資料順序與 batch 8。

## 5. 固定選樣的錯誤觀察

保存 dev-fast 排序中「前 3 個 graded 完全失敗」及「第一個 gold miss、graded 命中」案例，未挑選最漂亮的結果。原文與商品描述見 [案例 JSON](C:/Users/yuting/Desktop/HW2_ADL/results/w06_dev_fast_cases.json)。以下是人工觀察，不是另一套 relevance 標註：

- `beauty-train-000193`：顧客明確要 **1.6 oz** 香水，前幾名卻是 **2.5／1.7 oz**。語意上找對大類仍可能漏掉數量／尺寸限制。
- `beauty-train-002624`：對話提及過去的 lavender face brush，前面推薦含 lavender cleanser／wipes。這與「歷史文字干擾目前需求」的假說相符，但還沒有對話消融證明原因。
- `beauty-train-003071`：前幾名都是名稱相似的 nail image plates，官方 judge 卻全判不相關。不能只看商品標題就替裁判補出理由，也不能自行把這些判定改成正例；這類案例需閱讀完整規格，並留意小型 judge 的判斷限制。
- `beauty-train-001206`：gold 沒在前十，但第二名紅色染髮產品被接受；這展示 gold-only 漏算合理替代品的情況。

## 6. 如何重跑

以下是目前開發環境的命令，不取代 W11 最終提交的 README／run.sh。使用 **WSL Ubuntu 22.04** 與專案 `.venv`；它依賴既有 ADL-final 環境，不能當成可直接搬到助教電腦的獨立環境。

```bash
cd /mnt/c/Users/yuting/Desktop/HW2_ADL

# 測試分析工具與既有行為
.venv/bin/python -m unittest discover -s code/script/tests -v

# 官方檢索入口，另取檔名以保留先前實驗
HW2_RETRIEVER=dense HW2_RERANK=0 .venv/bin/python code/script/run_measured.py \
  --output results/recheck/dev_timing.json -- retrieval.py \
  --dialogues data/derived/dev.jsonl --products data/train/products.jsonl \
  --output results/recheck/dev_top10.json

# 便宜的 gold-only 診斷
.venv/bin/python code/script/analyze_results.py \
  --retrieval-results results/recheck/dev_top10.json \
  --dialogues data/derived/dev.jsonl --output results/recheck/dev_gold.json

# 原版官方評分；與 retrieval 分開計時、分開使用 GPU
.venv/bin/python code/script/run_measured.py \
  --output results/recheck/dev_judge_timing.json -- evaluate.py \
  --retrieval-results results/recheck/dev_top10.json \
  --dialogues data/derived/dev.jsonl --products data/train/products.jsonl \
  --model artifacts/models/Qwen2.5-1.5B-Instruct \
  --batch-size 8 --max-new-tokens 12 --device cuda \
  --output results/recheck/dev_graded.json
```

`run_measured.py` 強制離線推論並保留 log、輸入／程式 hash、環境與資源；量測檔已存在時會拒絕覆寫。模型 revision 與檔案 hash 分別保存在 [模型鎖定檔](C:/Users/yuting/Desktop/HW2_ADL/code/script/configs/models.lock.json)、`results/embedding_download.json`、`results/generator_download.json`。

## 7. 驗收與下一步

17 個單元測試通過；`audit_w06.py` 的實際稽核也已通過：raw output 能重算出相同的逐題 gain、Recall／MRR 與解析失敗數，排名與 qid 對應正確，輸入及程式 hash 吻合。三個受保護官方檔案維持原版。

主要結果：[完整 dev 官方評分](C:/Users/yuting/Desktop/HW2_ADL/results/w06_b0_dev_graded.json)、[完整 dev 檢索量測](C:/Users/yuting/Desktop/HW2_ADL/results/w06_b0_dev_timing.json)、[完整 dev judge 量測](C:/Users/yuting/Desktop/HW2_ADL/results/w06_b0_dev_judge_timing.json)、[稽核輸出](C:/Users/yuting/Desktop/HW2_ADL/results/w06_audit.log)。

W07／W08 仍未啟動。下一個實驗應針對已觀測到的規格限制、候選召回與排序問題，單獨比較一種方法；不能因為有新技術就同時加多個模型。是否必須訓練仍依 PROJECT 的未解課程問題處理。Public、Q2 生成、最終報告與封裝留在後續工作。
