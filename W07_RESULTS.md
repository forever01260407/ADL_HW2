# W07 — MiniLM 重排單獨實驗

日期：2026-10-24（Asia/Taipei）。狀態：**W07 實作、完整 dev 評估與稽核完成；A1 退步，不採為預設。**

重現提醒：本報告使用未微調的 BGE。執行下列歷史指令前，先設定 `export HW2_MODEL_PATH=artifacts/models/bge-small-en-v1.5`；不可混入後續 W08 checkpoint。

## 問題與假說

W06 的 gold Hit@10 為 0.468944、Hit@50 為 0.649068。部分正解已在前 50 名，可能透過重排進入前十；這只是實驗依據，尚未證明重排有效。容量限制與商品長文截斷是另外兩個待驗證方向，此輪保持原 BGE、全文對話、商品表示與 dense 候選生成方式。

## 實作

- `retrieval_utils.py` 僅在 student block 增加 A1。重現 B0 使用 `HW2_RETRIEVER=dense HW2_RERANK=0`，A1 使用 `HW2_RETRIEVER=dense HW2_RERANK=1`。本文記錄 W07 當時配置；現行預設以 PROGRESS.md 為準。
- 模型：`cross-encoder/ms-marco-MiniLM-L6-v2`；revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`。6 檔共 91,815,758 bytes，hash 見 `results/reranker_download.json`。既有 transformers 直接執行 sequence-classification cross-encoder，未新增套件。
- 搜尋 `min(商品數, max(top_k, 50))` 候選；以原始角色對話與每個候選商品一起評分，取 raw logit 最大的 top_k。同分維持 dense 排名。
- 每個 pair 最多 512 tokens。Query 上限 192；超長取首 64＋末 128；商品使用剩餘預算並保留前段。直接組 token IDs，不經 decode／重新編碼。這個策略仍可能截掉商品後段資訊。
- Reranker batch 16，FP32 推論；釋放 BGE GPU 模型後再載入 MiniLM。僅使用本地固定模型，不讀取 gold 標籤。
- `HW2_RERANK_CANDIDATES`、`HW2_RERANK_BATCH_SIZE`、`HW2_RERANK_MODEL_PATH` 可覆寫候選數、batch、模型位置，量測工具會記錄。
- W06 原版實作與量測工具保存在 `code/script/snapshots/w06/`，用於核對歷史結果的 source hash。

## 已完成驗證

- 22 個單元測試通過（新增 5 個重排測試；3.401 秒）。涵蓋候選隔離、跨 query 組批與分數對應、同分順序、top-k、小池、長對話最後否定、pair 預算與錯誤處理。見 `results/w07_tests.log`。
- 真模型 dev-fast：100 題、完整 50k 商品池，冷啟動 107.580 秒；其中 rerank 階段 7.206 秒，評分 5,000 pairs；query 截斷 0。
- Dev-fast gold Hit@10：B0 0.460000 → A1 0.450000；gold MRR@10：0.248540 → 0.239190。這是小型 gold-only 診斷，尚不能當成完整官方 graded 結論。

## 完整 dev 結果與決策

完整 dev 966 題／50,000 商品，使用同一原版 Qwen judge、順序、batch 8：

| 指標 | B0 語意搜尋 | A1 MiniLM 重排 | A1 − B0 |
|---|---:|---:|---:|
| 官方 graded Recall@10 | 0.652174 | 0.606625 | −0.045549（−4.55 個百分點） |
| 官方 graded MRR@10 | 0.396446 | 0.327259 | −0.069186 |
| Gold Hit@10（診斷） | 0.468944 | 0.427536 | −0.041408 |
| Gold MRR@10（診斷） | 0.292416 | 0.236438 | −0.055978 |
| 檢索完整冷啟動秒數 | 105.082 | 170.393 | +65.310 |

**決定：維持 B0 為預設。**A1 可用 `HW2_RERANK=1` 重現，但兩項主要指標均退步，且增加時間，因此本次不採用。這是對目前模型與輸入設定的結論，不代表所有 reranker 都無效。

- Recall 逐題變化：76 題改善、120 題退步、770 題不變；命中題數 630 → 586。
- MRR 逐題變化：213 題改善、322 題退步、431 題不變。
- 3 則訊息：Recall 0.615542 → 0.578732；MRR 0.349925 → 0.302510。5 則訊息：Recall 0.689727 → 0.635220；MRR 0.444137 → 0.352631。兩組都退步，不能只歸因於較長對話。
- A1 檢索 peak CUDA allocated 0.469 GiB、reserved 0.668 GiB，process RSS 約 1.774 GiB。這是開發機量測，不是 3070 速度保證。
- 原版 judge 冷啟動 595.296 秒；9,247 次比較，0 parse failures。CUDA allocated 4.555 GiB、reserved 8.754 GiB，8 GB 驗收仍待完成。
- `audit_w07.py` 實際執行通過：原始 judge output 重算一致、qid／PID 與分數對應正確、受保護檔案及 student block 邊界未改、A1 僅回傳原 dense top-50 候選、W06 來源快照 hash 與本輪輸入 hash 正確、reranker 檔案 hash 正確。Dev-fast 的 100 題在完整 run 中排名全數一致。

完整結果見 [w07_summary.json](C:/Users/yuting/Desktop/HW2_ADL/results/w07_summary.json)、[原始官方評分](C:/Users/yuting/Desktop/HW2_ADL/results/w07_a1_dev_graded.json)、[稽核紀錄](C:/Users/yuting/Desktop/HW2_ADL/results/w07_audit.log)。

檢索階段已完成：A1 gold Hit@10 0.427536、MRR@10 0.236438；冷啟動 170.393 秒，其中重排階段 66.765 秒，48,300 pairs。B0 重跑 top-50 的輸出 SHA-256 與 W06 完全相同（`3734182534d17cc36cbcdf36ab8f21317e3bd94139e6803c8fd118ce57ba2963`）。

## Pair 截斷診斷

`analyze_rerank_inputs.py` 以固定 MiniLM tokenizer 對真實候選統計，結果見 `results/w07_pair_diagnostics.json`：

- 48,300 pairs、19,502 個不同商品；商品可用預算 319–464 tokens，query 截斷 0。
- 21,756 pairs（45.04%）的商品文字超過 pair 可用預算。這是按候選出現次數計算的比例，不能直接與 W06 全商品截斷率當成相同母體比較。
- Gold 在候選集且可完整放入 pair 的 446 題：B0 命中 317 題 → A1 271 題（40 題進入前十、86 題掉出）。
- Gold 在候選集但遭 pair 截斷的 181 題：B0 命中 136 題 → A1 142 題（26 題進入、20 題掉出）。
- 因此不能把整體退步全歸因於商品截斷。兩組難度不同，沒有做同商品不同截斷的消融，這些數字也不能證明截斷有益。通用 passage reranker 與多輪商品需求的適配仍是待驗證因素。

診斷程式只計算完整 token 長度，不把超長 IDs 送入模型；其 tokenizer 的長度警告不代表推論越界。

## 案例與後續計畫

`results/w07_cases.json` 按固定 dev 順序各取前 3 題 Recall 改善／退步，沒有挑選最漂亮的結果。例如：

- `beauty-train-000304`：A1 將 gold 指甲修護商品排到第 2 名，修復 B0 的未命中。
- `beauty-train-000033`：顧客最後詢問淡化妊娠紋；A1 前三名出現眼霜／argan oil，與對話先前用過的商品相關。
- `beauty-train-000078`：顧客要 SLS-free、made in USA 的洗髮精；B0 第 1 名為 gold，A1 前三名變成對話提到過品牌的狗用洗毛精。

這些案例支持檢查「歷史商品與當前需求混淆」的假說，但沒有消融證明因果；不得只看標題替官方 judge 補理由。單純放大模型或增加 chunk 不能保證修復這類功能／對象錯誤。

下一個優先實驗排為 **H1：BM25-only 與 Dense + BM25 RRF**，先固定完整 query、停用 A1，檢驗容量／型號等詞面資訊能否補足 B0。PLAN 已記錄分詞、小數保留、固定初始參數與比較流程。若歷史詞面干擾持續，另做 QRY1 對話表示比較，避免與 hybrid 同時修改。商品少量分段與 A2 訓練仍分開評估；若課程確認訓練必做，A2 優先。H1／QRY1 尚未實作或產生分數。

## 重現方式

在 WSL Ubuntu 22.04、專案根目錄執行，使用新的輸出路徑以保留歷史紀錄：

```bash
.venv/bin/python code/script/prepare_model.py --role reranker
.venv/bin/python -m unittest discover -s code/script/tests -v

HW2_RETRIEVER=dense HW2_RERANK=1 .venv/bin/python code/script/run_measured.py \
  --output results/recheck_a1/dev_timing.json -- retrieval.py \
  --dialogues data/derived/dev.jsonl --products data/train/products.jsonl \
  --output results/recheck_a1/dev_top10.json

.venv/bin/python code/script/analyze_results.py \
  --retrieval-results results/recheck_a1/dev_top10.json \
  --dialogues data/derived/dev.jsonl --output results/recheck_a1/dev_gold.json

.venv/bin/python code/script/run_measured.py \
  --output results/recheck_a1/dev_judge_timing.json -- evaluate.py \
  --retrieval-results results/recheck_a1/dev_top10.json \
  --dialogues data/derived/dev.jsonl --products data/train/products.jsonl \
  --model artifacts/models/Qwen2.5-1.5B-Instruct \
  --batch-size 8 --max-new-tokens 12 --device cuda \
  --output results/recheck_a1/dev_graded.json
```

本輪保存結果的稽核命令是 `.venv/bin/python code/script/audit_w07.py`，它讀取 `results/w07_*`／`results/w06_*` 固定實驗檔；`recheck_a1` 為獨立重跑產物，沒有冒充原始 run。

## 限制

開發環境為 RTX 4090／Python 3.12.12；官方 RTX 3070 8 GB／Python 3.12.9 尚未驗收。W06 已觀察到 judge 對組批敏感；本輪沿用完整 dev 順序與 batch 8，但排名改變也會改變候選組批。未跑 public、未訓練模型、未完成 Q2 與最終提交。
