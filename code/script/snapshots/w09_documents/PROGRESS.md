# PROGRESS.md — Phase 3 Implementation

更新日期：2026-10-24（Asia/Taipei；H1U 完成，維持 H1 預設）

依據：[PLAN.md](C:/Users/yuting/Desktop/HW2_ADL/PLAN.md)、[DECISIONS.md](C:/Users/yuting/Desktop/HW2_ADL/DECISIONS.md)。每個模組先實作、執行適當測試，再往下推進。測試替身通過不代表真模型品質或官方硬體驗收通過。

## Completed Tasks

- **W00 完成（開發環境盤點）**：已核對既有規範、starter 邊界及 WSL／GPU／套件實際載入結果。紀錄：[environment_w00.json](C:/Users/yuting/Desktop/HW2_ADL/results/environment_w00.json)。課程未解問題仍未宣稱已確認。
- **W01 完成**：BGE-small 固定 commit，10 個模型檔及 SHA-256 已記錄；專案 `.venv` 真實離線載入與 FAISS exact smoke 通過。Qwen 的首次相容性測試已於 W06 補完；Q2 生成流程仍留在 W09。
- **W02 完成**：4,546 個 connected groups → train 3,864／dev 966；dev-fast 100（3／5 turns 各 50）、固定 Q2 5 題，候選池仍完整 50,000 商品。分割 manifest 包含原始與衍生檔 hash；12 商品／8 對話手寫 fixtures 與訓練隔離。
- **W03 完成**：student block 以固定 PID 順序建完整索引；FAISS exact 與 NumPy fallback、邊界同分次排序及清楚錯誤處理均已測試。
- **W04 完成**：原始短對話逐字保留，instruction 僅加在 query；超長 query 在 tokenizer 預算內保留首尾，並印出截斷筆數。未宣稱保留被省略中段的語意。
- **W05 完成（2026-10-24 補核實）**：12 個單元測試通過；官方 `retrieval.py` 以真實 BGE 模型離線執行兩次小池 smoke，8 題各回傳 10 個唯一合法 PID，兩次排名相同。先前程序已成功結束（exit code 0）；使用上限中斷後尚未回寫進度，現在依既存輸出補記。
- **W06 完成（開發環境）**：完成 dev-fast／完整 dev 的全 50k 商品池檢索、gold-only／官方 graded 評分、top-50 候選診斷、token 統計、Qwen smoke、完整資源量測與原始結果稽核。新增工具、17 個測試與原始 log 均已保存。完整說明：[W06_RESULTS.md](C:/Users/yuting/Desktop/HW2_ADL/W06_RESULTS.md)。
- **W07 完成（開發環境；A1 不採用）**：可切換的 MiniLM top-50 重排已實作，22 個單元測試通過；dev-fast／完整 dev、B0 回歸、原版 judge、逐題比較、pair 截斷診斷與來源稽核完成。A1 官方 Recall／MRR 均退步，維持 B0 預設。完整說明：[W07_RESULTS.md](C:/Users/yuting/Desktop/HW2_ADL/W07_RESULTS.md)。
- **H1 完成（開發環境；採用 Hybrid）**：BM25-only／Dense + BM25 RRF 已實作；完整 dev 的原版 graded Recall@10 **0.739130**、MRR@10 **0.465780**，均高於 B0。28 個測試、原始結果與 source hash 稽核、B0 回歸及切換預設後的完整 dev 排名一致性驗證均通過。現行預設 `hybrid`，MiniLM 關閉；結果：[H1_RESULTS.md](C:/Users/yuting/Desktop/HW2_ADL/H1_RESULTS.md)。
- **H1U 完成（BM25 表示單一變因消融）**：37 個測試通過；四個配置的完整 dev Gold 與 beauty-train 分組、逐題得失已保存；U0 完全重現 H1。僅 U2（users＋末輪 2 倍）通過 Gold 門檻，正式 top-50 與快篩一致後只執行一次官方 Judge。U2 graded Recall 0.737060／MRR 0.480002，是命中率／排序取捨，依預定政策不改預設。結果：[H1U_RESULTS.md](C:/Users/yuting/Desktop/HW2_ADL/H1U_RESULTS.md)。
- **W08 完成（採用微調 BGE）**：固定一個配置／1 epoch，train 3,864 筆／242 steps；43 tests、真模型 smoke、離線重載、Gold 六項門檻及來源稽核通過。唯一一次官方 Judge：Recall@10 **0.829193**／MRR@10 **0.611836**，均高於 H1；beauty-train 也提高。已設為模型預設，完整 966 題無覆寫排名與被評估配置全數一致。結果：[W08_RESULTS.md](C:/Users/yuting/Desktop/HW2_ADL/W08_RESULTS.md)／`results/w08_final.json`。
- **W09 完成（Q2 實驗與分析）**：54 tests、固定五題×三 prompts、同題同 W08 top-3、指定 Qwen 離線生成與完整來源／token IDs 核對完成；15 筆均 EOS、證據無截斷。原始回覆、證據、逐例助理質性評語與比較均保存。這是實驗完整性通過，三種 prompts 仍有內容／格式錯誤，並非生成品質全通過。結果：[W09_RESULTS.md](C:/Users/yuting/Desktop/HW2_ADL/W09_RESULTS.md)。

## Current Task

- **W09 已完成，本輪停止。**固定 15 筆生成、token／同證據核對及助理逐例分析均完成，沒有額外生成、修改 W08 或執行 public。主要 retrieval 預設仍為微調 BGE＋H1 Hybrid；下一工作項目是 W10，正式 `report.pdf` 與封裝尚未完成。

## Pending Tasks

| 工作 | 狀態 |
|---|---|
| W01 模型相容性、token 與資源 smoke | 完成；Qwen 相容性已在 W06 補驗證 |
| W02 分組 train/dev manifest 與 fixtures | 完成 |
| W03 商品索引與 PID mapping | 完成 |
| W04 完整多輪對話表示 | 完成 |
| W05 B0 官方入口整合 | 完成；中斷後已核實測試與程序結果 |
| W06 Dev 指標與冷啟動評估 | 完成；開發環境結果與限制已記錄 |
| W07 Reranker | 完成比較；A1 退步，保留程式但不採用為預設 |
| H1 BM25／RRF | 完成；Hybrid 兩項指標改善，預設提升與完整排名驗證通過 |
| H1U BM25 user 回合／加權消融 | 完成；U2 Gold 合格、graded 呈取捨，維持 H1 |
| W08 對比訓練 | 完成；微調模型已採用，完整預設驗證通過 |
| W09 Q2 生成 pipeline | 完成 15 筆、原始紀錄與逐例分析；語意缺陷保留 |
| W10 凍結與 public 官方驗收 | 待執行 |
| W11 報告、重現與封裝 | 待前述產物完成 |

## Test Results

- W09：初期 51 tests／5.833 秒通過；加入質性彙整的完整覆蓋／合法分數測試後，**54 tests／5.962 秒通過**（`results/w09_tests_final.log`）。Context 測試涵蓋完整角色對話、同題同證據、最長 prompt 共用縮短、超長拒絕而不靜默刪除、target／重複 qid／未知 PID／控制標記、續跑前綴與主觀／客觀統計分離。
- W09 真實生成：指定本地 Qwen、bf16、batch 1、greedy、seed 0，完整 5×3 grid。最長 input 2337／output 211 tokens，15／15 EOS、0 長度停止／0 證據截斷／0 空答；程序含載入 84.250 秒，CUDA allocated 3.148 GiB／reserved 3.545 GiB、RSS 5.892 GiB。未重跑或改寫回答；`results/w09_q2/manifest.json`／`outputs.jsonl`。
- W09 稽核：完整 prompt／input IDs／top-3／共用 evidence hashes 可由固定來源重建，raw output 可由 output IDs 一致解碼；模型／資料／來源 hashes 通過。15 筆皆無證據外 PID，但只有 4 筆提任何 PID、0 筆遵守指定格式，不能據此宣稱正確。可核對統計與逐例助理質性評分分開保存於 `audit.json`／`review.json`／`review_summary.json`；P1／P2／P3 的 5.8／4.4／3.2 是本次非盲助理自訂量表均分，不是官方分數或及格門檻。
- W08：43 tests／5.816 秒通過（`results/w08_tests_initial.log`）。新增資料切分隔離、去重 batch、全資料一次覆蓋、singleton 修補、不可用負例拒絕及正式／訓練 query 前處理一致性；另核對真實 train 3,864／50k 商品及 manifest。
- W08 smoke：4 個實際訓練步驟，有限 loss／非零梯度、權重更新、離線存檔／重載通過；重載向量最大差 0。正式 Hybrid 入口於 12 商品／8 題回傳合法 top-10，保護檔 hash 不變；`results/w08_training_smoke.log`／`w08_smoke_retrieval_timing.json`。這是工程 smoke，不是品質成績。
- W08 訓練：固定 FP32／batch 16／lr 2e-5／1 epoch，全部 3,864 筆使用一次，242 steps（末批 8）、warmup 25 steps；純訓練迴圈 30.018 秒，腳本總時間 53.555 秒。CUDA allocated 峰值 4,244,037,120 bytes（3.953 GiB）、reserved 4,838,129,664 bytes（4.506 GiB）；checkpoint 重載最大差 0。完整 loss／資料與 checkpoint hashes 見 `artifacts/training/w08_epoch1/training_manifest.json`；尚未據此宣稱檢索進步或官方 8 GB 驗收通過。
- W08 Gold：完整 966／50k 正式入口冷啟動 110.848 秒，Gold Hit10 0.665631／MRR10 0.481551／Hit50 0.832298；beauty-train 0.450801／0.270179／0.707094，六項均高於 H1。Hit10 整體 +112／−8、beauty +51／−7；全來源分組與逐題得失見 `results/w08_screening.json`。預先固定的門檻與 checkpoint hashes 已核對，`w08_judge_plan.json` 僅授權此一配置的一次原版 Judge。
- **W08 官方與最終稽核：**單次原版 Judge 的 Recall 0.829193／MRR 0.611836（H1 0.739130／0.465780），命中 714 → 801，Recall +110／−23、MRR +382／−94；beauty-train 0.741419／0.481396（H1 0.633867／0.359200）。9,017 次非 gold 比較／0 parse failures、冷啟動 577.955 秒；CUDA allocated 4.569 GiB／reserved 11.178 GiB。Raw output／gain／aggregate、資料、模型與來源 hash 均通過。
- **W08 預設驗證：**舊來源保存 `snapshots/h1u/` 後僅更改模型預設路徑；43 tests／5.625 秒再次通過。無任何模型或檢索覆寫的正式入口跑完整 966／50k top-50，冷啟動 104.861 秒，逐題與被評估 W08 全數一致。`results/w08_final.json`：`adopt=true`、`official_judge_runs=1`、`default_verified=true`。B0／A1／H1／H1U 歷史來源及輸入 hash 也核對通過。
- H1U：37 個測試通過（3.578 秒；`results/h1u_tests_screening.log`），涵蓋 user 回合保留、前文限制、短指代補前一 assistant、否定／單字需求、普通冒號與續行、異常角色回退、末輪／title 權重、Dense 原文與商品不變、零分候選與 Gold 門檻。
- H1U 快篩完成：U0 完全重現 H1 top-50；共用 Dense 的前 50 名也與 H1 原 Dense 一致。僅 U2（users＋末輪 2 倍）通過整體／beauty-train 三 Gold 指標門檻：整體 Hit10 0.571429／MRR10 0.376779／Hit50 0.718427；beauty-train 0.375286／0.222055／0.540046。U1／U3 的整體 Hit10 下降，不進入 Judge。完整數據：`results/h1u_screening.json`。
- H1U 正式管線驗證已通過：完整 966 題 top-50 與共用 Dense 快篩全數一致，冷啟動 104.686 秒。一次性 `h1u_judge_plan.json` 僅授權 U2 原版 Judge，固定模型／順序／batch 8；未評分其他配置。
- **H1U 單次官方評分與稽核完成：**整體 Recall@10 **0.737060**、MRR@10 **0.480002**；相較 H1 少命中 2 題、MRR +0.014222。Beauty-train Recall **0.636156**／MRR **0.359873**（H1 0.633867／0.359200）。9,108 次比較／0 parse failures，judge 冷啟動 578.081 秒；allocated 4.587 GiB／reserved 13.287 GiB。共用非 gold 的 6,051 對中有 47 對判斷不同，未重跑模型或改分數。最終 `adopt=false`、保留 H1；`results/h1u_final.json`／`h1u_audit_final.log`。
- H1：新增 5 個獨立 BM25／RRF 測試，連同既有測試共 **27 個通過**（3.486 秒；`results/h1_tests.log`）。BM25 對照獨立公式 oracle，涵蓋 query 詞頻政策、增刪／換序、小數／型號／否定與長文尾端；融合測試涵蓋候選來源、PID 去重、同分、零詞面命中與小池。真實資料結果另列如下。
- H1：新增精確歷史快照／防資料與 source 竄改測試後，共 **28 個通過**（3.617 秒；`results/h1_tests_final.log`）。BM25-only 完整 dev gold Hit@10 0.494824、MRR@10 0.362346、Hit@50 0.627329；top-10 冷啟動 9.576 秒，top-50 前綴 966 題一致。Hybrid dev-fast Hit@10 0.57、MRR@10 0.305381；完整官方結果另列。
- H1 Hybrid 完整 dev：gold Hit@10 **0.557971**、MRR@10 **0.354759**、Hit@50 **0.703934**，top-10 冷啟動 105.499 秒；獨立 top-50 前綴在 966 題一致。原版官方評分與預設提升結果另列。
- H1 B0 全 dev top-50 回歸：輸出 SHA-256 與 W06 完全相同，冷啟動 97.286 秒。BM25-only 未初始化 CUDA，peak RSS 443,670,528 bytes；Hybrid CUDA allocated 503,805,952 bytes、reserved 717,225,984 bytes。兩輪完整官方 judge 均已完成。
- H1 BM25-only 原版官方評分完成：Recall@10 **0.650104**、MRR@10 **0.450887**，相較 B0 命中少 2 題、MRR 提高；9,182 次比較、0 parse failures、judge 冷啟動 610.263 秒。Allocated 峰值 4.568 GiB／reserved 12.943 GiB；Hybrid 結果與預設選型見下項。
- **H1 Hybrid 官方評分與收尾完成：**Recall@10 **0.739130**、MRR@10 **0.465780**；命中 714／966 題，相較 B0 多 84 題。Recall 132 題改善／48 題退步，MRR 336 題改善／166 題退步；9,121 次比較／0 parse failures。Judge 冷啟動 598.660 秒，allocated 4.569 GiB／reserved 9.719 GiB。
- **H1 預設提升驗證：**保存原實驗 source 至 `code/script/snapshots/h1/` 後改預設為 hybrid；28 個測試再次通過（3.722 秒；`results/h1_tests_promoted.log`）。不設定 `HW2_RETRIEVER` 的正式入口完成全 966 題，冷啟動 112.755 秒，排名與原 Hybrid 全數一致。最終稽核 `selected_default_verified=true`，見 `results/h1_summary.json`／`h1_audit_final.log`。舊 B0／A1 的重現命令明確指定 dense。
- W07 模組：新增 5 個重排測試，連同原測試共 **22 個通過**（3.401 秒；`results/w07_tests.log`）。涵蓋候選隔離、跨 query 組批映射、同分維持 dense 次序、query 首尾／否定與 512-token pair 預算、小池／top-k、NaN／Inf 與錯誤設定。單元測試不作為品質指標；真模型結果另列如下。
- W07 模型：MiniLM revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`，6 檔／91,815,758 bytes，下載與 hash 已保存於 `results/reranker_download.json`，稽核再次核對全部檔案一致。
- W07 真模型檢索已完成：dev-fast gold Hit@10 0.45；完整 dev gold Hit@10 0.427536、MRR@10 0.236438，冷啟動 170.393 秒。B0 top-50 重跑與 W06 檔案 hash 完全相同。A1 檢索 peak CUDA allocated 0.469 GiB。
- W07 pair 分析：48,300 pairs 中 21,756（45.04%）商品被截斷；query 截斷 0。Gold 可完整放入的組別也退步，不能將整體結果歸因於截斷。結果見 `results/w07_pair_diagnostics.json`。
- **W07 官方評分與稽核完成：**Recall@10 0.606625、MRR@10 0.327259；9,247 次比較／0 parse failures，judge 冷啟動 595.296 秒。Recall 有 76 題改善、120 題退步。原始 output／gain／aggregate 一致，來源 hash、student block 邊界、候選隔離與 B0 回歸均通過。見 `results/w07_summary.json`／`w07_audit.log`。
- W00：WSL Ubuntu 22.04.4、約 32 GB RAM、RTX 4090（24 GB）可用。
- W00：ADL-final 的 torch 2.8.0+cu128、transformers 4.56.2、sentence-transformers 5.1.0、numpy 2.3.5、datasets 4.0.0、hub 0.36.0、accelerate 1.10.1 均 import 成功；faiss import 失敗（ModuleNotFoundError）。完整輸出已保存。
- Windows 預設 Python 3.11.5 的套件不符；未修改該環境。WSL 在 sandbox 內首次讀取被拒，經工具的自動審核後成功進行只讀檢查；不是測試成功紀錄。
- W06 已有完整 gold-only 與原版 evaluator 結果；尚未訓練或跑 public。
- W01：`py_compile` 通過；128 商品／128 對話產生 384 維有限、單位長向量；FAISS scores 與回傳 rows 均符合 NumPy 精確內積。128 商品中 8 個超過 512 tokens，query 0 個；這是前 128 筆 smoke，不代表全資料分布。
- W01：模型載入起算 1.323 秒（不含 Python／套件 import）；CUDA peak allocated 452,739,584 bytes、reserved 654,311,424 bytes。這不是完整冷啟動或官方硬體測量。見 `results/w01_embedding_smoke.json`。
- W02：5 個 unittest 通過（`results/w02_tests.log`），涵蓋跨規則傳遞、排序重現、角色／否定區別、無標籤 Q2，以及真實資料 train/dev qid／gold PID／商品文字／對話均無交集。Q2 selection_tag 僅是可重現選樣線索，不是人工語意標註。
- W03：3 個 tests 通過，實際 FAISS 1.12.0 與 NumPy fallback 均與暴力 oracle 一致；15 個同分候選、換序、增刪／修改商品、NaN／zero vectors 已驗證（`results/w03_tests.log`）。
- W04：8 次 test 執行通過（含 3 個繼承而重跑的 W03 tests；後續已移除重複執行），新增兩個多輪／超長案例；見 `results/w04_tests.log`。
- W05：`results/w05_tests.log` 顯示 12 tests／2.999 秒／OK。`results/w05_smoke.json` 與兩份實際排名檔一致，兩次獨立程序耗時 20.048／18.324 秒；這是 12 商品、8 題的 smoke，不是 50k／75k 全池效能或官方品質分數。
- 2026-10-24 狀態核查：原始資料與四份衍生資料的 SHA-256 均吻合 split manifest；兩份 smoke 排名逐題合法且完全相同；`retrieval.py`、`evaluate.py`、`eval_utils.py` 的 hash 仍與原版一致。已讀取前次工具程序的最終 exit code 0；沒有把本次只讀核查冒充重新執行模型。
- W06 第一階段：15 個 tests 通過；100 題 dev-fast／50,000 商品的 gold Hit@10 = 0.4600、gold MRR@10 = 0.24854，完整程序 112.305 秒，CUDA allocated 峰值 503,805,952 bytes。這不是官方 graded 分數，也不是 3070 效能結果。見 `results/w06_b0_dev_fast_gold.json`／`w06_b0_dev_fast_timing.json`。
- W06：Qwen 模型下載成功，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，7 檔／3,098,955,668 bytes；完整 hash 見 `results/generator_download.json`，judge smoke 結果見下方。
- W06 第二階段：完整 966 題 dev 的 gold Hit@10 = 0.468944、gold MRR@10 = 0.292416，程序 105.082 秒；另跑 top-50 的 gold Hit@50 = 0.649068，966 題的前十名與正式 top-10 run 全數相同。候選改善空間不等於已證明 reranker 有效。
- W06：完整 50,000 商品有 13,488 個（26.976%）超過 BGE 512-token 上限；966 段 dev 對話含 instruction 全數未超長（最大 200 tokens）。見 `results/w06_token_statistics.json`。
- W06 Qwen smoke 已通過：原版 judge、bf16、batch 8、greedy、12 新 tokens、不截斷；first-8 與最長 8 筆共 16 次判斷無解析失敗，CUDA allocated 峰值 4,575,377,408 bytes，reserved 5,532,286,976 bytes。17 個單元測試通過（`results/w06_tests_final.log`）。
- W06 dev-fast 官方評分：graded Recall@10 = 0.680000、graded MRR@10 = 0.375484；954 次比較、0 parse failures，程序總耗時 88.149 秒。
- **W06 完整 dev 官方評分：966 題，graded Recall@10 = 0.652174、graded MRR@10 = 0.396446。**453 題命中 gold，另 177 題由替代品命中，合計 630 題；9,207 次 LLM 比較，0 parse failures。原版 judge 總程序 583.114 秒，allocated 峰值 4.554 GiB、reserved 峰值 8.631 GiB；prompt 最長 2,265 tokens，未超 context。
- W06 稽核通過：原始 judge output、逐題 gain、aggregate metrics、qid／ranking 對應與來源 hash 一致。Dev-fast 與完整 dev 的 100 題檢索排名全數相同；top-50 前十名也與獨立 top-10 全數相同。不同組批的 judge 每題分數有 4／100 題不同，兩份紀錄保留；見 `results/w06_summary.json` 與 `results/w06_audit.log`。

## Known Issues

- PROJECT Q1–Q5 未獲新官方答覆；不將本次實作視為對免訓練、額外套件、cache 或 private 時限的批准。
- 開發使用 WSL Ubuntu 22.04／Python 3.12.12／RTX 4090；官方要求 Python 3.12.9／RTX 3070 8 GB，因此開發成功不能當作官方環境驗收。
- 後續大檔下載前仍須重新確認磁碟空間，不以 W00 舊量測當作目前可用容量。
- W06 下載後 C: 約剩 15.97 GB。整張 GPU 觀測到約 19–20 GiB 用量，但 BGE 程序自身 allocated 峰值僅約 0.47 GiB；WDDM／WSL 無法列出完整 GPU 程序清單，不能將整張卡用量歸給本作業，速度測量也不能視為無其他負載的硬體基準。
- 完整 dev 已完成，但尚無 public 或官方硬體效能結果。超長對話的首尾處理主要由 tokenizer 替身測試驗證；真實 dev 對話全數短於上限，不能把這次品質評估當成長對話語意驗收。
- **8 GB 風險：**完整 judge reserved 峰值 8.631 GiB，包含 allocator cache；allocated 4.554 GiB。不能據此直接斷言 3070 會 OOM，也不能宣稱已通過 8 GB；後續須驗證，不能改 batch／截斷／judge 掩蓋問題。
- **Judge 組批敏感性：**相同 100 題、相同排名在 dev-fast／完整 dev 的不同 batch 搭配下，有 4 題 Recall 或 MRR 不同；確切原因未做隔離實驗。後續以完整 dev 與固定順序／batch 8 作主要對照。
- W04／W05 的額外 reviewer 工作因使用上限中斷，沒有完成的 review 結論；上述單元測試與真模型結果則已有可核對紀錄。
- W07 的原版 judge CUDA allocated 峰值 4.555 GiB、reserved 8.754 GiB；官方 8 GB 驗收尚未完成。MiniLM 不採用是因 dev 品質退步，不能將結論擴大為所有重排模型都不適用。
- H1 仍有歷史品牌干擾與 `1.6-Ounce`／`1.6 oz` 等格式差異；Hybrid 並未修復所有案例。BM25 完整商品與 BGE 512-token 表示不同，不能把改善完全歸因於單一公式。Judge reserved 峰值曾達 12.943 GiB（BM25 結果評分）／9.719 GiB（Hybrid 結果評分），allocated 約 4.57 GiB；8 GB 正式環境仍待驗收。
- H1U 的短指代保護未在真實 dev 觸發（0／966），只有手寫測試證據。User 回合中的歷史商品仍可能干擾；末輪「明確」是非固定短指代詞的近似判斷。不得把 Gold 六指標改善等同官方整體 Recall 改善。
- 已建立 W08 訓練／run.sh，以及 W09 `generate_report.py`、固定 prompts／設定、稽核、15 筆輸出與分析。`download.sh`、`report.pdf` 尚未建立，README 仍是開發紀錄而非最後封裝版，因此 Phase 3 與最終繳交均未完成。後續 download.sh 必須取得選定的微調 checkpoint，不能只下載原版 BGE；完整對話需求改寫、分段仍未實作。
- W08 使用同一 dev 選型，尚非獨立泛化驗證；不同 PID 的合理替代品仍可能是 in-batch false negatives。Judge reserved 11.178 GiB／allocated 4.569 GiB，官方 8 GB 驗收仍待完成，不能因本機訓練使用不到 8 GB 就視為全作業驗收通過。
- W09 三種 prompts 均有內容缺陷，包括跨商品保證／技術混用、產地與容量捏造、未證實限制說成滿足、將商品說明當使用者陳述，以及錯誤拒答；指定 PID 格式 0／15。證據與輸出均無截斷，不能把這五例的缺陷歸因於 token 上限；模型容量／提示詞機制的因果未隔離。五題與助理非盲評閱也不足以建立普遍生成品質結論。

## Next Actions

1. W09 本輪結束，不更換樣本、不追加生成挑結果；保留所有錯誤與完整分析，W08 維持凍結。
2. 下一工作項目 W10：確認正式環境限制並在凍結版本做 public 官方驗收；開發 4090 結果不能冒充 3070 8 GB 驗收。
3. W11：將 Q1／Q2 已完成結果納入 report.pdf，完成微調 checkpoint 取得方式、download.sh、最終 README／重現與正確學號封裝；尚未提交作業。
