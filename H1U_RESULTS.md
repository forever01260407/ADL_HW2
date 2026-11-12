# H1U — 僅優化 BM25 對話表示

日期：2026-10-24（Asia/Taipei）。狀態：**H1U 完成；U2 Gold 六指標改善、單次官方評分呈現取捨，維持 H1 預設。**

重現提醒：上述狀態為 H1U 當時的決策，最新選型見 PROGRESS.md。本報告使用未微調的 BGE；執行下列歷史指令前，先設定 `export HW2_MODEL_PATH=artifacts/models/bge-small-en-v1.5`，不可混入後續 W08 checkpoint。

## 固定範圍

- Dense 始終使用原完整角色對話、原商品文字、同一 BGE-small 模型與正規化／FAISS；兩路各 100、等權 RRF 60、BM25 k1=1.5／b=0.75 不變。
- 不訓練，不啟用 reranker，不使用 public。qid 前綴只用於分析，不進入推論分流。
- BM25 users 模式保留所有顧客回合及否定，僅刪 assistant；格式異常時回退原文。行首角色解析只接受 user 開頭、角色交替、user 結尾；一般冒號／續行保留。
- 固定短確認／指代詞表（最多 5 詞，如 yes／that one／sounds good）才補緊鄰前一 assistant；不把 `red`、`not scented` 當成指代。短指代不套用最新明確需求加權。
- 最後明確 user 的不同詞項可由權重 1 改為 2。原 H1 是 unique-term 計分，單純重複字串不會生效，因此直接乘 query term weight。
- Title 權重 2 僅增加商品第一行的 term frequency；IDF 與原商品詞數長度正規化不變，Dense 商品仍原文。不是 BM25F。

原始 dev preflight：966 題，489 題為 user／assistant／user，477 題為五則交替；內容沒有換行或偽裝行首角色標籤。展平原文一般情況仍無法無損分辨任意內嵌角色標籤，不宣稱 parser 普遍消除歧義。

## 單一變因消融

| 配置 | BM25 query | 末輪詞項權重 | Title TF 權重 | 對照 |
|---|---|---:|---:|---|
| U0 | H1 完整原文 | 1 | 1 | 原 H1 |
| U1 | 所有 user＋短指代保護 | 1 | 1 | U0：僅 query 表示 |
| U2 | 同 U1 | 2 | 1 | U1：僅末輪權重 |
| U3 | 同 U1 | 1 | 2 | U1：僅 title TF |

沒有混合末輪與 title 加權。原 H1 `HW2_BM25_QUERY_MODE=full`／兩項權重 1 保留可重現。

## 測試與省時流程

**37 個測試通過，3.578 秒**（`results/h1u_tests_screening.log`）。新增回合篩選、先前 user 限制、否定、指代保護、普通冒號／續行、異常角色回退、末輪實際加權、title TF、Dense 與商品原文不變、零分候選 trace、Gold 門檻測試。

開發快篩只建立一次正式 Dense top-100（冷啟動 98.188 秒），四個 BM25-only top-100 分別執行，再以相同 RRF 合併成 top-50。Trace 記錄正分候選數，零分不投票。**U0 的完整 966 題 top-50 與原 H1 完全一致**，共用 Dense 的前 50 名也與 H1 Dense 完全一致。這些快篩不是每個配置的完整 Hybrid 冷啟動速度。

所有指標均由 `analyze_results.py` 在完整 966 題／50,000 商品池計算，程式新增 qid 前綴分組及 top-100 分析支援。

## Gold 快篩結果

以下是 **gold-only**，與官方接受替代品的 graded 指標不同。

| 配置 | 整體 Hit@10 | 整體 MRR@10 | 整體 Hit@50 | 進入 Judge |
|---|---:|---:|---:|---|
| U0／H1 | 0.557971 | 0.354759 | 0.703934 | 基準 |
| U1 user-only | 0.544513 | 0.351070 | 0.713251 | 否 |
| **U2＋末輪 2 倍** | **0.571429** | **0.376779** | **0.718427** | **是，唯一勝出** |
| U3＋title TF 2 倍 | 0.552795 | 0.365790 | 0.712215 | 否 |

beauty-train 分組 437 題：

| 配置 | Hit@10 | MRR@10 | Hit@50 |
|---|---:|---:|---:|
| U0／H1 | 0.350114 | 0.193782 | 0.505721 |
| U1 user-only | 0.345538 | 0.205196 | 0.526316 |
| **U2＋末輪 2 倍** | **0.375286** | **0.222055** | **0.540046** |
| U3＋title TF 2 倍 | 0.361556 | 0.228338 | 0.530892 |

門檻事先固定為整體與 beauty-train 三項均不下降，且兩組各至少一項提高；符合者依 beauty MRR、整體 MRR、Hit10、Hit50 排序。U2 六項實際皆提高。U3 的 beauty MRR 雖較高，但整體 Hit10 下降，因此不送 Judge。

U2 對 H1 的逐題 Gold 得失：

| 範圍／指標 | 改善題數 | 退步題數 | 淨增 |
|---|---:|---:|---:|
| 整體 Hit10 | 53 | 40 | +13 |
| 整體 MRR10 | 181 | 110 | +71 |
| 整體 Hit50 | 27 | 13 | +14 |
| beauty Hit10 | 28 | 17 | +11 |
| beauty MRR10 | 69 | 43 | +26 |
| beauty Hit50 | 18 | 3 | +15 |

MRR 的「淨增」僅是改善題數減退步題數，不等於 MRR 值差。完整 qid 清單與所有配置見 [h1u_screening.json](C:/Users/yuting/Desktop/HW2_ADL/results/h1u_screening.json)。

實際 966 題的 parse fallback 0、短指代補 assistant 0；U2 末輪加權 966 題。指代保護只由手寫測試驗證，本次真實 dev 分數未測到它的語意效果。

## 單次官方評分與採用

僅 U2 可進入原版 `evaluate.py`，先以正式 Hybrid 重新編碼全商品／完整對話並檢查 top-50 與快篩完全一致。通過後建立一次性 `h1u_judge_plan.json`；其他配置不執行 Judge。正式 graded Recall／MRR 都不下降且至少一項提高才更新預設，否則保留 H1。

正式管線驗證已完成：966 題 top-50 全數與快篩一致，冷啟動 104.686 秒。U0／U1／U3 沒有執行 Judge，**只有 U2 執行一次完整原版官方評分**：

| 範圍 | 配置 | Graded Recall@10 | Graded MRR@10 |
|---|---|---:|---:|
| 完整 dev 966 題 | H1 | 0.739130 | 0.465780 |
| 完整 dev 966 題 | U2 | 0.737060 | 0.480002 |
| beauty-train 437 題 | H1 | 0.633867 | 0.359200 |
| beauty-train 437 題 | U2 | 0.636156 | 0.359873 |

整體官方 Recall 少命中 2 題（714 → 712，下降 0.207 個百分點），MRR 提高 0.014222；beauty-train 兩項微幅提高（Recall 多命中 1 題）。全體逐題 Recall 53 題改善／55 題退步，MRR 229 題改善／169 題退步。

**不提升 U2 為預設。**原因是未符合事先定義的兩項 graded 指標都不下降；現行仍為 H1 `query_mode=full`、末輪／title 權重皆 1。U2 的 Gold 與排序品質改善值得保留，但不能宣稱整體官方命中率提高。只記錄這一次結果，不改門檻或再評分另一個配置。

Judge 9,108 次非 gold 比較、0 parse failures，冷啟動 578.081 秒；allocated 峰值 4.587 GiB／reserved 13.287 GiB。Reserved 包含 allocator cache，官方 8 GB 環境仍待後續驗收，沒有修改 batch、模型或截斷規則。

原始輸出／gain／aggregate、來源與資料 hash、正式／快篩 top-50 一致性及唯一勝出配置門檻均已稽核通過。最終結果：[h1u_final.json](C:/Users/yuting/Desktop/HW2_ADL/results/h1u_final.json)、[單次官方原始結果](C:/Users/yuting/Desktop/HW2_ADL/results/h1u_winner_graded.json)、[最終稽核](C:/Users/yuting/Desktop/HW2_ADL/results/h1u_audit_final.log)。`default_verified=false` 表示本輪沒有提升新預設，不是驗證失敗；`default_action=retain_h1_full_1_1`。

僅用已存在輸出比較，沒有再呼叫模型：H1／U2 共同的 6,051 個非 gold query–PID 配對中，47 對的 Judge gain 不同。資料／模型／原版規則相同，但候選排名會改變組批；確切原因未隔離。因此少 2 題不能解讀為具有統計意義的確定退步，仍依本輪預定門檻保留 H1；未改寫官方分數。

## 重現與紀錄

主要程式：`retrieval_utils.py` student block、`run_bm25_representation.py`、`analyze_results.py`、`audit_bm25_representation.py`。

- 快篩：`.venv/bin/python code/script/run_bm25_representation.py`；使用固定 dev 與輸出名稱，已有完整快篩時拒絕覆寫，部分完成的詞面 run 只在來源／設定 hash 一致時可接續。
- 審核既存結果：`.venv/bin/python code/script/audit_bm25_representation.py`。
- `--prepare-judge` 只允許 Gold 門檻與正式管線一致性通過後建立一次計畫；已有計畫或 Judge 結果會拒絕再次授權。
- `HW2_BM25_QUERY_MODE=full|users`、`HW2_BM25_LAST_WEIGHT=1|2`、`HW2_BM25_TITLE_WEIGHT=1|2` 控制表示。`HW2_BM25_TRACE_PATH` 僅是開發診斷，可完全停用。

U2 是可選配置，重現檢索時明確設定 `HW2_RETRIEVER=hybrid HW2_RERANK=0 HW2_BM25_QUERY_MODE=users HW2_BM25_LAST_WEIGHT=2 HW2_BM25_TITLE_WEIGHT=1`；不設定這些變數則仍使用 H1。稽核程式只重讀既存資料，不會重跑 Judge。本輪的一次配額已使用完畢。

開發為 4090／Python 3.12.12；public、官方 8 GB 硬體、訓練、Q2 與最終封裝均不在本輪範圍。本輪是固定 dev 選型，不能推定未見資料一定改善；user 回合中的歷史商品仍保留，不能宣稱歷史干擾已完全解決。

本輪「最新明確需求」以末輪不屬於固定短指代詞表作近似判斷，沒有新增意圖模型；含糊的末輪仍可能被加權。短詞表與解析規則已在看 Gold 結果前固定，本輪不再擴大規則或掃描權重。
