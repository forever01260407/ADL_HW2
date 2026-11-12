# PROGRESS.md — Phase 3 Implementation

更新日期：2026-10-25（Asia/Taipei；最新ZIP完整重驗通過，使用者已確認上傳NTU COOL）

### 最新ZIP含TA補充規則的完整重驗完成（2026-10-25 23:53；最新）

- Completed Tasks：從SHA為cea063345fb73324b25a11fb6312ea2f5790d059ce73ba1593f73cb0fd8660cd的最新ZIP再次解壓至output/ta_final_recheck_20261025，空模型快取重新下載全部三個模型，完整執行README的訓練、Public檢索、官方評分與15份生成，全部一次通過。沒有修改、重新封裝ZIP或雲端模型；使用者已明確確認上傳最新版，尚未獨立下載NTU COOL端的最新版。
- Test Results：下載95.834秒；兩階段完整訓練235.559秒，各3,864題／242 steps，權重SHA與原模型相同，reserved峰值2.876953／3.339844GiB。Public檢索272.895秒，1,000題排名全部相同；官方Judge631.445秒，Recall 0.952／MRR 0.8024043650793651，9,096比較／0解析失敗。15份生成91.680秒，原文全部相同，29.235秒生成稽核通過。獨立核算10,000候選gain與指標、27模型檔、627份來源／參考hash、ZIP不變均通過；PDF 13頁可讀、pip check及shell語法／755權限通過。
- Confirmed Rules：TA會跑run.sh；訓練無時限但限8GB；data/train相對路徑、requirements安裝責任、download只下載與核對解壓、Train用Qwen標記、原版官方先retrieval後evaluate及File Layout均核對。這輪沒有Public訓練、調參或模型選擇。
- Known Issues：實際為4090／Python3.12.12，沒有實體3070／Python3.12.9或全新pip安裝驗證；allocator峰值不包含全部driver/context與其他程序。Private未知，PDF本輪只讀不重建。因本機磁碟不足，核對權重後清掉這輪重訓checkpoint與下載ZIP副本；保留訓練manifest／loss／雜湊、所有輸出與正式模型，清理有獨立紀錄。
- Current Task：完整重驗完成，沒有新問題；紀錄output/ta_final_recheck_20261025/audit/summary.json、final_integrity_audit.json與VERIFICATION.md。最新正式ZIP仍22,621,533 bytes／628檔，SHA不變。
- Next Actions：使用者已回覆上傳最新版，無須再換ZIP或重傳模型；保持Drive連結有效至少三週。沒有代理課程提交或外部訊息。

### 助教討論區規則複核完成（2026-10-25；最新）

- Confirmed Requirements：依使用者貼上的TA回覆，run.sh會實際執行；訓練無時限但需符合8GB VRAM；TA依requirements安裝套件、提供根目錄data/train/；Train候選可用官方Qwen標記；先retrieval再evaluate；重現效能相近即可。
- Completed Tasks：逐項檢查正式ZIP。正式run.sh使用gradient checkpointing，與早期未最佳化的高記憶體開發實驗分開。發現8GiB除以可見顯存的比例可能在略低於8GiB的卡上大於1，加入min(1.0, ...)；README補充TA已確認規則。模型、資料、超參數、正式推論與PDF未變。
- Test Results：再完整重訓各3,864題／242 steps；第一階段reserved 2.876953 GiB、第二階段3.339844 GiB。兩階段權重SHA、組批檔與每一步loss均與修正前完全相同；7.5／7.9995／8／24GiB容量情境的比例範圍檢查通過。pip check通過；未重新pip安裝或重跑Public／Judge。
- Final ZIP：output/r15922a14.zip為22,621,533 bytes／628檔，SHA-256 cea063345fb73324b25a11fb6312ea2f5790d059ce73ba1593f73cb0fd8660cd。相較上一份只改train_release.py、README與manifest；CRC、逐檔hash、嚴格頂層和shell權限通過。此版本取代下方較早ZIP hash。
- Known Issues：未在實體3070／Python3.12.9驗收；PyTorch allocator數據不包含全部driver/context或其他程序。早期未最佳化base實驗曾超過8GiB，歷史紀錄不刪改。20GB核對為專案資產／輸出大小，不含既有OS與Python套件環境；目前含兩次完整重訓的測試專案約6.21GB。
- Current Task：規則複核與修正完成。證據output/ta_recheck_20261025/audit/ta_clarification_check.json；規則來源記入PROJECT補充，原TA貼文不是我們的Qwen3架構。
- Next Actions：若先前已上傳，使用最新output/r15922a14.zip替換；雲端模型不需重傳。沒有代理使用者上傳課程。

### 從正式ZIP完整重建驗證（2026-10-25；取代下方較早狀態）

- Completed Tasks：從封存ZIP解壓至全新目錄，依README從網路下載全部三個模型、核對27個模型檔案，初始化後完整重訓兩階段，再執行完整Public檢索、官方Judge及15份生成回答。沒有借用既有模型、索引或預測快取；原始課程資料由驗證端另行提供。
- Test Results：下載93.624秒；完整兩階段重訓232.720秒，各3,864題／242 steps，兩個權重SHA均與原結果一致。Public檢索261.841秒，1,000題排名全部相同；Judge624.340秒，Recall@10 0.952、MRR@10 0.8024043650793651與原結果相同，9,096次比較／0解析失敗。15份生成原文亦全部相同，提示詞、證據與token解碼稽核通過；另以獨立程式核算10,000個候選的gain與兩項指標。
- Known Issues / Fix：額外生成稽核工具原先用相對目錄時，在最後建立來源路徑紀錄會報錯；已於audit_w09.py入口加入directory.resolve()並以真實15份輸出驗證。原錯誤紀錄保留，不能把初次整條流程冒稱無錯通過。訓練、正式檢索、Judge與生成均未受影響。
- Final ZIP：已納入上述一行修正，output/r15922a14.zip現在為22,621,327 bytes／628檔，SHA-256 9c7c5fd9c3582efbb13669b994be697c7cecf31712cdf11978382a955813c272。與原ZIP僅差audit_w09.py及對應manifest，其他626項完全相同；CRC、逐檔hash、六檔加code的頂層與shell權限通過。模型、README、PDF、正式檢索及既有實驗結果未更動。
- Constraints：使用既有且符合requirements的Ubuntu Python 3.12.12／RTX4090環境，沒有重新pip安裝，也不是指定3070／Python3.12.9驗收。Private未知；本輪固定設定單次Public重現，未用Public訓練、調參或挑分數。
- Current Task：完整重建與發現問題的修正已完成；詳細證據在output/ta_recheck_20261025/audit/summary.json、final_integrity_audit.json與final_zip_reseal.json。舊ZIP與失敗紀錄僅作驗證歷史。
- Next Actions：使用者只需將output/r15922a14.zip上傳NTU COOL，並維持原Drive模型下載權限至少三週；沒有替使用者上傳課程或變更雲端模型。

### 正式ZIP完成（2026-10-25；最新）

- Completed Tasks：依使用者要求建立output/r15922a14.zip，22,621,343 bytes，628檔。SHA-256為bb2c8201bbc1fe168061dccce9effef61cfb7d3ae601f8e95299c58143834e3d。學號最外層、六個必要檔案與code/script排列完全符合投影片；其餘根目錄開發筆記未打包。
- Test Results：ZIP CRC、逐檔SHA、無重複或越界路徑、LF及Unix執行權限通過；模型URL與遠端下載證據一致，PDF與正式檢索來源正確。Ubuntu實際解壓後shell權限／語法及516份參考初始化均通過。詳見output/r15922a14.zip.audit.json及results/w11_final_completion.json。
- Current Task / Next Actions：本機交付準備已完成，由使用者將唯一正式r15922a14.zip上傳課程HW2的NTU COOL；模型維持既有Drive託管，不需重新上傳模型。
- Known Issues：Private與實體3070成績仍未知；本輪未訓練、未跑Public／Judge，沒有自動上傳課程或向外部分享檔案。舊pending-url.zip保留為歷史草稿，不應繳交。

### README與相對路徑複核完成（2026-10-25）

- Completed Tasks：逐一核對README指令與移動後路徑，補上命令須於學號根目錄執行、PDF重建字型需求，以及歷史消融程式可能需要舊模型的說明。
- Test Results：獨立副本從其他工作目錄呼叫download.sh與run.sh --help成功；生成prepare找到15組輸入，contexts／requests與原保存計畫完全一致，沒有重生成回答。報告程式從其他工作目錄重建13頁PDF與Markdown，文字／原稿均與正式版一致；使用本機Windows字型，不宣稱Ubuntu自帶字型。85個隨附Python檔語法通過。正式模型與PDF雜湊保持不變。詳見results/w11_relative_path_validation.json。
- Current Task / Next Actions：正式流程路徑檢查完成，README同步至乾淨交件目錄；尚未建立ZIP，等待後續指示。舊實驗程式不等同全部可在未下載其舊checkpoint時直接執行。

### 依投影片嚴格排列交件檔案（2026-10-25；最新）

- Completed Tasks：使用者指出最外層應完全對齊p.15，已修正output/submission_ready/r15922a14為六個必要檔案加code/；全部其他檔案集中code/script，含official、records、docs與內部manifest。舊版目錄保留於output/previous_layout。新prepare_runtime.py在執行時還原既有程式相對路徑，run.sh先呼叫它；download.sh、retrieval_utils.py、正式模型及PDF不變。
- Test Results：最外層七個項目精確符合，627檔清單與SHA逐一核對通過。在獨立副本以實際bash download.sh安裝模型、513份參考檔案初始化與重複初始化通過；不同既有檔案保留、來源遭改與路徑穿越拒絕等3項檢查通過。bash run.sh兩階段各4步smoke完成且重載差0；8題／12商品的原入口排名與原工作區完全一致，皆有合法唯一top10。紀錄：results/w11_strict_layout_validation.json、output/submission_ready.audit.json。
- Current Task / Next Actions：嚴格格式交件目錄已完成，等待後續建立正式ZIP的指示；不再使用下方D39時的額外根目錄排列。本輪沒有新ZIP、Public或Judge，正式checkpoint保持不變。

### 乾淨交件目錄已建立（2026-10-25）

- Completed Tasks：於output/submission_ready/r15922a14建立獨立乾淨交件目錄，保留原開發工作區與舊submission測試目錄。根目錄10檔中只有README.md是Markdown，另有code／results／data／artifacts四個相關程式與證據目錄；報告原稿位於code/script/docs/report.md。
- Current Task：交件目錄已可供使用者檢閱，尚未壓成ZIP。
- Test Results：623檔的實際清單與manifest逐檔SHA一致，必要6檔齊全，PDF／報告原稿與已檢查新版相同，雲端URL設定與下載證據一致，無模型權重或原始Train／Public資料；舊ZIP雜湊不變。詳見results/w11_clean_staging_validation.json。
- Pending Tasks / Next Actions：使用者確認後再建立正式r15922a14.zip與壓縮檔檢查，最後由使用者上傳NTU COOL。模型已由使用者放在自己的Drive，這輪只下載及本機複製，外部上傳0次。

### 雲端模型下載完成（2026-10-25）

- Completed Tasks：已讀取使用者提供的Google Drive資料夾，確認h5_final_r15922a14.zip；設定直接HTTPS下載網址。Ubuntu空目錄實際下載395,347,153 bytes，19.97228451秒，archive與全部10個checkpoint SHA-256通過。設定檔僅變更URL，其他模型資產規格與權重不變。
- Current Task：下載驗證與README狀態同步完成，等待使用者後續封裝指示。
- Pending Tasks / Next Actions：之後再建立並驗證正式r15922a14.zip，使用者自行繳交NTU COOL。本輪遵守暫不封裝，舊pending-url ZIP維持原樣。
- Test Results：results/w11_remote_download_validation.json記錄Ubuntu／空目錄／10檔hash驗證；未訓練、未跑Public或Judge。
- Known Issues：尚未產生正式ZIP；雲端分享須依作業要求維持至少三週。

### 文件整理進度（2026-10-25）

- Completed Tasks：README由226行改為107行中文操作說明；新版報告以方法名稱取代開發代號，移除授權／對話口吻。未來提交根目錄只保留README這份Markdown，報告原稿映射到code/script/docs/report.md；其餘根目錄開發筆記留在本機。
- Current Task：使用者關閉PDF後，已將新版13頁report.pdf與report.md同步至根目錄，兩檔均與已檢查預覽版SHA相同；文件整理與後續雲端下載驗證皆已完成。
- Test Results：新版13頁已render並逐頁檢視，15筆原始回答與全部提示詞保留，9份指標來源數值核對通過；501個既有來源／模型／資料／評分／舊ZIP檔案雜湊不變。封裝清單無重複路徑、必要檔案齊全。詳見results/w11_editorial_review.json。
- Pending Tasks / Next Actions：託管URL與遠端下載驗證已完成；之後再依使用者指示封裝。這輪沒有建立或更新ZIP，也沒有重跑訓練／Public／Judge。
- Known Issues：w11_local_completion.json與舊report audit是整理前的歷史文件hash；原件與seal保留，舊文件備份於code/script/snapshots/w11_pre_editorial，這輪文件狀態依新editorial audit。未來封裝前須先使用已檢查的新版報告。

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
- **W10 本機 public 評分與配分判斷完成**：凍結 W08、完整 1,000 題／75k 商品、一次原版 Judge，Recall@10 **0.902000**／MRR@10 **0.704348**，均過 Strong；依 p.19 累加估算 public **6／8**。59 tests、原始結果／資料／來源核對通過，9,181 比較／0 parse failures。維持模型，未啟動更大模型或第二輪 public。官方 3070／Python 3.12.9 驗收仍待；結果：[W10_RESULTS.md](C:/Users/yuting/Desktop/HW2_ADL/W10_RESULTS.md)。
- **H2 RRF 權重消融完成（Gold-only）**：永久保存 W08 微調 Dense top-100，先重現完整 966 題 1:1 top-50，再比較固定八組比例。66 tests、7,728 份獨立排名與分組／逐題統計核對通過；4:1 的 Gold Hit10／MRR10／Hit50 為 **0.721532／0.517324／0.862319**。保留 pet MRR／sports Hit10 退步，本輪未跑 Judge／public、未改預設或進入 W11。結果：[H2_RESULTS.md](C:/Users/yuting/Desktop/HW2_ADL/H2_RESULTS.md)。
- **H2B 完成（高權重消融＋單次 dev Judge）**：69 tests 與四組新排名／Gold 獨立核對通過。5:1 Gold Hit10 最高，6:1 Gold MRR10 最高，8／10:1 回落；唯一送評 6:1 的原版 dev Recall **0.845756**／MRR **0.629401**，比 W08 淨多命中 16 題。8962 比較／0 parse failures、raw gain／aggregate 稽核通過；保留 pet MRR／sports 兩項退步。未改正式預設、未跑 Public 或進入 W11。結果：[H2B_RESULTS.md](C:/Users/yuting/Desktop/HW2_ADL/H2B_RESULTS.md)。
- **H3 完成（base同條件訓練與兩組官方比較）**：74tests、唯一正式base／242steps／相同W08 batch、離線重載、完整Gold與兩組原版Judge稽核完成。small5 Recall／MRR **0.846791／0.631014**，base5 **0.870600／0.658298**；base整體淨多23題，但Pet Recall98→93／102，超過預先最多退.02的門檻，是唯一採用檢查失敗。保留small預設；依gate未執行正式權重／8GiB cap驗證，沒有Public或W11。結果：[H3_RESULTS.md](C:/Users/yuting/Desktop/HW2_ADL/H3_RESULTS.md)。
- **H4 完成（依新授權接受Pet取捨並採用base5）**：77tests＋2項工具回歸通過；正式dev966題top50全數等同H3，dev／Public冷啟動185.421s／274.645s、8GiB allocator／RSS／時間門檻通過。於新Public前已提升並凍結base＋5:1；唯一完整Public原版Judge **Recall .933000／MRR .747915**，比small1+.031000／+.043567，淨多31題；9146比較、0解析失敗。仍Strong、估計Public6/8，未Boss；分組退步與48組共同候選Judge變化保存。沒有新訓練、dev Judge或W11。[H4_RESULTS.md](C:/Users/yuting/Desktop/HW2_ADL/H4_RESULTS.md)。
- **H5完成並採用**：84tests初期／切換預設後均通過；Train限定7728候選經Qwen篩選，3796題有1個pseudo-negative。唯一3864題／242steps續訓121.215s，FP32 checkpointing在8GiB allocator內、重載差0。完整Dev Gold門檻及唯一原版Judge通過，Recall **.893375**／MRR **.707191**，比基準+.022774／+.048893，淨多22題。正式預設H5＋5:1的966題top50全數重現、冷啟動175.912s／8GiB allocator通過，舊來源與權重保留。Sports Recall少1題但MRR提高，其他四品類兩項提高；本輪Public0次，未進W11。[H5_RESULTS.md](C:/Users/yuting/Desktop/HW2_ADL/H5_RESULTS.md)。

- **H6完成（固定H5的Public確認）**：完整1000／75k，唯一原版Judge Recall **.952／MRR .802404**、兩項達Boss，估Public8／8；9096比較／0解析失敗，原始結果與凍結配置稽核通過。比H4淨多19題，沒有訓練／挖負例／調參／改預設或W11。[H6_RESULTS.md](C:/Users/yuting/Desktop/HW2_ADL/H6_RESULTS.md)。

## Current Task

- **W11本機收尾完成，等待URL：**學號r15922a14，固定H5。完整兩階段3864題／各242steps重現，原H3／H5各199張量最大差0、模型檔SHA相同，8GiB allocator通過。88tests／8.490s通過；13頁report.pdf逐頁檢查、15筆Q2原文核對；README／run.sh／download.sh已更新。獨立提交目錄還原原H5，完整Dev966 top50全數相同／186.161s；portable run.sh兩階段smoke通過。原H5/H6 seal保留、不跑Public或Judge。**只差使用者的模型託管URL與實際遠端下載驗證，尚不標記正式提交就緒**。詳見W11_RESULTS.md／MODEL_UPLOAD.md。

- 歷史H5／H6階段當時沒有進入W11；其完成與封存紀錄保留，最新收尾狀態以上列W11為準。

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
| W10 凍結與 public 官方驗收 | 本機 public／分數分析完成；官方 3070／3.12.9 驗收待完成 |
| H2 微調 Dense／BM25 RRF 權重 | 完成八組 Gold 消融與獨立核對；4:1 為描述性最佳，尚未 graded／採用 |
| H2B 5／6／8／10:1 延伸消融 | 完成；唯一 6:1 dev Judge 整體兩項提高，分組有取捨；尚未接入正式推論／採用 |
| H3 BGE-base 同條件微調與5:1比較 | 完成；整體graded改善但Pet Recall保護門檻失敗，未採用／未進正式驗證 |
| H4 接受分組取捨後正式驗證與一次Public | 完成；採用base5，Public .933000／.747915，仍Strong，未Boss；正式硬體待驗收 |
| H5 Train難負例續訓 | 完成並採用；本階段Public0次，後續H6另驗證 |
| H6 固定H5 Public確認 | 完成；.952／.802404，兩項Boss，原始結果封存 |
| W11 報告、重現與封裝 | 本機完成；待URL／真正遠端下載驗證後產生正式ZIP |

## Test Results

- **W11：**88tests／8.490秒通過、下載4項測試與Shell／py_compile通過。兩階段完整3864題各242steps，73.795s／125.739s，allocated2.384／2.780GiB、reserved2.877／3.340GiB，重載差0；原H3／H5各199張量最大差0、model.safetensors SHA相同。獨立提交目錄本機archive解壓／完整Dev966 top50逐題等同H5，186.161s及8GiB allocator／RSS／時間通過；另portable run.sh兩階段4步smoke通過。13頁PDF逐頁render檢查、15筆Q2原文核對通過；原H5／H6 seal再次通過。遠端下載未驗證，URL仍待提供，不宣稱正式包已就緒。

- **H6完成：**py_compile及12項既有測試通過（grading5／measurement2／analysis5）；H5封存55份產物及10個checkpoint檔、官方保護檔與新Public前凍結來源一致。正式完整Public無覆寫top50／top10合法、272.101910s、allocated1.035655／reserved1.308594GiB、RSS2.190132GiB，8GiB allocator／<1h／<32GiB通過。Gold Hit10 .904／MRR10 .728475／Hit50 .968。
- **H6唯一原版Public Judge：**Recall **.952／MRR .8024043650793651**，952／1000題；9096比較／0解析失敗、prompt最長2399tokens不截斷，633.660281s。Raw output／gain／aggregate及排名／來源均核對；相對H4 Recall29改善／10退步，MRR196改善／83退步，五品類兩項皆提高。共同5970非gold配對54個gain變化，原因未隔離、未重跑。估Public8／8，Boss餘裕Recall2題／MRR+.002404；judge未cap且reserved9.095703GiB，不是8GB驗收。模型未改、Public訓練／挖負例／調參均0次；見h6_summary與h6_completion。

- **H5初期**：84tests／9.130s／OK（h5_tests_initial.log），新負例資料邊界、嚴格JSON、batch重複保護及獨立loss／gradient數值檢查通過。訓練計畫已凍結，尚無H5品質或顯存結論。
- **H5資料／smoke**：Train Dense冷啟動206.750s，7728候選獨立核對、Qwen篩選與3796對編碼不相同核對通過；Qwen0解析失敗。4步FP32 smoke／2.534s、64負例exposures、allocated2.733GiB／reserved3.021GiB、重載差0，8GiB allocator測試通過；不是實體3070驗收，也不是品質改善結論。
- **H5正式訓練**：全3864題／242steps／121.215s，3795次own-negative exposures、1次正例流重複略過；FP32／batch16／non-reentrant checkpointing／8GiB allocator，allocated2.750GiB／reserved3.305GiB，權重實際更新、重載差0。訓練完整性與loss／batch／checkpoint核對後已凍結Dev評估計畫；未以loss宣稱品質提升。
- **H5 Dev Gold**：完整966／50k新Dense100冷啟動169.448s；Dense-only .813665／.637067／.916149，固定5:1 .819876／.615763／.919255（Hit10／MRR10／Hit50）。1932份排名及Gold重算一致，五項事前gate皆通過，五品類三項皆提高；原版Dev Judge唯一一次進行中。Gold不等於官方graded或Public分數。
- **H5官方Dev／正式推論**：唯一原版Judge534.721s，8868比較／0解析失敗，max2285tokens未截斷，raw output／gain／aggregate通過；Recall .8933747412／MRR .7071909198，品質gate通過，命中淨多22題，第一名命中534→593。共同5819非gold配對有40個gain變化，兩版原始輸出保留。選定預設後84tests／8.350s再次通過；無覆寫正式966題top50全數重現，冷啟動175.912s、allocated1.036GiB／reserved1.311GiB／RSS1.890GiB，8GiB allocator／32GiB RSS／1h門檻通過。

- **H4**：完整77tests（8.921s／OK）＋工具回歸2tests（0.003s／OK）；正式dev966題top50與H3全數一致、無覆寫；dev185.421s／Public274.645s，兩者8GiB allocator／RSS／時間門檻皆通過。第一次dev量測parent失敗已保存並重測，未列為成功完整計時。Public Judge649.720s、9146比較／0解析失敗，逐題raw output／gain／aggregate核對通過；Gold獨立重算通過，h4_completion保存37份產物與checkpoint SHA，已再次核對全部指紋。

- H3 初期：**74 tests／8.372 秒／OK**（`results/h3_tests_initial.log`）；新增實際 W08 batch qid 完整一致、768 維重載、Gold／graded gate 及配對 bootstrap 測試。官方 base revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`、10檔／438900399 bytes，Git／LFS與本地SHA-256核對通過；四項 tokenizer 資產與原small完全相同。
- H3 smoke：實際四步FP32／batch16，權重更新、有限loss／非零gradient、離線重載向量最大差0；訓練迴圈1.490秒、CUDA allocated8628853760 bytes。這是工程驗證、不作為第二個品質候選，也不是8GiB推論驗收。
- H3 正式訓練：相同 W08 3864題與242批（末批8）、warmup25、1epoch、FP32／batch16／lr2e-5／MNR scale20；訓練迴圈57.869秒，CUDA allocated8796487680 bytes（約8.19GiB），離線重載最大差0。完整batch／loss／checkpoint hashes保存於`artifacts/training/h3_base_epoch1/`；尚未據此宣稱dev品質改善。
- H3完整Gold：base Dense-only 0.747412／0.556166／0.884058；base5 0.753623／0.550325／0.886128（依序Hit10／MRR10／Hit50），對small5 +29題／+.032405／+23題。1932份排名及整體／五品類／六來源Gold獨立核對通過，預定Gold gate全通過。base Dense冷啟動175.354秒。兩組Qwen preflight max2285 tokens、不截斷；small5有8961／base5有8932個非gold比較，後續兩組官方結果與稽核見下一項。
- H3官方Judge／稽核：small5 Recall **0.8467908903**／MRR **0.6310144106**，base5 **0.8706004141／0.6582980052**；命中818→841，Recall+57／−34、MRR+223／−168。兩組8961／8932比較、均0解析失敗，程序562.244／553.435秒；原始回答／gain／aggregate及固定參數核對通過。對small5的探索性qid-paired bootstrap 95% CI：Recall差[.004141,.043478]、MRR差[.009052,.045353]，仍為重用dev單seed描述，不宣稱未見泛化。
- H3唯一失敗gate：Pet Recall98→93／102（−.049020），允許最多退.02；Pet MRR反而提高.053591。Pet共有6題Recall退步、1題改善，6題原有被認可候選皆離開base top10；詳細案例保存於`h3_judge_diagnostics.json`。整體共有4715相同非gold配對，55個gain不同，原因未隔離，不重跑修分。
- H2B：**69 tests／6.003 秒／OK**（`results/h2b_tests.log`）。四組延伸配置 3,864 份排名通過獨立 rank-map 融合核對，1:1 再次逐題重現 W08 完整 top-50；H2 歷史產物／資料／來源／微調模型 hashes 全通過。原版 Qwen preflight 8,962 比較、最大輸入 2,285 tokens、不截斷；其後僅執行一次完整 Judge，完成與稽核結果見下一項。
- H2B Judge 已完成稽核：**Recall 0.8457556936／MRR 0.6294008183**，W08 為 0.8291925466／0.6118357488；命中 801 → 817（698 gold＋119 替代品）。Recall +52／−36、MRR +222／−172；8962 比較、0 parse failures，程序 564.747 秒。原始 output／gain／aggregate、batch 8／12 tokens／固定模型與凍結排名一致。4773 共同非 gold 配對有 60 個 gain 改變，原因未隔離；不追加 Judge 修分。`results/h2b_rrf_final.json`。
- H2：新增 7 tests，完整 **66 tests／5.983 秒／OK**；`results/h2_rrf_tests.log`。W08 checkpoint 10 檔、既有 BM25 來源／設定／trace／資料指紋通過。新 Dense top-100 冷啟動 105.101 秒；allocated 0.469 GiB／reserved 0.668 GiB，僅為 4090 開發量測。
- H2 重現與結果：微調後 Dense＋既有 BM25 的 1:1 **966 題 top-50 全數相同、0 題差異**；八組各題排序以另一公式獨立重算，**7,728 份排名**及五品類／六來源指標／逐題得失全部核對一致。4:1 對 1:1 Gold Hit10 +77／−23（淨 +54 題）、MRR10 +223／−95、Hit50 +38／−9。沒有呼叫官方 Judge、重跑 public 或更動預設；`results/h2_rrf_summary.json`／`h2_rrf_audit.json`。
- W10：**59 tests／6.117 秒通過**（`results/w10_tests.log`）；新增 public／private 級距權重、Recall／MRR 獨立累加、相等門檻雙重估算與非法指標拒絕測試。W08 checkpoint／官方三檔／public 來源 hashes 均通過核對。
- W10 public 檢索：原版入口完整 1,000 題／75,000 商品，無模型／檢索環境覆寫，top-50 與其正式 top-10 合法。冷啟動 163.892 秒，CUDA allocated 0.469 GiB／reserved 0.668 GiB、RSS 1.867 GiB；這是 4090／Python 3.12.12 開發環境，不是官方 3070 驗收。Gold Hit10 0.819／MRR10 0.615437／Hit50 0.928；`results/w10_w08_public_gold.json`。
- W10 Judge preflight：9,181 個非 gold 比較，原版 prompt 最長 2,399 tokens，context 上限 32,768、輸出 12；無截斷。`w10_judge_plan.json` 只授權凍結 W08 的一次初始 public Judge；尚未據 Gold 宣稱達到任一 graded 級距。
- **W10 原版 Judge／稽核完成：**Recall 0.902000（902／1000）、MRR 0.7043484126984128；819 題 gold＋83 題替代品。9,181 次比較／0 parse failures，raw output／gain／aggregate、排名與凍結來源一致。Judge 冷啟動 664.868 秒，CUDA allocated 4.653 GiB／reserved 9.875 GiB。Public Simple 4＋Strong 2＝估 **6／8**，Boss 未達；嚴格與含等號判斷相同，private／report 均未知。`results/w10_summary.json`。
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

- H6新Public .952000／.802404通過本機Boss，餘裕僅Recall2題／MRR+.002404；不是Private或整份作業成績保證。共同候選Judge變化與組批原因未隔離，Public也先前看過，不宣稱獨立未見holdout。H5只有一次品質訓練、單seed、重用Dev，且無等步數無負例續訓對照，改善歸於本次整體介入。Qwen pseudo-negative可能誤判，既有in-batch負例也未逐對人工驗證。
- H5續訓／正式推論在4090的8GiB allocator內通過，不等於實體3070驗收。W11已透過gradient checkpointing重現來源H3及H5相同權重、更新run.sh／README；原H3約8.19GiB的記憶體工程問題在本機cap驗證已解決。官方Dev Judge未cap，reserved9.396GiB，不能宣稱8GB Judge驗收。

- H4當時尚無最終訓練入口／封裝；W11現已補齊並驗證本機一致性，但遠端模型URL／實際下載仍待。H4 Judge reserved8.389GiB且未cap、開發8GiB allocator通過均不等同實體3070。

- H3當時整體分數提高，但Pet Recall超過事前工程保護門檻，故原H3未採用base；原gate仍維持失敗。後續使用者已依D28接受取捨，H4正式推論與8GiB allocator驗證通過後採用base，不把H3 gate改寫為通過。分組退步、實體官方硬體與訓練8GB限制仍須分別看待。
- H3歷史FP32訓練peak約8.19GiB；W11不改batch／精度，增加non-reentrant checkpointing後在8GiB allocator內重現相同模型檔SHA。實體3070／Python3.12.9仍未驗證，測試環境是否也約束重訓未獲額外官方答覆。
- H2B 的 6:1 整體 graded 兩項提高，但 pet MRR −0.076906、sports Recall −0.023077／MRR −0.012021、home Recall 少 1 題；toys Recall +7 題、MRR 約 −0.000070。5:1／6:1 Gold 差距小，同一 dev 多次選型；沒有已確認的新 Public／Boss 分數。正式推論權重尚未改，預設仍 1:1。
- H2 的 4:1 整體三項 Gold 都優於 1:1，但 pet-supplies Gold MRR10 0.651264 → 0.621168、sports Hit10 0.923077 → 0.892308；分組取捨已保留。Gold 不包含替代品判斷，尚無新 graded 分數或 Boss 達成證據；不得據此更新 W10 的 public 得分。
- PROJECT Q1–Q5 未獲新官方答覆；不將本次實作視為對免訓練、額外套件、cache 或 private 時限的批准。
- 開發使用 WSL Ubuntu 22.04／Python 3.12.12／RTX 4090；官方要求 Python 3.12.9／RTX 3070 8 GB，因此開發成功不能當作官方環境驗收。
- 後續大檔下載前仍須重新確認磁碟空間，不以 W00 舊量測當作目前可用容量。
- W06 下載後 C: 約剩 15.97 GB。整張 GPU 觀測到約 19–20 GiB 用量，但 BGE 程序自身 allocated 峰值僅約 0.47 GiB；WDDM／WSL 無法列出完整 GPU 程序清單，不能將整張卡用量歸給本作業，速度測量也不能視為無其他負載的硬體基準。
- 完整 dev 與本機 public 已完成，正式 3070／Python 3.12.9 效能結果尚無。超長對話的首尾處理主要由 tokenizer 替身測試驗證；真實 dev 對話全數短於上限，不能把這次品質評估當成長對話語意驗收。
- **8 GB 風險：**完整 judge reserved 峰值 8.631 GiB，包含 allocator cache；allocated 4.554 GiB。不能據此直接斷言 3070 會 OOM，也不能宣稱已通過 8 GB；後續須驗證，不能改 batch／截斷／judge 掩蓋問題。
- **Judge 組批敏感性：**相同 100 題、相同排名在 dev-fast／完整 dev 的不同 batch 搭配下，有 4 題 Recall 或 MRR 不同；確切原因未做隔離實驗。後續以完整 dev 與固定順序／batch 8 作主要對照。
- W04／W05 的額外 reviewer 工作因使用上限中斷，沒有完成的 review 結論；上述單元測試與真模型結果則已有可核對紀錄。
- W07 的原版 judge CUDA allocated 峰值 4.555 GiB、reserved 8.754 GiB；官方 8 GB 驗收尚未完成。MiniLM 不採用是因 dev 品質退步，不能將結論擴大為所有重排模型都不適用。
- H1 仍有歷史品牌干擾與 `1.6-Ounce`／`1.6 oz` 等格式差異；Hybrid 並未修復所有案例。BM25 完整商品與 BGE 512-token 表示不同，不能把改善完全歸因於單一公式。Judge reserved 峰值曾達 12.943 GiB（BM25 結果評分）／9.719 GiB（Hybrid 結果評分），allocated 約 4.57 GiB；8 GB 正式環境仍待驗收。
- H1U 的短指代保護未在真實 dev 觸發（0／966），只有手寫測試證據。User 回合中的歷史商品仍可能干擾；末輪「明確」是非固定短指代詞的近似判斷。不得把 Gold 六指標改善等同官方整體 Recall 改善。
- 已建立 W08 訓練／run.sh，以及 W09 `generate_report.py`、固定 prompts／設定、稽核、15 筆輸出與分析。`download.sh`、`report.pdf` 尚未建立，README 仍是開發紀錄而非最後封裝版，因此 Phase 3 與最終繳交均未完成。後續 download.sh 必須取得選定的微調 checkpoint，不能只下載原版 BGE；完整對話需求改寫、分段仍未實作。
- W08 使用同一 dev 選型，尚非獨立泛化驗證；不同 PID 的合理替代品仍可能是 in-batch false negatives。Judge reserved 11.178 GiB／allocated 4.569 GiB，官方 8 GB 驗收仍待完成，不能因本機訓練使用不到 8 GB 就視為全作業驗收通過。
- W09 三種 prompts 均有內容缺陷，包括跨商品保證／技術混用、產地與容量捏造、未證實限制說成滿足、將商品說明當使用者陳述，以及錯誤拒答；指定 PID 格式 0／15。證據與輸出均無截斷，不能把這五例的缺陷歸因於 token 上限；模型容量／提示詞機制的因果未隔離。五題與助理非盲評閱也不足以建立普遍生成品質結論。
- W10 public 已看過，之後任何 public 改進評測都不是未見過的獨立測試；不可用其 target 做梯度、負例或 qid／PID 規則。6／8 只為本機 public 累加估算，不是全作業 75 分或已確認總分；private、report、正式環境與格式扣分仍未知。Judge reserved 9.875 GiB／allocated 4.653 GiB，實際 8 GB 驗收仍待。

## Next Actions

1. 等待託管URL，收到後以空目錄實際下載原H5 archive、驗全部SHA，再生成正式r15922a14.zip。現有pending-url包不是最終可交版本；不自動提交NTU COOL。
2. 維持H5 checkpoint＋5:1及H5／H6歷史封存；W11本機重現／PDF／README／腳本已完成，不根據Public再訓練、調參或挑較高Judge分數。
3. 正式提交前由使用者確認內容並上傳課程平台；實體3070／Python3.12.9與Private仍未知，本機allocator測試不擴張為已完成官方硬體驗收。
