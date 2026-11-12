# W11 — 報告、訓練重現與提交準備

日期：2026-10-25（Asia/Taipei）  
學號：**r15922a14**。  
狀態：**本機收尾、文件整理、雲端下載與正式ZIP驗收完成；output/r15922a14.zip可供使用者上傳NTU COOL。**

正式封裝補記：使用者要求重新ZIP後，已建立22,621,343 bytes／628檔的r15922a14.zip，SHA-256 bb2c8201bbc1fe168061dccce9effef61cfb7d3ae601f8e95299c58143834e3d。CRC、逐檔SHA、投影片嚴格排列、LF與Unix執行權限，以及Ubuntu實際解壓／516份參考初始化均通過。證據為output/r15922a14.zip.audit.json及results/w11_final_completion.json。以下暫不封裝／等待URL敘述保留為歷史紀錄；本輪無新訓練／Public／Judge，也沒有自動上傳。

2026-10-25補記：使用者提供Google Drive模型資料夾後，已設定其中最終archive的直接HTTPS網址。Ubuntu空目錄實際下載395,347,153 bytes，19.97228451秒，archive及10個checkpoint SHA核對通過，見results/w11_remote_download_validation.json。README已同步，未建立新ZIP、未訓練或跑Public／Judge。新版13頁報告與Markdown已同步根目錄，文字整理見results/w11_editorial_review.json。以下等待URL的段落是本機收尾當時的歷史紀錄，現已完成下載驗證。

## 固定的正式模型

- 維持原H5 checkpoint＋Dense:BM25=5:1，正式推論程式／權重未改。
- H6已保存的完整Public：Recall@10 **.952000**／MRR@10 **.8024043650793651**。
- 本輪沒有新增品質選型、Public檢索或Public Judge；兩階段重訓僅做重現，不更換正式模型。
- H5／H6原artifact及checkpoint hashes再次核對通過，歷史seal與原始回答保留不改。

## 已完成

### 可攜訓練與兩階段實測

原`run.sh`只訓練舊W08 small；H3／H5歷史腳本的frozen計畫也依賴開發期檔案，不能當乾淨提交的入口。現在`run.sh`使用新的`code/script/train_release.py`，沿用原有query處理、sampler、loss及固定Train-only負例，不需要執行歷史品質gate或讀Public。

流程：固定官方raw BGE-base → H3相同一epoch → H5相同一epoch。每階段3864題、242steps、batch16、seed42、FP32；lr依序2e-5／5e-6、AdamW／wd.01／clip1／scale20、warmup25／linear decay。原H3 FP32約8.19GiB，重現版兩階段均使用non-reentrant gradient checkpointing與8GiB PyTorch allocator cap，沒有縮小batch或改變超參數。

| 實際完整階段 | 訓練迴圈秒數 | allocated GiB | reserved GiB | 與原模型最大參數差 |
|---|---:|---:|---:|---:|
| raw base → H3 | 73.794957 | 2.384078 | 2.876953 | **0** |
| H3 → H5 | 125.739494 | 2.779627 | 3.339844 | **0** |

逐一比較兩階段各199個參數張量，差異張量0；兩個`model.safetensors`的SHA-256也分別與原H3／H5相同。batch qids完整一致、全部訓練資料一次覆蓋／階段、H5實際3795次own-negative使用、重載向量最大差0。這是本機指定環境的實測，不能擴張成跨GPU／套件完全相同的保證。

先執行兩階段各4步工程smoke，再執行唯一一次完整兩階段重現；輸出在`artifacts/release_smoke/`、`artifacts/release_reproduction/`，不覆寫正式模型。另在獨立提交目錄透過真正的`bash run.sh --smoke`完成兩階段／4步，證明wrapper及路徑可用，無需原工作區歷史checkpoint。重現數據與逐步loss皆保存。

負例是已保存的派生訓練資料`configs/release_negatives.json`，3796題／每題最多1個；保留原7728教師回答、Train Dense排名與選取程式作provenance。`run.sh`不重新挖掘／重跑Qwen教師，也不把Pubic當訓練輸入。Qwen pseudo-label不等同人工真負例。

### 下載與文件

- `download.sh`僅呼叫stdlib下載／SHA檢查／安全解壓，不安裝環境、不訓練。
- 最終archive：`output/models/h5_final_r15922a14.zip`，**395,347,153 bytes**。
- archive SHA-256：`714d7a9be5486de5db8c59a8fba1a9f65d4e868f3c00b668500c20e5d8309768`。
- 內含原H5的10個檔案；相同檔案驗hash後重用，不覆寫不同模型，拒絕路徑穿越、額外成員與不符SHA的下載。
- 正式manifest `release_assets.json`記錄H5 archive／各檔hash及固定raw base／Qwen revisions；預設也下載訓練與Q2所需的官方資產，可`--skip-auxiliary`只取推論權重。
- README更新環境、資料、下載、正式命令、兩階段訓練、Q2重現及限制；舊run.sh／README／requirements另存`code/script/snapshots/w11_pre_release/`。
- requirements使用實際CPU FAISS，移除未被最終流程使用的datasets；報告套件另列requirements-dev，不由download.sh安裝。

### 正式報告

`report.pdf`共**13頁**，另有可編輯`report.md`與產製程式`build_final_report.py`。

- Q1：資料切分、多輪處理、商品表示、索引／RRF、兩階段訓練、負例與loss、超參數、成功／失敗消融、Dev／Public區別及資源限制。
- Q2：固定Qwen、原W08 top3證據、完整共通prompt與三種變體、五題逐例分析、客觀診斷與明示的非盲Codex主觀評閱。
- 附錄呈現全部15筆原始生成回答；以移除排版空白後的文字比對，15/15與原始輸出一致。保留捏造與格式錯誤，不改寫成較佳答案。
- Q2明確標示是歷史W08證據，沒有假稱H5生成結果。第二個例子的對話在PDF明示為摘要，完整原文與商品證據隨plan／evidence提交。
- 已實際render並檢視所有13頁：無裁切、重疊或缺字；PDF文字無replacement character。`results/w11_report_audit.json`記錄hash及文字核對。

### 乾淨提交目錄

於`output/submission/r15922a14/`放入明確allowlist的程式、設定、原始觀測與派生切分；不複製虛擬環境、秘密、原始Train／Public資料或模型權重到ZIP。測試時另提供官方Train檔案，實際從本機模型archive解壓下載流程，驗全部H5 hashes。

原版正式入口在該目錄跑完整Dev966／50k／top50：**966題全部與原H5相同、0題差異**；冷啟動**186.161275秒**，allocated1.035563／reserved1.310547GiB、RSS約1.880GiB，8GiB allocator／<32GiB／<1h門檻通過，沒有模型或檢索設定覆寫。這次沒有重跑Judge或Public。

封裝工具強制小寫學號最外層、必要檔案、每檔SHA／ZIP CRC及`.sh`的Unix可執行權限。只有manifest URL與實際遠端下載驗證都通過，才會輸出正式`r15922a14.zip`；否則只能產生明確命名的`.pending-url.zip`草稿。

## Test Results

- 新下載安全測試4項通過；加上既有測試，完整 **88 tests／8.490秒／OK**，見`w11_tests_all.log`。
- 新程式py_compile、`bash -n run.sh download.sh`通過。
- 兩階段4步smoke與兩階段完整3864題訓練、離線重載、199張量／模型SHA比較通過。
- 獨立提交目錄archive解壓與完整966題正式排名重現通過；在同目錄的兩階段run.sh smoke通過。
- PDF全部13頁視覺檢查及15筆原文比對通過。

## 尚待完成的最後一步

**等待使用者提供託管連結。**本機archive測試不等於實際網路下載。請上傳上述H5 archive，再提供可直接下載的HTTPS URL；詳見MODEL_UPLOAD.md。

取得後寫入manifest、以空目錄實際下載／驗hash並保存`w11_remote_download_validation.json`，重新封裝正式學號ZIP。連結需截止後至少保留3週，截止後不替換權重。沒有自動上傳程式、模型或提交NTU COOL。

實體RTX3070／Python3.12.9未驗證、Private未知；8GiB allocator不包含CUDA context／其他程序。本輪不因這些限制捏造硬體驗收，也不以Private／Public再選模型。

## 主要證據

- `results/w11_training_smoke.log`、`w11_training_reproduction.log`。
- `results/w11_reproduction_comparison.json`，各階段batch／loss／manifest。
- `results/w11_portable_smoke.log`。
- `results/w11_tests_all.log`、`w11_tests_download.log`。
- `results/w11_clean_dev_*`、`w11_clean_validation.json`。
- `results/w11_report_audit.json`。
- `results/w11_local_completion.json`：本機收尾封存，會明示遠端下載仍未驗證，不冒稱完整提交已就緒。
