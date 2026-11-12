# DECISIONS.md — RAG 架構決策紀錄

版本：v1.1｜更新日期：2026-10-24（Asia/Taipei）  
依據：[PROJECT.md](C:/Users/yuting/Desktop/HW2_ADL/PROJECT.md)；執行設計：[PLAN.md](C:/Users/yuting/Desktop/HW2_ADL/PLAN.md)。

本文件記錄「選什麼、為什麼、何時應改變」。早期條目的「採用」反映當時規劃，不等於已驗證或取得助教對未解問題的答覆；後續 W06／W07／H1 已記錄實際版本、品質與資源量測。實測狀態以最新紀錄與 PROGRESS.md 為準。

目前選型更新：W08 完整 dev 支持 **微調 BGE-small + H1 BM25 RRF**，兩項主要指標高於 H1，預設已提升且完整排名驗證通過。B0／H1 保留作對照，MiniLM 不採用。下列早期決策記錄維持其當時脈絡，已實作與量測的最新結果見文末與 PROGRESS.md；官方硬體與 public 仍待驗收。

狀態分為：**採用**、**條件式採用**、**暫不採用**、**待確認**。PROJECT 的正式要求優先於工程決策；線上模型卡只用來核對技術特性，不取代作業規範。沒有取得新的 NTU COOL 公告。

## 決策索引

| ID | 決策 | 狀態 | 對應 PLAN |
|---|---|---|---|
| D01 | 單機批次 RAG，檢索、judge、報告生成分離 | 採用 | §§1–3、6 |
| D02 | BGE-small 作為首個 embedding baseline | W08 微調同一模型並採用；原模型留作對照 | §§5、7 |
| D03 | 一商品一向量，先不做多段切片 | 採用 | §4 |
| D04 | 本地精確向量搜尋，不建向量 DB 服務 | 採用 | §§4、7、9 |
| D05 | 每題無狀態完整多輪表示，不建 session memory | 採用 | §5.1 |
| D06 | MiniLM top-50 reranker，必須由 dev 支持 | 已實測退步，暫不採用 | §§5.3、7 |
| D07 | 保留最小對比訓練分支，免訓練提交資格仍未知 | W08 已執行；課程原問題仍未取得答覆 | §7.4 |
| D08 | Train 內分組 80/20，public 留作凍結後確認 | 採用 | §8.1 |
| D09 | Offline 是邏輯階段，正式入口可冷啟動建完整索引 | 採用 | §§3–4 |
| D10 | 正式推論留在 student block，透過介面測行為 | 採用 | §§3、10–11 |
| D11 | 指定 Qwen 做 5×3 生成比較，使用同題同證據 | 採用 | §6 |
| D12 | 官方 graded 指標與開發診斷分開 | 採用 | §§8.3–8.4 |
| D13 | 最小依賴、固定模型與設定、推論不連網 | 採用 | §§2、9、11 |
| D14 | 最多優先一條進階路線，按品質與時間取捨 | 採用 | §§7.3、9.2 |
| D15 | GPU 工作分開執行，不以權重大小代替資源實測 | 採用 | §§6、9 |
| D16 | 文件、需求、實驗結果分層維護 | 採用 | 本文件末節 |
| D17 | Dense + BM25 等權 RRF 作開發版預設 | H1 完整 dev 兩指標改善，採用 | H1_RESULTS.md |
| D18 | BM25 user-only／末輪與 title 加權消融 | H1U 完成；U2 為取捨，維持 H1 預設 | H1U_RESULTS.md |
| D19 | W08 單一 BGE-small 微調實驗，H1 其他設定固定 | 採用；完整預設驗證通過 | §7.4 |
| D20 | W09 固定五題、同證據三種 Qwen 提示詞比較 | 實驗與分析完成；保留所有內容缺陷 | §6 |

## D01 — 主系統採單機批次，三種用途分離

**背景：**官方入口處理既有對話、輸出排名；Q2 才要求生成顧客回覆。固定 judge 可以看到 gold，但推論與生成不能依賴這項資訊。[PROJECT §§1、3、4]

**決定：**正式 retrieval 程序輸出 PID；官方 judge 獨立執行；Q2 generator 在保存的檢索結果上實驗。Offline／Online 是資料處理責任，不對應常駐服務。

**理由：**避免改動作業介面、避免 gold 洩漏，也不需要額外服務維運。

**未採方案：**聊天網站、API server、agent framework、工具搜尋、把 Qwen 當主流程的一部分。這些並非作業必要條件。

**重新檢討條件：**使用者另行要求互動式產品或正式服務部署；不因發現新框架而改架構。

## D02 — 第一個 embedding 選 BGE-small

**決定：**B0 預選 `BAAI/bge-small-en-v1.5`。其 384 維、512-token 設定可先以較小表徵建立基準；本次資料抽樣以英文為主。模型卡提供的 query instruction 是 `Represent this sentence for searching relevant passages: `，只加於 query，不加於商品。[官方模型卡](https://huggingface.co/BAAI/bge-small-en-v1.5)

**比較：**`intfloat/e5-small-v2` 同為 384 維候選，但需嚴格區分 `query:` 與 `passage:` 前綴；可作相同規模備選。[E5 模型卡](https://huggingface.co/intfloat/e5-small-v2) BGE-base 為 768 維，保留作有算力餘裕時的比較，先不預設它一定在本題更好。[BGE-base 模型卡](https://huggingface.co/BAAI/bge-base-en-v1.5)

**代價：**小模型可能忽略細節，且 512 tokens 可能截斷商品規格。英文模型的選擇是根據目前資料的工程判斷，不是課程保證所有未見資料均為英文。

**重新檢討條件：**模型不能在固定環境載入，或 dev 顯示主要瓶頸確為 embedding 表徵而非 query、截斷或排序。更換模型必須重建全部商品向量，不混用不同模型的 index。

## D03 — 一商品一向量，保留原始商品文字

**決定：**使用官方 passages 的商品文字，保守整理空白，以 PID 為唯一回傳單位。Baseline 使用模型 tokenizer 的上限，保留截斷診斷；先不將每個商品拆成多段。

**理由：**符合一個 PID 一個推薦單位，也避免 chunk 聚合、重複 PID、索引膨脹與新超參數。只讀掃描顯示 public 商品 P95 約 3,129 字元，但還不能據此推定 token 截斷比例。

**代價：**重要規格可能在文字後段。這是需要量測的風險，不能以「標題通常足夠」作保證。

**重新檢討條件：**實際 tokenizer 統計及錯誤分析顯示截斷是主要失敗來源，再比較保留首尾或少量分段；多段相似度必須聚合回 PID，正式仍回傳十個不同商品。

## D04 — FAISS exact search 足夠，暫不建向量資料庫

**決定：**商品與 query 做 L2 normalization，使用本地 CPU FAISS `IndexFlatIP`。FAISS 在此是向量搜尋函式庫，不是資料庫服務。API 以向量 row 索引回傳結果，應由固定 PID mapping 轉回商品 ID。[FAISS 文件](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexFlatIP.html)

**理由：**100k×384 的一份 float32 向量約 153.6 MB（自行計算，不含副本與模型），並不需要為向量容量建立獨立服務。精確搜尋也避免 ANN 漏召回成為另一個變因。

**備援：**以既有 NumPy／Torch 分批精確內積搜尋，兼作小型 oracle；必須測試正規化、排序方向、top-k 邊界同分與 PID 對照。

**未採方案：**Qdrant 類資料庫提供集合、儲存及過濾等能力，但本題沒有相應服務管理需求，新增依賴目前不划算。[Qdrant 官方文件](https://qdrant.tech/documentation/overview/) HNSW／IVF 也暫不導入。

**重新檢討條件：**實測證明 exact search 本身占主要時間，且非商品編碼／rerank 才是瓶頸；或者需求真的增加持續更新與多使用者查詢。

## D05 — Multi-turn 管理採完整語境、每題隔離

**決定：**Baseline 直接保留官方 role-tagged query 的順序與全文。`qid` 只用來對應輸出，不能作為預測特徵；不建立跨題的 session／長期記憶。

**理由：**既有對話很短，最後一句可能只說 yes、顏色或偏好。刪掉 assistant 或只保留最後一輪，會失去指涉與更正的語境。重新把展平字串解析成結構也有不必要的錯誤風險。

**語意政策：**區分歷史商品與當前需求、硬限制與軟偏好；後續明確更正優先；assistant 未被確認的選項不能變成使用者條件。但 B0 只是保留證據，沒有宣稱已實作可靠 slots extractor。

**未採方案：**每題先呼叫 LLM 改寫、長期向量記憶、主動追問。它們增加成本，且改寫可能刪除否定或補造限制。

**重新檢討條件：**固定 fixtures／dev 顯示多輪干擾是主要失敗來源，再加入有 evidence spans 的摘要或改寫；失敗時回退原文，與 B0 同資料比較。

## D06 — Reranker 是條件式改善，不是必備架構層

**決定：**預選 `cross-encoder/ms-marco-MiniLM-L6-v2`，先處理每題 top-50，若時間不足比較 top-20。其用途是 query–passage ranking；模型設定為 6 層、512 個位置，不能把長對話與長商品無限制拼接。[模型卡](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)、[模型設定](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/blob/main/config.json)

**理由：**把昂貴的 joint encoding 限制在候選集，是簡單可量測的兩階段方案。[Sentence Transformers 說明](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)

**限制：**模型在 MS MARCO 訓練，不保證適合本作業。Reranker score 只用來排序，不解讀成校準的 relevant 機率。初版不直接加總 embedding 與 cross-encoder 分數。

**採納條件：**候選召回足夠，且相同 dev 上的官方 graded 指標改善，冷啟動仍在預算內。若候選不存在好商品，先修 retriever。模型卡的 V100 數據不是本機 RTX 3070 吞吐量。

## D07 — 保留最小訓練方案，但不假裝 Q1 已有答案

**決定：**B0 可先建立技術對照；若課程要求訓練或 dev 支持訓練，啟用 A2：以 training partition 的 query–gold 商品對訓練同一小型 encoder，採 in-batch contrastive loss，起始設定見 PLAN §7.4。[MultipleNegativesRankingLoss 文件](https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html#sentence_transformers.losses.MultipleNegativesRankingLoss)

**理由：**不改部署形狀、可交代資料／loss／hyperparameters，也能用原本檢索入口測 checkpoint。先不引入 LLM fine-tuning、LoRA 堆疊或多 GPU 訓練。

**負樣本政策：**同 PID／相同商品文字不能互為 in-batch negatives；其他 PID 也可能是可接受替代品，因此「不同 PID」不是保證負例。先不用大規模 hard-negative mining，不用 public 或 dev 查詢挖訓練負例。

**重要代價：**微調可能破壞原本語意能力，loss 下降不等於正式指標上升。Gradient accumulation 不會自動擴大同一次 loss 的負例集合。

**待確認：**PROJECT Q1 仍未解答；無訓練策略是否可作最終提交、run.sh 與報告 loss 欄位如何處理，不能由本決策自行批准。若確定必須訓練，A2 升為必要工作並重估時間。

## D08 — 評估資料分組隔離，public 不用來反覆選型

**決定：**Train 內約 80/20，依相同 target PID 與完全重複對話的連通群組切分；若不同 PID 的商品文字正規化後完全相同，對應標註對話也併群。Seed、qid manifest 與資料 hash 固定。Dev 仍檢索完整 50k train 商品池。

**理由：**避免同商品 supervision／相同對話同時跨兩側；商品出現在可檢索庫不等於將 dev label 用來訓練。近重複檢查先採簡單抽查，避免建立新的語意去重系統。

**決策流程：**100 筆 dev-fast 先作診斷，最多兩個 finalist 跑完整 dev judge；策略與參數凍結後才跑完整 public。資料切分只提供較可信的比較，不保證等同 private 分布。

**禁止：**public target 進入梯度、query construction、negative mining、手工逐題規則或推論 lookup；生成也不能從它取得額外商品證據。[PROJECT §§3、5、8]

**重新檢討條件：**發現重複污染就修正 manifest 並重跑比較；不能因某次 dev 分數不好而反覆換 seed 挑最有利切分。

## D09 — 正式入口能完整冷啟動，不依賴預建 private 索引

**決定：**正式呼叫收到 passages 後，就能編碼全部商品並建完整索引。開發期預建 index 可縮短重複實驗，但可完全關閉。

**理由：**助教只明訂官方 retrieval 命令，private 有新增商品；不能假設助教會執行額外 build-index，也不能只封裝 public 75k 的索引。

**計時政策：**模型載入、商品編碼、建索引、query encoding、search、rerank、寫檔全部算入我們的冷啟動量測。官方 public 一小時是硬限制；45 分鐘只是自行保留餘裕的目標。Private 時限仍待確認。

**Cache 政策：**若後來允許正式使用 cache，其 key 必須涵蓋商品 ID 與內容、模型／tokenizer／前處理與 normalization 等；新增、刪除或變更失效時先全量重建。不能靠 cache 省略未見商品。

**重新檢討條件：**有明確課程答覆允許分開預處理或指定 cache 流程；把來源附到本紀錄後才能調整驗收方式。

## D10 — 邏輯模組化，不突破 student block

**決定：**正式 imports、helpers 與流程留在指定區塊，不改簽名、常數、loader、writer 或 evaluator；目前不依賴自訂外部 inference modules。[PROJECT F03–F04；retrieval_utils.py 的標記區塊]

**測試方式：**對公開函式使用小商品池及可控制的第三方 encoder／index 替身，擷取 query／商品文字與已知向量，驗證行為；另保留真模型／真 FAISS smoke。即使 helper 是 nested，也能獨立驗收職責。

**未採方案：**為直接測 nested helpers 加 AST extraction、程式產生器、兩套同步演算法或自建框架。測試替身只證明契約，不代表通過語意品質或效能驗收。

**其他程式：**訓練、報告生成與分析腳本獨立放在 code/script，符合報告與重現需求；這不等於已獲准由正式推論 import 任意自訂模組。

**重新檢討條件：**PROJECT Q3 得到明確允許且程式複雜度確實需要拆檔，再重構；不是為套用慣用專案模板就先拆。

## D11 — Q2 固定模型、固定證據，主動做 15 格比較

**決定：**Qwen2.5-1.5B-Instruct，5 個預選 dev 樣本，每題使用同一組 top-3 檢索商品，在三種 prompts 下生成。5×3＝15 是我們的實驗設計，不冒稱投影片明訂最低 15 筆。[PROJECT F07–F09、Q5]

**公平比較：**同一模型 revision、證據順序、context budget、greedy decoding 與輸出長度，僅改 prompt 策略。分別測直接推薦、需求與證據對應、證據不足表達。

**資源決策：**先以總 4,096-token budget、最多 256 output、batch 1 運作，遠低於該模型卡列出的 32,768-token context；是否安全仍需實測。[Qwen 模型卡](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

**評估決策：**使用自訂 0–2 的質性量表與具體例句，不假裝是官方量化 rubric。Gold 不進 prompt；retrieval miss 如實保留，不能補 gold 後稱作 RAG 成功。

**重新檢討條件：**助教對樣本格式給出新規則，或所選固定樣本無法覆蓋預先訂定的測試情況；任何更換必須說明原因，不以輸出好壞挑樣本。

## D12 — 官方 graded 指標不與 gold-only 診斷混用

**決定：**正式以未修改 evaluate.py 的 `graded_recall@10` 與 `graded_mrr@10` 驗收。前者是前十名至少一個可接受商品的題目比例；後者是第一個可接受商品排名倒數的平均。[PROJECT §8.3；evaluate.py]

**診斷用途：**Gold-only Hit@10／MRR@10 與 candidate Hit@50 可便宜比較候選／排序，但漏掉單一 gold 並不證明推薦錯誤。

**禁止的捷徑：**不換 judge、不改 gain、不修改 parse-failure 計分、不把 top-10 相關項數除十當 Recall、不刪除難題。正式評估依指定 batch 8；本地較小 batch 的診斷結果須標示不同條件。

**模型選擇：**比較同一 dev 的兩個正式指標及時間；若有取捨，列明對各作業級距的影響。沒有證據支持複雜模型時保留簡單方案，不任意造一個加權總分取代官方配分。

## D13 — 使用既有依賴，正式推論不連網

**決定：**優先現有 torch、transformers、sentence-transformers、numpy 與 faiss。固定實際 model revision、依賴版本、preprocessing、seed 與設定；正式執行讀已下載模型。

**理由：**套件清單與投影片有差異，測試環境不允許由 download.sh 改動；少加依賴可減少相容性風險。線上文件可能描述新版 API，因此不直接依其 `pip install -U` 範例升級固定環境。

**未採方案：**新增向量 DB、LangChain／LlamaIndex 等 orchestration 依賴、付費 API、需要即時連線的模型服務。這是本題的成本選擇，不是認定這些工具普遍不好。

**資產政策：**download.sh 只下載模型與確實使用的外部資料；本計畫不新增外部訓練資料。訓練、套件安裝及建索引不放下載腳本。固定下載內容並符合截止後三週連結有效規範。

**待確認：**PROJECT Q2 的正式套件版本與安裝流程；模型 revision／檔案大小／hash 要在 W01 真正取得後記錄，本次不虛填版本識別。

## D14 — 限制實驗數量，以交付完成度優先

**決定：**B0 + Q2 必先可跑；通常只追加一條有 dev 證據的進階分支。課程若要求訓練，A2 優先；否則候選召回夠好時先試 A1。

**理由：**截止時間近，完整報告、可重現腳本與封裝也是正式評分內容。多個 embedding、hybrid、rewrite、reranker、fine-tuning 一次疊加會增加成本，也無法知道改善來源。

**暫不採用：**ANN、量化、分散式、知識圖譜、大模型重排、多輪 agent、長期記憶及大範圍超參數搜尋。BM25 hybrid 與 query rewrite 也只在對應錯誤型態成立時重開。

**成本：**最小路線估 11–20 人工作時，不含 GPU、下載、排隊與環境故障；每條進階另估 3–6 小時及未量測算力。這是規劃區間，不是交付時間保證。必要訓練升級後須重估。

**停止加功能條件：**會侵占至少六小時的最後報告／封裝／乾淨重跑餘裕，或進階改善不明顯、資源不合格。停止的是額外功能；必交的 Q2、報告與重現不能被當成選配砍掉。

## D15 — 按程序分配 GPU，所有速度聲明須有測量

**決定：**商品／query 編碼、選用 rerank、官方 judge、Q2 generation 依序使用 GPU。程序結束或明確釋放模型後再載下一個，初版不做多模型常駐。

**理由：**8 GB VRAM 下，權重只占部分空間，prompt padding 與執行暫存也可能導致 OOM。Qwen 1.54B 的 2-byte 權重粗估約 3.08 GB，不能據此宣稱任何 batch 都放得下。[參數依據](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

**量測：**獨立記錄 encode products/sec、rerank pairs/sec、judge pairs/sec、冷啟動總時間、峰值 allocated／reserved VRAM、RAM 與磁碟量。模型卡的硬體與長度條件不同，不作本題速度承諾。

**故障策略：**開發期可以調 batch 並記錄；凍結後缺模型、格式錯誤、NaN、失效索引要明確失敗，不靜默輸出任意商品。正式 judge 的 batch／不截斷行為不由我們改動。

## D16 — 需求、計畫、決策與實驗紀錄各司其職

**決定：**保留 PROJECT 為 Phase 1 需求基準；PLAN 記錄元件與 W00–W11；DECISIONS 記錄選擇、狀態與重開條件。沒有實驗前不填品質或 GPU 時間數字。

後續維護規則：

1. 每次決策變更寫日期、原決策、新決策、觸發證據及影響的工作項目。
2. 只有官方說明改變時，才同步修訂 PROJECT 的 confirmed requirements，並附可查來源；工程偏好不升格為課程規定。
3. 實驗結果另保存配置、split、模型／資料 hash、原始輸出與 metrics。PLAN 的「待執行」只有真的完成驗收後才改成完成。
4. 保持一份明確凍結的 final configuration；不要在 README、run.sh、inference 之間出現不一致版本。

## 尚未解答的課程問題及影響

| 問題 | 相關決策 | 現在能繼續的工作 | 最終需要確認的事項 |
|---|---|---|---|
| PROJECT Q1：訓練必要性 | D07、D14 | 建 B0 技術對照及準備 A2 設計 | 免訓練是否合格；run.sh／loss 欄位如何交代 |
| PROJECT Q2：環境差異 | D02、D13 | 先依最小依賴做相容性 smoke | 助教最終版本及額外套件安裝機制 |
| PROJECT Q3：額外 module／cache | D09、D10 | student block 自足、cache 可停用 | 是否允許自訂推論模組與預建資產 |
| PROJECT Q4：計時／連網／private | D09、D15 | 全冷啟動計時，模型預先下載 | 官方計時邊界、private 時限與網路規則 |
| PROJECT Q5：生成樣本矩陣 | D11 | 直接規劃 5×3 同證據比較 | 有無額外官方展示格式；不影響目前 15 格設計 |

這些未知不阻止完成本次架構文件，但不能在最後繳交時偽稱已得到官方許可。

## 本次檢查與版本紀錄

| 日期 | 變更 | 驗證狀態 |
|---|---|---|
| 2026-10-23 | 建立 v1.0：B0、A1／A2 條件分支、G1、資料／評估協議及測試責任 | 已核對 PROJECT／starter 介面、官方模型卡／函式庫文件與資料字元長度；未執行模型 |
| 2026-10-23 | Phase 3 W00：採既有 WSL Ubuntu 22.04 的 ADL-final 作開發基礎，專案內隔離 overlay；CPU FAISS 不改索引架構 | 原環境的主要 ML 套件／CUDA 已實際 import，faiss 缺漏；Python 3.12.12、RTX 4090 與官方不同，不宣稱正式驗收完成。後續見 PROGRESS.md。 |

外部來源核對於 2026-10-23，均為模型發布者或函式庫官方文件。Phase 2 沒有下載權重或執行實作；Phase 3 的實際安裝、模型與測試以 PROGRESS.md 及 results/ 為準。W01 先驗證 embedding；Qwen 在 W06／W09 首次使用前驗證，避免一次下載尚未使用的資產。

## 2026-10-24 — W06 實測後的決策狀態

- **B0 保留為已量測的對照，尚未宣告為最終提交方案。**完整 dev 966 題／50k 商品，原版 judge 得到 graded Recall@10 0.652174、MRR@10 0.396446；gold Hit@10 0.468944、Hit@50 0.649068。這是 dev 結果，不能直接套用 public 級距。
- **本階段不修改 B0，也未啟用 A1／A2。**第 11–50 名含更多 gold，且有尺寸限制錯誤；同時約 26.98% 商品文字遭 512-token 截斷。這些是後續單獨實驗的依據，不代表已證明 reranker、訓練或分段哪一種較好。
- **固定完整 dev 順序／batch 8 作後續主要評估。**相同 100 題的檢索排名完全一致，但在 dev-fast／完整 dev 的不同組批下，有 4 題的 judge 指標不同；保留兩次 raw outputs，不把這種差異當模型改進。
- **8 GB 相容性仍待驗證。**完整 judge 的 allocated 峰值 4.554 GiB、reserved 8.631 GiB；cache 與活躍張量不同，不能直接推定在 3070 成功或失敗。原版 judge、batch 與不截斷規則未改動。
- **Qwen revision 已鎖定**為 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`；實際下載與 batch 8 smoke 均成功。未增加 API 服務或更動全域套件。

證據與重跑方式見 [W06_RESULTS.md](C:/Users/yuting/Desktop/HW2_ADL/W06_RESULTS.md)；機器可讀結果見 `results/w06_summary.json`。W06 已完成開發環境評估，W07–W11 與官方驗收仍依 PROGRESS.md 追蹤。

## 2026-10-24 — 啟動 W07 A1 單獨比較

使用者授權制定改善計畫並修改程式。先執行原 W07 的 MiniLM top-50 重排；W06 中 gold 位於第 11–50 名的比例支持做此實驗，但不代表效果已證實。先不更換 embedding 或同時加入 BM25／chunking。模型以固定 revision 下載，利用既有 transformers 的 sequence-classification API 執行 cross-encoder，明確組合 query／商品 token 預算，不新增套件。

正式推論只修改 student block，以環境變數切換 B0／A1；完成固定完整 dev 的原版 graded 評估後決定是否設為預設。雙指標退步則不採，互有得失則保留 B0 並記錄。所有比較保留原始輸出與冷啟動／資源資訊，不將開發 4090 結果視為官方 3070 驗收。

## 2026-10-24 — W07 結果：不採用 A1，H1 作為下一個優先實驗

完整 dev 966 題／50k 商品、原版 evaluator：B0 graded Recall@10 0.652174、MRR@10 0.396446；A1 為 0.606625／0.327259。Recall 有 76 題改善、120 題退步。A1 冷啟動檢索 170.393 秒，B0 為 105.082 秒。兩項主要指標都退步，因此 **維持 `HW2_RERANK=0` 預設**；模型與程式保留供重現，不作最終推薦配置。

22 個測試與完整稽核通過；B0 top-50 重跑與原檔案 hash 相同，A1 沒加入候選集外商品。Pair 分析顯示 45.04% 候選商品被截斷，但 gold 可完整放入的組別也退步；因此不將截斷當成已證實的單一原因。固定案例中有當前身體／頭髮需求被過去眼霜、argan oil、品牌商品干擾的現象，仍需對話表示消融才能判定原因。

下一個優先比較為 H1：BM25-only、Dense + BM25 RRF，保持原 query、關閉 MiniLM，檢驗詞面資訊。若歷史詞面干擾持續，再單獨比較 QRY1；分段與訓練仍保持獨立。這是基於實測調整可選分支的順序，未宣稱 H1 已實作或有效；課程若確認必須訓練，仍以 A2 優先。詳細數據、資源限制與重現指令見 [W07_RESULTS.md](C:/Users/yuting/Desktop/HW2_ADL/W07_RESULTS.md)。

## 2026-10-24 — 啟動 H1 BM25／RRF

使用者授權下一實驗。沿用 PLAN 的固定 BM25／RRF 起始參數，以既有 NumPy 和標準函式庫實作倒排與排名融合，不增加服務或套件。這輪使用相同完整 query、不加 MiniLM；BM25 使用完整商品文字，因此與 BGE 長文截斷的差異也屬本輪系統比較的一部分，不能把成績差異完全歸因於單一搜尋公式。先維持 dense 預設，完整 dev 的官方 graded 結果支持後才決定採用。

## D17 — H1 結果：採用 Hybrid RRF 為開發版預設

完整 dev 966 題／50k 商品、原版 Qwen judge、固定順序與 batch 8：B0 Recall@10 0.652174／MRR@10 0.396446；BM25-only 0.650104／0.450887；**Hybrid 0.739130／0.465780**。Hybrid 兩項均提高，符合預先定義的採用政策；BM25-only 是 Recall／MRR 取捨，不選作預設。

Hybrid Recall 比 B0 高 8.70 個百分點，命中題數 630 → 714；132 題改善、48 題退步。檢索冷啟動約 105.499 秒，與 B0 的 105.082 秒接近；這是開發機觀測值，非隔離 benchmark。所有設定固定，沒有依 qid 分流、使用 dev 標籤訓練或為特定案例硬篩商品。

正式 student block 的 `HW2_RETRIEVER` 預設改為 `hybrid`；保留 `dense`／`bm25`，MiniLM 保持關閉。被評估的 source 保存於 `code/script/snapshots/h1/`；切換預設後完整 dev 966 題排名全數一致，28 個測試與最終稽核通過。歷史 B0／A1 重現指令明確指定 mode，避免新預設影響舊實驗。

原始 judge output／gain／aggregate、資料與來源 hash、候選合法性、B0 回歸已通過稽核。仍有歷史品牌干擾、複合單位分詞差異與 judge 組批敏感性；8 GB 正式環境與 public 尚未驗收，不能將 dev 改善當成最終提交保證。完整數據及重現指令見 [H1_RESULTS.md](C:/Users/yuting/Desktop/HW2_ADL/H1_RESULTS.md)。

## 2026-10-24 — H1U 僅改 BM25 輸入表示

使用者限定 Dense 原文不變，不進入訓練或 reranker；授權 BM25 user-only 與極短指代補前一 assistant 的保護。依序比較 U0／U1／U2 末輪 2 倍／U3 title TF 2 倍；不組合兩項加權。完整 Gold 的整體與 beauty-train 三指標都不下降、每組至少一項提高才進入一次官方 Judge，最佳配置排序與預設提升條件在 PLAN 預先固定。開發快篩共用一次 Dense top-100，正式勝出配置另驗證端到端一致，不將分析 cache 時間冒充正式冷啟動。

## D18 — H1U 完成：保留 U2 可選方案，維持 H1 預設

Gold 快篩中只有 U2（user 回合＋最新非短指代 user 詞項 2 倍）同時改善整體及 beauty-train 三指標。U1 直接刪 assistant 與 U3 額外加 title TF 的整體 Hit10 都下降，因此不送官方評分。正式無 cache 管線全 966 題 top-50 與快篩一致後，僅 U2 執行一次原版 evaluate.py。

U2 graded Recall@10 **0.737060**、MRR@10 **0.480002**；H1 為 0.739130／0.465780。雖 MRR 提高 0.014222、beauty-train 兩項微幅提高，整體命中 714 → 712 題，未通過事先的非退步門檻。**不變更 H1 預設、不再評另一組配置。**新增程式保留 users 模式、指代保護、末輪／title 權重開關，預設仍 full／1／1。

37 個測試與來源／原始結果稽核通過，沒有改 Dense 輸入、模型、RRF、訓練或 reranker。真實 dev 的短指代保護觸發 0 題，只能據手寫測試說明其行為。共享的 6,051 個非 gold 配對有 47 個 Judge 判斷不同，原因未隔離，不把 2 題差距解讀為確定顯著退步，也不重寫本次官方分數。數據、逐題得失與單次 Judge 紀錄見 [H1U_RESULTS.md](C:/Users/yuting/Desktop/HW2_ADL/H1U_RESULTS.md)。

## D19 — W08 單一 BGE-small 微調實驗

使用者已授權在 H1U 之後執行 W08。目的是利用已有 train 3,864 筆監督資料驗證 encoder 適應商品對話的效果，而不是無限制調整 BM25 或從頭預訓練模型；免訓練提交資格仍未取得官方答覆，本次授權不代表課程強制訓練的規定已獲確認。

沿用既有本地 BGE-small；以完整對話／gold 商品對執行一個 epoch 的 FP32 對比訓練，batch 16、AdamW lr 2e-5、cosine MNR loss scale 20、seed 42。同批排除相同 PID／商品文字，不挖 hard negatives。H1 的完整對話、商品表示、BM25 full／1／1、兩路 100 與 RRF 60 固定，U2、分段與 reranker 不加入。訓練入口使用現有 sentence-transformers 5.1.0 與 torch，不新增 trainer 依賴。

已完成 43 個測試、真模型 smoke、正式入口驗證與 242-step 訓練，checkpoint 離線重載一致。完整 dev Gold 六項門檻均提高，唯一一次原版 Judge 的 Recall@10 **0.829193**／MRR@10 **0.611836**，H1 為 0.739130／0.465780；beauty-train 也提高，9,017 次非 gold 比較／0 parse failures。符合事先政策，**採用 W08 微調 checkpoint，H1 的其他設定固定**。舊來源保存於 `snapshots/h1u/`，歷史結果重現需指定原 BGE 路徑；切換後 43 tests 通過，完整 966 題無覆寫排名與被評估配置全數一致，最終 `default_verified=true`。完整配置、紀錄與採用結論見 [W08_RESULTS.md](C:/Users/yuting/Desktop/HW2_ADL/W08_RESULTS.md)。

## D20 — W09 完成：固定同證據比較，誠實保留生成失敗

使用先前鎖定的五個無 target 樣本，凍結 W08 每題 top-3；指定 Qwen revision／bf16／greedy／batch 1／256 new tokens，三種 prompts 共用相同對話與證據。生成端只讀商品、無標籤樣本與凍結排名，沒有 gold 補入、Judge 篩選或看過輸出後換題。先保存完整 prompt／input IDs，逐筆保存 raw text／output IDs；完成後拒絕覆寫，允許驗證後續跑未完成的前綴。

54 tests 與完整 token／來源稽核通過，15／15 EOS，證據 0 截斷。但三種 prompts 都有內容錯誤：P2 增加未提供的退款期限、P3 把不同商品保證移植或無根據拒答，狗零食案例甚至把 Nepal 說成 USA。所有原始回答保留，不補 PID、不改文案、不追加較好的一輪。PID 格式 0／15 遵守，也說明「沒有未知 PID」不是品質保證。

助理以自訂五維 0–2 量表逐例檢閱（非使用者標註、獨立評審或盲測），P1／P2／P3 的五題平均總分為 5.8／4.4／3.2；僅為這五例的描述性比較，不作正式分數、及格門檻或普遍優劣結論。W09 成果是程式、完整比較與分析；**不選用任何 prompt 當成已驗證可靠的正式生成預設**。正式 retrieval 仍為 W08，未改 official Judge。詳見 [W09_RESULTS.md](C:/Users/yuting/Desktop/HW2_ADL/W09_RESULTS.md)，接續工作為 W10／W11。
