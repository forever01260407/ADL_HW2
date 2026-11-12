# PROJECT.md — ADL 2026 HW2 Phase 1 Requirement Analysis

## 2026-10-25 助教討論區補充（後續確認，取代下方對應待確認項）

來源：使用者於本對話貼上的 [NTU COOL HW2 討論區回覆](https://cool.ntu.edu.tw/courses/64983/discussion_topics/530817)，本次未登入獨立擷取網站。以下與本機提交程式交叉核對；原Phase 1內容保留為歷史。

- Confirmed：助教會實際執行run.sh；訓練沒有時間限制，但同樣必須能在RTX3070的8GB VRAM限制內執行。不能只證明檢索推論省記憶體。
- Confirmed：助教將原始data/（含train/）放入提交的學號根目錄；相對路徑data/train/...符合預期。
- Confirmed：助教依requirements.txt安裝套件；download.sh仍不可pip install或修改套件環境。
- Confirmed：可使用官方Qwen2.5-1.5B-Instruct標註Train候選，再訓練檢索模型；Public／Private不得用於訓練的限制不變。
- Confirmed：必交檔案依投影片File Layout；官方先跑retrieval.py，再跑evaluate.py；重現訓練需同方法且效能相近，不要求位元完全相同。
- Confirmed：Private與其他split均為英文。
- 本地核對：目前正式模型是BGE-base＋BM25 5:1，沒有使用原發問者的Qwen3-Embedding／Qwen3-Reranker架構。run.sh使用低記憶體兩階段訓練，兩階段模型檔已完整重現。早期未最佳化實驗曾超過8GiB，不得將歷史實驗一概標為符合8GB；正式重現以train_release.py為準。

分析日期：2026-10-23（Asia/Taipei）  
狀態：**Phase 1 已完成；尚未進入架構設計、實作、訓練或正式評估。**

本文件整理工作區內全部提供資料：25 頁投影片 PDF、完整錄音文字稿、README、套件清單、四份 Python 程式與四份 JSONL 資料。投影片已完成全文閱讀與頁面圖像核對；JSONL 已逐行解析、檢查結構及數量，並抽樣閱讀內容，並非逐項人工語意審閱所有商品。尚未讀取 NTU COOL 討論區或雲端資料夾的線上更新，不能宣稱本地文件是最新公告版本。

引用方式：`[S1 p.10]` 表示投影片 PDF 第 10 頁；`[S2 271.5–286.5s]` 表示文字稿時間；`[C2 L26–37]` 表示程式行號。完整檔案對照見 Source References。頁碼以 PDF 從封面起算的實際頁序為準。

## 1. Project Overview

### 1.1 專案背景與核心任務

這是 ADL 2026 Homework 2，主題為 **Retrieval-Augmented Generation（RAG）**。情境是顧客與 salesbot 對話後，系統從指定商品池找出符合需求的商品規格與商品。核心作業介面是離線、批次處理既有對話，再輸出每段對話的商品排名。[S1 pp.1、5–11；C1 L43–57]

本作業有兩個必須分開理解的工作範圍：

| 範圍 | 已確認的任務 | 對應成果 |
|---|---|---|
| 主要系統／模型效能評分 | 根據完整對話檢索商品，輸出 top-10 排名，由固定評估程式評分 | `retrieval_utils.py`、可由官方入口產生的排名 JSON、可重現的模型與相關程式 |
| 報告 Q2 的生成實驗 | 使用 Qwen2.5-1.5B-Instruct，根據檢索出的商品規格生成顧客回覆，分析不同提示詞 | `report.pdf` 的 Q2、生成實驗相關程式及檔案 |

因此，「主要評分流程不做回覆生成」不代表「整份作業不用做生成」。文字稿前段的說明必須搭配後段報告要求一起閱讀。[S1 pp.6、20、24；S2 92.3–139.3s、839.2–910.6s]

### 1.2 使用者需求與 RAG 的角色

顧客可能在多輪訊息中才逐步說明商品用途、偏好與限制。系統需利用這些資訊，從商品池中找出真正能滿足需求的商品，而不只是字面相似的商品。[S1 pp.5–6；C3 L28–53]

RAG 中的檢索負責存取外部商品知識；排序結果在主要作業中直接接受評分，在報告 Q2 中則成為生成回覆的依據。固定 LLM judge 是**評估工具**，與 Q2 的**回覆生成實驗**是不同用途，即使指定模型相同也不能混為一個步驟。[S1 pp.11、24；C3 L252–277]

### 1.3 任務邊界：哪些涉及、哪些沒有被要求

| 項目 | 分析結論 | 依據 |
|---|---|---|
| 多輪對話 | 必須處理資料中既有的上下文；不是只有最後一句問題 | C4 L6–12、L28–39；D1、D2 |
| 意圖與限制理解 | 屬於完成檢索所需的能力；沒有要求額外輸出意圖分類標籤或建立獨立分類器 | S1 pp.5–6；C3 L28–42 |
| 商品推薦 | 要輸出商品 ID 的排名；主要介面沒有要求直接輸出自然語言推薦說明 | C1 L16–31；C2 L29–34 |
| 即時追問／對話管理 | 沒有列為必做；對話已由資料集提供 | S1 p.5 是 Scenario；C4 L6–12 |
| 生成回覆 | 報告 Q2 必做；不在主要 retrieval 輸出契約中 | S1 p.24；C1、C2 |
| Query rewriting | 可研究的策略，未指定為必要元件 | S1 p.6 圖示；S2 106.3–117.3s、803.2–812.2s |
| 獨立 reranker | 可選；現有 student block 明寫 optional reranking | C2 L26–34 |
| Web UI、API 服務、登入、購物車、交易、長期個人化 | 未在提供文件中列為作業要求 | S1 pp.5–11、15、23–24 |
| 腰痛判斷、醫療問答、椅子專用推薦 | 不可由示範情境推導為強制功能 | S1 p.5 |

## 2. Problem Statement

給定一段在推薦前截止的顧客與助理對話，以及指定 split 的商品池，系統須找出符合顧客目前核心需求及明確限制的商品，並為每段對話回傳十個不重複、有效且有序的商品 ID。

每段對話雖只標註一個 gold product，商品池中仍可能存在其他合理推薦。作業以 gold label 加上固定 LLM 的替代商品判斷來評估，因此「只命中唯一標籤」與「找到可接受商品」需要區分。[S1 p.8、p.11；C3 L20–22、L291–317]

另一個挑戰是商品池會從 train 的 50,000 項擴大到 public 的 75,000 項與 private 的 100,000 項，不能假設所有測試商品都在訓練商品池內。[S1 p.9]

## 3. Expected Inputs and Outputs

### 3.1 資料集與本地檢查結果

| Split | 對話數 | 商品數 | 商品池關係 | 本地狀態 |
|---|---:|---:|---|---|
| Train | 4,830 | 50,000 | Base pool | 已全檔解析核對 |
| Public test | 1,000 | 75,000 | Base 50,000 + New 25,000 | 已全檔解析核對 |
| Private test | 2,000 | 100,000 | Public 75,000 + New 25,000 | 官方不提供；僅依投影片記錄 |

來源：[S1 pp.8–9；D1–D4]。

本地觀察如下，這些是提供資料的事實，不是額外制定的規格：

- 四個 JSONL 均可解析；各檔的主要 ID 沒有重複或空值。
- Train 的 2,415 筆對話有 3 則訊息，另 2,415 筆有 5 則；public 分別為 510 筆與 490 筆。角色只有 `user` 與 `assistant`。文字稿稱約 2–3 輪，實際結構是推薦前的 3／5 則訊息，不應錯當 3／5 個完整問答回合。[D1、D2；S2 161.3–178.3s]
- Train 與 public 的每個 `target` 都能在各自商品池中找到；對話沒有空 conversation。
- Public 完整包含 train 的 50,000 個商品，這些共同商品的內容一致，另外增加 25,000 個；兩個 split 的 dialogue ID 沒有交集。
- 商品的 `title`、`description` 都是非空字串。抽樣內容是英文美容、個人照護及相關用品；這不構成「僅支援某單一品類／語言」的額外正式要求。

### 3.2 JSONL 格式與推論介面

| 檔案／介面 | 結構 | 用途與限制 |
|---|---|---|
| `dialogues.jsonl` 每行 | `dialogue_id: str`、`conversation: list`、`target: str` | 每個 conversation 元素包含 `role`、`content`；target 是唯一 gold PID |
| `products.jsonl` 每行 | `parent_asin: str`、`title: str`、`description: str` | `parent_asin` 是商品 ID；不用把它轉為數值 |
| 傳入 `retrieve_and_rerank` 的 `queries` | `list[dict]`，每筆有 `qid`、`query` | `query` 由所有對話訊息按順序組成 `role: content`，以換行分隔 |
| 傳入的 `passages` | `dict[str, str]`，即 `{PID: 商品文字}` | 官方入口預設將 title 與 description 以換行組合 |
| 標籤隔離 | retrieval loader 不傳 `gold_pid`；evaluation loader 才加入 | public 檔案含 target 不代表可將它用作推論答案或測試訓練資料 |

來源：[D1–D4；C4 L6–95；C1 L43–52；S1 p.14]。

### 3.3 排名輸出

`retrieve_and_rerank(queries, passages, *, top_k=10)` 回傳一個 list，每個元素為 `{"qid": ..., "ranked_pids": [...]}`；外層 JSON 由官方 `retrieval.py` 寫入。[C1 L16–31；C2 L12–39]

以下僅示意結構，`p01` 等是佔位符，不是可提交的商品 ID：

```json
{
  "top_k": 10,
  "results": [
    {
      "qid": "example-dialogue-id",
      "ranked_pids": ["p01", "p02", "p03", "p04", "p05", "p06", "p07", "p08", "p09", "p10"]
    }
  ]
}
```

官方 validator 明確要求：[C3 L96–142]

- `top_k` 與評估值一致，正式作業為 10。
- 每個輸入 qid 恰好對應一筆結果；不能缺漏、多出未知 qid 或重複 qid。
- 每筆恰有十個 PID，均為非空字串、存在於本次商品池，且同一排名內不能重複。
- PID 順序就是排名；介面不要求提供分數、商品全文或生成回覆。

評估輸出另包含 `metrics`、逐題 `records` 與 `judge` 資訊，例如 `graded_recall@10`、`graded_mrr@10`、解析失敗數及 judge 資源統計。這些是程式產物，投影片未將評估 JSON 列為必繳的獨立檔案。[C3 L320–336、L409–428；S1 p.15]

## 4. Confirmed Requirements

### 4.1 功能、模型與介面

| ID | 明確要求／已確認邊界 | 來源 |
|---|---|---|
| F01 | 依對話檢索相關商品規格，從提供的商品池產生商品排名 | S1 pp.5–6、10 |
| F02 | 使用官方 `retrieval.py` 載入資料並儲存結果，配合其命令列參數 | S1 p.10；C1 L34–57 |
| F03 | 只能修改 `retrieval_utils.py` 的 `retrieve_and_rerank()`；本地範例進一步限制在 `STUDENT CODE START/END` 區塊內 | S1 p.10；C2 L26–37 |
| F04 | 不修改其他既有流程來改變測試行為；不得修改官方 `evaluate.py` | S1 pp.10–11、20 |
| F05 | 每段對話輸出十個不重複且有效的 PID，完整覆蓋 qid | S1 p.11；C3 L96–142 |
| F06 | 評分模型為 `Qwen/Qwen2.5-1.5B-Instruct`；接受 gold 與被判為合理的其他商品 | S1 p.11；C3 L20–22、L291–317 |
| F07 | 報告 Q2 必須使用 Qwen2.5-1.5B-Instruct，依檢索出的商品規格生成回覆 | S1 p.24 |
| F08 | 報告 Q2 至少呈現 5 筆來自不同資料樣本的生成結果，並提供質性分析 | S1 p.24 |
| F09 | 報告 Q2 至少實驗 3 種不同 generation prompts，且比較表現；相關檔案必須提交 | S1 p.24 |

**未指定的技術不能列為強制要求：**文件沒有指定 retriever 的模型名稱、embedding 維度、向量資料庫、相似度、loss 類型或一定要使用 FAISS／Sentence Transformers；沒有規定一定要 query rewriting 或獨立 reranker。Qwen 的明確指定適用於 judge 與報告 Q2，不能據此認定 retriever 也必須使用 Qwen。[S1 pp.6、11、13、23–24；C2 L29]

### 4.2 報告內容

| ID | 明確要求 | 來源 |
|---|---|---|
| R01 | Q1「Retrieval & Reranking Strategy」占 4%，詳細說明策略；評分看描述品質 | S1 p.23 |
| R02 | 說明訓練資料如何建構，包含例如 anchor、正樣本、負樣本策略與使用的外部資料種類 | S1 p.23 |
| R03 | 說明訓練 loss 的規格及理由 | S1 p.23 |
| R04 | 列出實驗採用的 hyperparameters | S1 p.23 |
| R05 | 說明試過哪些策略、進行比較，並解釋為何提交的策略最好；文件未給策略數量下限 | S1 p.23；S2 803.2–839.2s |
| R06 | Q2「Generation Analysis」占 2%，符合 F07–F09，評分看分析深度與豐富度 | S1 p.24 |
| R07 | 訓練、撰寫報告及生成分析所使用的程式／腳本與相關檔案都要包含在提交內容 | S1 pp.15、23–24 |

Q1 以訓練為前提描述要回答的內容，但文件沒有直接說明「完全不做訓練的策略是否可作為最終提交」以及該如何處理 loss／training 欄位。不能自行將它解讀成「一定可免訓練」，也不能把指定某種微調方法當成已確認要求；見 Q1。

## 5. Constraints

### 5.1 執行環境與效能限制

| ID | 限制 | 來源／備註 |
|---|---|---|
| C01 | Python **3.12.9** | S1 p.13；S3 L3 |
| C02 | 測試 OS：**Ubuntu 22.04** | S1 p.18；本地 Windows 並非官方測試平台 |
| C03 | **32 GB RAM、RTX 3070 8 GB VRAM、20 GB 可用磁碟空間** | S1 p.18 |
| C04 | **Public set 的 retrieval process 必須在 1 小時內完成** | S1 p.18；S2 651.2–678.2s 說明逾時會停止 |
| C05 | `download.sh` 須於 **30 分鐘內**完成，系上環境最高 **10 MB/s** 頻寬 | S1 p.16 |
| C06 | `download.sh` 只能執行下載用途，不得 `pip install` 或變更測試環境；評分前執行 | S1 p.16 |
| C07 | 可用其他套件，但必須列入提交的 `requirements.txt` | S1 p.13；如何與固定測試環境協調見 Q2 |
| C08 | 助教評估命令使用 **`--batch-size 8`** | S1 p.20；程式預設值 1 不等於正式設定 |

一小時限制明確指 public 檢索流程，不能擴張成「訓練、下載、LLM judge、報告生成合計一小時」。Private 是否同樣限時，以及建索引／模型載入的計時範圍，文件未明確界定，見 Q4。

### 5.2 套件版本對照

投影片明稱這些是建議套件，允許列出其他依賴；但測試環境又聲明採相同環境。因此下表完整保留差異，不自行替換或裁定版本。[S1 pp.13、18；S4 L1–9]

| 套件 | 投影片 p.13 | 本地 requirements.txt |
|---|---|---|
| transformers | `==4.56.1` | `==4.56.1` |
| torch | `==2.8.0` | `==2.8.0` |
| datasets | `==4.0.0` | **`==5.0.1`** |
| tqdm | `==4.67.1` | `==4.67.1` |
| faiss-gpu-cu12 | `==1.12.0` | `==1.12.0` |
| sentence-transformers | `==5.1.0` | `==5.1.0` |
| python-dotenv | `==1.1.1` | 未列 |
| accelerate | `==1.10.1` | `>=1.10.1` |
| huggingface_hub | `==0.24.0` | `>=0.24.0` |
| gdown | 未指定版本 | 未列 |
| numpy | 未列 | `==1.26.4` |

這份套件表不表示每個套件都必須被解法使用。現階段沒有安裝或修改任何套件。

### 5.3 資料、學術規範與檔案保留

| ID | 明確規範 | 來源 |
|---|---|---|
| C09 | 不得從 `Abner0803/Amazon-Beauty-S2` 使用額外外部訓練資料 | S1 p.14 |
| C10 | 不得在 test data 上 fine-tune 模型 | S1 p.14 |
| C11 | 不得試圖尋找 private test data | S1 p.14 |
| C12 | 不得直接套用他人已公開或未公開的程式，包含同學及網路上的程式；不得互給／取得他人的訓練模型、預測、報告答案或圖表 | S1 p.14 |
| C13 | 不得在截止日前公開自己的程式 | S1 p.14 |
| C14 | 作弊或抄襲可能導致零分、負分及校方處分 | S1 p.14 |
| C15 | 截止後不得再修改下載連結指向的提交檔案／模型；投影片稱違反視為作弊 | S1 p.16；S2 503.1–519.1s |
| C16 | 下載 URL 在截止後至少維持有效 **3 週** | S1 p.16 |
| C17 | 若使用外部訓練資料，須透過下載腳本提供，並在報告交代資料種類 | S1 pp.16、23 |

不能把 C09 誤讀成「禁止一切外部資料」：投影片 p.16、p.23 明確提及外部訓練資料，但沒有完整列出每一種外部／合成資料的許可範圍。若日後考慮特殊來源，須在採用前查核課程規則；本階段沒有取得額外訓練資料。

文字稿最後提到使用 coding agents 的同學應確認訓練程式真的能運作且提交完整；沒有在這段宣告全面禁止這類工具。這也不能被擴張為可以違反前述資料與抄襲規範。[S2 976.6–999.6s]

### 5.4 期限、遲交與扣分

| 項目 | 明確規範 | 來源 |
|---|---|---|
| Deadline | **2026/10/25 23:59:59** | S1 pp.1、21 |
| 提交平台 | NTU COOL | S1 pp.15、21 |
| 遲交不超過 2 天 | `0 < 遲交時間 ≤ 2 天`：原始分數乘 **0.70** | S1 p.21 |
| 遲交超過 2 天 | 原始分數乘 **0.0** | S1 p.21 |
| 多次提交 | 僅採計 NTU COOL **最後一次提交**；截止後更新會適用遲交處理 | S1 p.21 |
| README 缺漏或空白 | 扣 **2%**，保留投影片原單位 | S1 p.17 |
| 檔案結構錯誤 | 可能扣 **2%** | S1 p.19 |
| 結果無法重現 | 可能扣分，文件未給固定數值 | S1 p.17 |

投影片未另寫時區；本文件在使用者與課程情境下暫以 Asia/Taipei 解讀，並未宣稱投影片明載時區。文字稿中秒數辨識錯誤不取代投影片的 `23:59:59`。

## 6. Expected Deliverables

### 6.1 必交封裝

以**小寫學號**作為 ZIP 檔及其中最外層資料夾名稱，提交到 NTU COOL。解壓縮後必須先出現學號資料夾，不能直接散開所有檔案。`r13000000` 是投影片範例，不是使用者的學號。[S1 p.15；S2 442.8–475.1s]

```text
<lowercase_student_id>.zip
└── <lowercase_student_id>/
    ├── download.sh
    ├── run.sh
    ├── retrieval_utils.py
    ├── requirements.txt
    ├── report.pdf
    ├── README.md
    └── code/script   # 投影片原標示；包含訓練、報告等所有相關程式／腳本
```

`code/script` 的原文意義是相關程式／腳本須一併交付；投影片沒有進一步界定該標示是否要求字面上的雙層目錄。最終封裝應保留清楚的程式目錄與可重現路徑，不把目錄慣例當作已證實的新規定。

| 成果 | 必須包含／做到的內容 | 來源 |
|---|---|---|
| `download.sh` | 下載模型及使用的外部訓練資料；僅下載、30 分鐘內完成、連結有效至少 3 週 | S1 p.16 |
| `run.sh` | 重現模型訓練流程，與 README 說明一致 | S1 p.17 |
| `retrieval_utils.py` | 在允許修改區塊中提供正式檢索策略，符合官方函式介面 | S1 pp.10、15；C2 |
| `requirements.txt` | 列明使用的套件，包含額外依賴 | S1 pp.13、15 |
| `report.pdf` | Q1 策略與訓練分析；Q2 指定模型的生成與提示詞比較 | S1 pp.23–24 |
| `README.md` | 非空，逐步說明環境建立、訓練命令及所用程式；能配合 `run.sh` 重現 | S1 p.17 |
| 相關 code／script／檔案 | 訓練、報告產製及 Q2 分析所使用的所有相關內容 | S1 pp.15、24 |

助教使用官方 `evaluate.py`，明確說明不必上傳該檔。[S1 p.20] 排名 JSON 是必須能由官方入口產生的執行產物；投影片未另要求將它作為獨立提交件。`PROJECT.md` 是本次 Phase 1 的分析文件，不是投影片列出的必交報告。

### 6.2 官方執行命令（僅記錄，尚未執行）

```bash
python retrieval.py --dialogues "dialogues_path" --products "products_path" --output "output_path"
python evaluate.py --retrieval-results "path_to_retrieval_result" --dialogues "dialogues_path" --products "products_path" --output "output_path" --batch-size 8
```

來源：[S1 pp.10、20]。主流程是載入資料 → retrieve／可選 rerank → 儲存排名 → 官方評估。`download.sh` 在評估前執行；`run.sh` 用於訓練重現，不應自行改解為官方推論入口。

### 6.3 現有專案狀態

- 投影片 p.7 列出的 starter 檔案與 train／public 資料目前都已存在。
- `retrieval.py` 與 `evaluate.py` 已有完整入口、讀寫及驗證流程。
- `retrieval_utils.py` 的 student block 仍為空實作，會拋出 `NotImplementedError`。[C2 L35–36]
- 現有 README 是 starter 操作說明，不是使用者最終訓練與環境重現文件。[S3]
- 工作區尚無 `download.sh`、`run.sh`、`report.pdf`、訓練／生成實驗程式、模型產物或效能結果；不能視為作業已完成。
- 本階段僅新增此需求分析文件，未修改 starter 程式、README、套件清單或資料。

## 7. Requirement Classification

### 7.1 Confirmed Requirements

第 4–6 節中附有來源的明確功能、介面、模型、報告、資源、期限與提交規定，構成已確認要求。另需區分兩種依據：投影片提供正式作業規範，現有程式提供下載版本實際執行的資料契約；兩者發生差異時，不默默以任一方取代另一方。

### 7.2 Inferred Requirements

| ID | 推論／工程判斷 | 推論依據與尚未證實之處 |
|---|---|---|
| I01 | 系統需將真正需求與歷史購買、閒聊或被否定的選項區分 | 對話樣本與 judge 規則支持此能力的重要性，但未要求特定意圖／否定辨識模組 |
| I02 | 應能對新增商品做一般化，不能只記憶 train PID | 商品池擴張已確認；應採何種一般化方法尚未指定 |
| I03 | 訓練可能是課程預期的一部分 | Q1 與 run.sh 都以訓練為前提，但未清楚說明無訓練方法是否可接受，見 Q1 |
| I04 | 資料載入、商品向量計算、建索引與 rerank 的總成本可能影響一小時限制 | 正式計時邊界未定，不能先假設預計算免費或被允許排除，見 Q4 |
| I05 | 以同一組 5 個樣本比較 3 種 prompts，可形成較公平的 15 格比較 | 這是候選實驗設計；投影片沒有明定每個 prompt 都要 5 筆，見 Q5 |
| I06 | 在訓練集內另留驗證資料，有助於選策略並避免把 test 當訓練資料 | 這是工程建議；作業沒有指定 validation 切分比例 |

### 7.3 Open Questions

最重要的五題按影響程度集中列於第 9 節。已確認的截止日期、模型名稱、K 值、公開資料數量與禁止 test fine-tuning，不列為重複提問。

### 7.4 Optional Improvements / Potential Advanced Features

以下僅為可研究的候選能力，本階段未選定或實作，也不代表助教強制要求：

| 候選改善 | 可能改善的問題 | 可觀察的效果 |
|---|---|---|
| 對話重寫或需求摘要 | 多輪資訊分散、歷史商品干擾 | 對照原始對話的 Recall@10、MRR@10 |
| 稀疏／稠密檢索融合 | 字面約束與語意相似性各有不足 | 命中率與難例類型改變 |
| 候選集重排序 | 正確商品在候選中但排序較後 | MRR@10 與額外推論時間 |
| 合規的負樣本策略與模型訓練 | 相似品類／不同用途商品容易混淆 | 留出驗證資料的表現；前提是滿足資料規範 |
| 商品文字整理、分段或欄位權重 | 商品描述冗長或關鍵限制不突出 | 長商品樣本品質與運算成本 |
| 批次處理、快取、索引與記憶體最佳化 | 商品池擴張與資源限制 | 全流程時間、RAM、VRAM、磁碟占用；快取使用範圍需符合 Q3／Q4 的釐清 |
| 失敗案例分類與消融分析 | 不知道哪個策略有效或造成退步 | 同一評估條件下的數值差異與錯誤分布 |
| 擴大 Q2 的樣本與提示詞比較 | 最低展示數可能不足以解釋成功／失敗模式 | 質性分析的覆蓋度與深度 |

UI、即時聊天與更多產品功能並非本作業完成條件，現有文件也未提供這些功能的評分方式。

## 8. Success Criteria

### 8.1 Minimum Viable System

技術上的最小可行檢索系統必須：

1. 能由未修改的官方 `retrieval.py` 執行，只接收既有對話與商品資料介面。
2. 對所有輸入對話各回傳十個合法、互異、有序的 PID，通過官方結果格式驗證。
3. 使用完整對話資訊進行商品檢索，能處理測試時傳入的商品池及未見過的商品。
4. 在官方硬體／環境限制內完成 public 檢索，且總流程可重現。
5. 能由原版 `evaluate.py` 輸出正式 Recall@10 與 MRR@10。

**只會產生合法 JSON 不代表已達成品質目標；只完成檢索也不代表整份作業完成。**是否可以不訓練而提交，須先解決 Q1。這裡沒有將任意 baseline 數值宣告為官方唯一「及格線」。

### 8.2 Required Deliverables 的完成判定

- 第 6 節所有必交檔案與資料夾結構齊備，依小寫學號封裝。
- 模型／外部資料可下載且未逾時，截止後保持固定並維持連結有效。
- README 與 run.sh 一致，可重現實際採用的訓練流程；若無訓練，須依 Q1 的正式答覆處理。
- Q1 回答資料、loss、hyperparameters、策略比較及選擇理由，且附相關程式。
- Q2 使用指定模型及檢索結果，至少覆蓋 5 個不同樣本與 3 種 prompts，提供有內容的質性比較並附相關檔案。
- 準時以正確 ZIP 提交到 NTU COOL，符合資料與學術規範。

目前上述條件尚未驗收；本階段沒有訓練、生成或執行正式評分。

### 8.3 Measurable Evaluation Criteria：正式指標

對第 i 段對話的第 r 名商品，令 `g(i,r)` 為：

- PID 等於該對話 gold PID：`1`，不呼叫 judge。
- 非 gold，但固定 judge 判為 relevant：`1`（本地 `ALTERNATIVE_GAIN = 1`）。
- Judge 判為不相關，或其輸出解析失敗：`0`。

來源：[S1 p.11；C3 L20–22、L67–93、L263–265、L291–317]。

```text
Recall@10 = 所有對話的 max(g(i,1), ..., g(i,10)) 之平均
MRR@10    = 所有對話的 max(g(i,r) / r，r = 1..10) 之平均
```

在目前二元 gain 下，Recall@10 是「前十名至少出現一個可接受商品的對話比例」；MRR@10 是「第一個可接受商品排名倒數的平均」。例如第一個可接受商品在第 4 名，該題的 Recall 為 1、MRR 為 0.25；十名都不合格則皆為 0。多放幾個合格商品不會讓單題 Recall 超過 1。

這不是把 top-10 相關商品數除以 10 的 Precision@10，也不是對整個商品池全部相關商品計算完整召回率。不得以 gold-only 指標冒充正式接受替代商品的評分。

Judge 會看到對話、gold 商品文字與候選商品文字，判斷核心功能和明確限制。相關但用途不同、只是搭配品或違反明確規格的商品可能被拒絕；軟性偏好不必然視為硬限制。[C3 L28–53、L186–194]

本地 judge 採 bf16、greedy decoding、固定 seed；官方命令指定 batch size 8，未覆寫時 `max_new_tokens=12`、K=10。這是 judge 設定，不是 Q2 回覆生成的 token 上限或 prompt 規範。[C3 L145–184、L223–230、L349–359；S1 p.20]

### 8.4 正式效能級距與配分

以下逐格抄錄投影片 p.19，括號是原投影片配分單位：

| Baseline | Public Recall@10 | Public MRR@10 | Private Recall@10 | Private MRR@10 |
|---|---:|---:|---:|---:|
| Boss | 0.950（1%） | 0.800（1%） | 0.930（1%） | 0.780（1%） |
| Strong | 0.885（1%） | 0.680（1%） | 0.880（1%） | 0.675（1%） |
| Simple | 0.860（2%） | 0.650（2%） | 0.845（1%） | 0.635（1%） |

Model Performance 共 14%，Report 共 6%（Q1 4% + Q2 2%）。不要把這些比例擅自換成「本作業滿分 100 下的百分比」。各格加總與 14% 相符；文件沒有另述邊界相等、四捨五入與各級距計分細則，故本文件只記錄原始門檻，不保證某個自算總分。[S1 pp.19、23–24]

若後續需要設定最低開發品質目標，可以將 public simple 的 `0.860 / 0.650` 作為第一個候選目標；這是對照官方級距的開發選擇，不代表使用者已選定目標或全作業完成門檻。Private 無法在目前本地資料驗證，不能預先宣稱已達其 baseline。

### 8.5 文件未給明確數值的項目與候選方法

| 項目 | 官方狀態 | 合理候選評估方法，均非新增強制規格 |
|---|---|---|
| Q1 報告品質 | 依說明品質評分；細部量化 rubric **待確認／未提供** | 檢查要求的四類內容是否齊備；同條件比較各策略，記錄參數、結果、耗時與取捨 |
| Q2 生成品質 | 有樣本／prompt 下限，但沒有數值品質及格線；**待確認／未提供** | 逐例評估需求滿足、商品規格忠實度、硬限制遵守、無依據敘述與可讀性；可自訂 0–2 分量表，清楚標示為自行分析 |
| Q2 樣本與 prompt 組合數 | 5 個不同樣本、3 種 prompts 已確認；是否要求 15 個輸出 **待確認** | 固定 5 個樣本逐一比較 3 種 prompts，保留原始輸出；見 Q5 |
| 結果重現容許誤差 | 要能重現，但數值容許誤差 **待確認／未提供** | 保存實際版本、seed、命令、模型與資料識別，乾淨環境重跑後比較排名與指標 |
| Private 執行時限 | public 一小時已明定，private 數字 **待確認** | 記錄隨商品數、對話數增加的時間與資源趨勢，不能用外推結果替代官方驗收 |
| 個別對話延遲 | 未要求互動式服務，沒有每題 latency 門檻 | 以批次總時間及 RAM／VRAM／磁碟峰值檢驗是否能在官方環境完成 |

上述候選量表、驗證切分、消融或額外效能統計都只是輔助方法，不替代官方評估，也尚未執行。

## 9. Open Questions / Clarification（最多五題，依影響排序）

以下問題已先比對全部本地文件；不要求使用者重答文件已明講的資訊。適合在進入後續階段前查核 NTU COOL 討論區，必要時詢問助教。本階段只整理問題，沒有代為發信或發布提問。

| 優先序 | 待確認問題 | 為何尚無答案／已知部分 | 對後續工作的影響 |
|---|---|---|---|
| **Q1** | **是否必須自行訓練／微調模型？若使用完全不訓練的檢索策略，Q1 的訓練資料、loss 與 `run.sh` 應如何交代才合格？** | p.17、p.23 以訓練為前提；p.10 的介面則沒有明訂訓練必要性。沒有無訓練方案的例外說明。 | 決定是否需要訓練管線、算力、模型產物與報告內容，是最大的範圍差異。 |
| **Q2** | **最終測試環境以投影片還是下載的 requirements 為準，新增套件由誰在何時安裝？** | p.13 允許列出額外套件，p.16 禁止在 download.sh 安裝／變更環境，p.18 說採相同環境；`datasets` 等版本又不同。 | 決定可用依賴、模型相容性與重現方式；不能自行假設評分時會安裝所有新增套件。 |
| **Q3** | **在限定 student block 內，是否允許 import 自行撰寫並隨附的輔助模組，或讀取預先建立的商品索引／快取？** | p.10 與 C2 明訂修改區域；p.15 要交程式，但沒有明說推論時額外模組／預計算資產的允許範圍。 | 影響程式拆分、模型與索引封裝，以及新商品池如何建立所需資產；不涉及放寬禁止使用 test 標籤。 |
| **Q4** | **Public 的一小時計時是否涵蓋模型載入、所有商品編碼與建索引；private 是否同樣一小時、以及推論期間能否連網下載？** | p.18 只寫 public retrieval process 一小時；p.16 明訂下載在評估前，但沒有細列冷啟動、快取及 private 時限。 | 影響模型大小、商品預處理、候選數與整體時間預算。可先採全流程冷啟動計時作為保守候選驗證方式，但不是官方已確認定義。 |
| **Q5** | **Q2 的「至少 5 個不同樣本」是否要求每種 prompt 都跑這 5 個樣本，亦即至少 15 個生成輸出？** | p.24 分別規定 5 個樣本與 3 種 prompts，沒有明確規定交叉組合數；文字稿也未補充。 | 影響生成實驗矩陣與報告工作量。5×3 是可採用的候選比較方式，不能寫成原文強制規格。 |

沒有把學號列為分析阻塞問題：學號只影響最終封裝命名，Phase 1 不需提前取得。尚未選定的個人分數目標與可用訓練設備，也不改變上述已確認作業要求。

## 10. 文件差異與誤讀防護

| 差異或容易誤讀處 | 分析處理 |
|---|---|
| 文字稿前段說不做 generation；p.24 要做 Generation Analysis | 主系統檢索與報告生成分開，兩者並不互相取消；文字稿後段也補充 Q2。 |
| 文字稿把 RAG、reranker、Qwen、Recall／MRR、NTU COOL、run.sh 等辨識成錯詞 | 使用投影片及程式中的可核對拼字，不將 ASR 錯字當成新技術規格。 |
| 文字稿約 210–244s 混用 public／private 名稱 | 依 p.8–9 與實際資料核對：只提供 train／public；private 不公開。 |
| 文字稿口頭說 Recall 有多少符合的商品 | 以 p.11 公式與 C3 的 `max(gains)` 解讀，不能換成 Precision。 |
| README 泛稱在 retrieval_utils.py 實作 | p.10 與 C2 的 student block 限制更明確，不能擴張成整個檔案任意改。 |
| p.18 引用套件頁面為 p.9 | 實際 PDF p.9 是資料量，環境表在 p.13；視為疑似交叉引用筆誤，保留說明。 |
| p.13 與 requirements 的套件版本不同 | 第 5.2 節並列差異，列入 Q2，未自行改檔。 |
| evaluate.py 預設 batch size 1 | 正式命令 p.20 明確傳入 8，因此不需要再詢問正式 batch size。 |
| 文字稿 deadline 秒數、README 扣分及截止後改模型的措辭不精確 | 依投影片記錄精確秒數與明文規則；不採用 ASR 的錯誤否定句或弱化說法。 |
| 圖示有 encoder、index、reranker、LLM | 圖示用於解釋 RAG；不代表每個方塊都是必須獨立實作的元件。 |
| p.5 顧客詢問腰痛並推薦椅子 | 是 Scenario。實際作業處理提供的商品資料，不額外增加醫療、椅子或主動追問系統。 |
| `download.sh`、Q1 都提及外部資料 | 不能自行推導一律禁止所有外部資料，也不能假設每種資料都無限制可用；明確禁止來源與 test 規則仍須遵守。 |

## 11. Source References

### 11.1 已閱讀／檢查的本地來源

| ID | 檔案 | 使用範圍 |
|---|---|---|
| S1 | [ADL2026-HW2.pptx.pdf](C:/Users/yuting/Desktop/HW2_ADL/ADL2026-HW2.pptx.pdf) | 全部 25 頁；文字及渲染頁面交叉核對 |
| S2 | [ADL_HW2.txt](C:/Users/yuting/Desktop/HW2_ADL/ADL_HW2.txt) | 完整 360 行，時間約 4.8–1003.6s；有明顯語音辨識錯字 |
| S3 | [README.md](C:/Users/yuting/Desktop/HW2_ADL/README.md) | 全文 25 行；starter 執行方式與 Python 版本 |
| S4 | [requirements.txt](C:/Users/yuting/Desktop/HW2_ADL/requirements.txt) | 全部 9 項套件宣告 |
| C1 | [retrieval.py](C:/Users/yuting/Desktop/HW2_ADL/retrieval.py) | 全文 62 行；載入、函式呼叫、JSON 輸出及 CLI |
| C2 | [retrieval_utils.py](C:/Users/yuting/Desktop/HW2_ADL/retrieval_utils.py) | 全文 39 行；允許修改區、介面、目前空實作 |
| C3 | [evaluate.py](C:/Users/yuting/Desktop/HW2_ADL/evaluate.py) | 全文 438 行；模型、judge、validator、指標與 CLI |
| C4 | [eval_utils.py](C:/Users/yuting/Desktop/HW2_ADL/eval_utils.py) | 全文 95 行；對話／商品轉換與 gold 隔離 |
| D1 | [data/train/dialogues.jsonl](C:/Users/yuting/Desktop/HW2_ADL/data/train/dialogues.jsonl) | 全檔結構檢查與內容抽樣，共 4,830 筆 |
| D2 | [data/public/dialogues.jsonl](C:/Users/yuting/Desktop/HW2_ADL/data/public/dialogues.jsonl) | 全檔結構檢查與內容抽樣，共 1,000 筆 |
| D3 | [data/train/products.jsonl](C:/Users/yuting/Desktop/HW2_ADL/data/train/products.jsonl) | 全檔結構檢查、商品池比對與內容抽樣，共 50,000 筆 |
| D4 | [data/public/products.jsonl](C:/Users/yuting/Desktop/HW2_ADL/data/public/products.jsonl) | 全檔結構檢查、商品池比對與內容抽樣，共 75,000 筆 |

### 11.2 投影片頁碼索引

| 頁碼 | 內容 |
|---|---|
| 1 | 作業名稱、deadline |
| 2 | Update Logs；本地版本沒有實質更新項目 |
| 3 | Homework 2 files、HW2 討論區連結 |
| 4 | Task & Guides 章節頁 |
| 5 | Salesbot 示範情境 |
| 6 | RAG 圖與商品檢索核心任務 |
| 7 | Starter 檔案清單 |
| 8 | 資料構成、單一 gold、其他可接受商品、公開 split |
| 9 | Train／public／private 資料量 |
| 10 | 可修改函式、檢索入口與命令 |
| 11 | 固定 judge、Recall／MRR 公式、禁止修改評估 |
| 12 | Rules & Submission 章節頁 |
| 13 | Python 與建議套件 |
| 14 | 禁止事項與學術規範 |
| 15 | ZIP 命名、目錄結構與必交檔案 |
| 16 | download.sh 限制、下載時限與連結保留 |
| 17 | run.sh、README、可重現性及扣分 |
| 18 | 官方執行環境與 public 檢索時限 |
| 19 | Baseline 表、14% 模型／6% 報告配分與格式扣分 |
| 20 | 官方評估命令與 batch size 8 |
| 21 | Deadline、遲交、最後一次提交規則 |
| 22 | Report 章節頁 |
| 23 | Q1：Retrieval & Reranking Strategy |
| 24 | Q2：Generation Analysis |
| 25 | 問題詢問管道與 TA hours |

### 11.3 投影片中的參考與聯絡管道

下列連結是從 PDF 註解／文字取得，**本次沒有驗證其線上內容、最新公告或可存取狀態**：

- [Homework 2 files](https://drive.google.com/drive/folders/1tBS9Qk8s86WzVabpVeHtbIZr9bfKDJ8I?usp=sharing)：p.3 的官方檔案資料夾。
- [NTU COOL HW2 討論區](https://cool.ntu.edu.tw/courses/64983/discussion_topics/530817)：p.3、p.25；投影片請同學寄信前先查討論區。
- [RAG 圖片來源論文](https://arxiv.org/abs/2406.06475)：p.6 的圖源，未被列為必須實作整篇方法的規格。
- [Abner0803/Amazon-Beauty-S2](https://huggingface.co/datasets/Abner0803/Amazon-Beauty-S2)：p.14 指名禁止用於額外訓練資料的來源；本次未下載。
- TA email：`adl-ta@csie.ntu.edu.tw`；郵件主旨前綴 `[ADL2026 HW2]`。[S1 p.25]
- 投影片記錄的 TA hours：週四 16:00–17:00，CSIE Building R524（陳剛頡）；週三 14:30–15:30，[Google Meet](https://meet.google.com/tyx-nzwr-jhk)（彭紀綸）。這是文件內容，未查核即時異動。[S1 p.25]

## 12. Phase 1 Completion Record

- 已盤點全部提供來源、核對投影片與文字稿，並檢查現有程式及資料格式。
- 已區分 confirmed、inferred、open questions、optional improvements，明確保留文件衝突。
- 已記錄輸入輸出、限制、交付成果、官方指標／baseline，以及沒有明確標準時的候選評估方法。
- 已整理五個優先待確認問題，沒有把示例情境提升為硬性需求。
- 已建立 `PROJECT.md`；未實作檢索、訓練模型、下載模型、執行正式評分、製作報告或提交作業。

**依使用者指示，到此停止 Phase 1。後續階段需由使用者另行指示。**
