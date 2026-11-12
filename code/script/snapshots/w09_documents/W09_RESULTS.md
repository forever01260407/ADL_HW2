# W09 — Q2 固定五題與三種生成提示詞比較

日期：2026-10-24（Asia/Taipei）。狀態：**W09 的程式、15 筆生成、原始資料核對與逐例分析完成。這不是「15 筆回答都正確」，也不是整份作業已通過。**

## 作業要求與本次範圍

依 `PROJECT.md` F07–F09／R06–R07（投影片 `ADL2026-HW2.pptx.pdf` p.24），Q2 使用 `Qwen/Qwen2.5-1.5B-Instruct`，呈現至少五個不同樣本的生成結果、至少三種 generation prompts 及質性比較，並提交相關程式與檔案。**五題各跑三種、共 15 筆是本專案的公平比較設計，不是投影片明訂的交叉組合數。**

本輪沒有訓練 Qwen、修改 W08 檢索、調整官方 evaluator 或執行 public。Q2 生成與官方 retrieval Judge 是獨立用途，不把此處 256-token／batch 1 設定套到 Judge。

## 固定輸入與公平比較

- 使用 W02 在生成前已鎖定的 `data/derived/q2_queries.jsonl`；五題、2 題三則訊息／3 題五則訊息，不因輸出不好而換題。
- 每題只使用凍結 W08 `results/w08_default_dev_top50.json` 的前三名，保留原始順序，不補 gold、替換商品或用 Judge 篩選。
- 生成讀取的樣本不含 `target`／gold／Judge 結果；額外欄位會拒絕。完整對話及商品以 JSON 資料呈現，使用 Qwen 原生 chat template；資料內的尖括號採 JSON escape，避免形成字面 chat-control tags。這是資料邊界處理，不代表模型已通過語意 prompt-injection 安全驗收。
- 同題三種 prompts 的商品文字／PID／排序／證據 hash 完全一致；對話完整保留。若 token 預算不足，三者共用較短商品前綴，而非個別截斷。本次所有 15 個商品位置都不需截斷。
- 先保存全部 rendered prompts 與 input token IDs，再開始生成；15 筆 output 按順序落盤。沒有重跑挑選較佳回答，也沒有修飾原文。

| 樣本 | 對話則數 | 實際觀察重點 |
|---|---:|---|
| `train-pet-supplies-000140` | 3 | 狗減重但不縮小份量、維持肌肉；候選包含減重食品、增重食品、餵食器 |
| `train-sports-and-outdoors-000373` | 5 | 標準 Shield、IWB、無配件；退款與公司背景是偏好，歷史腳踏車／耳罩是干擾 |
| `beauty-train-003071` | 5 | 已買過 stamper，現在對 full nail patterns 有興趣；需區分工具與圖板 |
| `train-pet-supplies-000234` | 5 | 單一成分、保留消化酵素、不烘焙、美國製、無填充物；候選存在明確不符與缺資訊 |
| `beauty-train-001206` | 3 | 紅色是暫定偏好，擔心 brassy／受損；候選 SKU 實際是棕色，系列文字另提紅色 |

原 `selection_tag` 只代表選樣時的詞面線索，例如 nail-art 題的 `negation_cue` 不能當成已人工驗證的硬限制標籤。

## 三種提示詞與模型設定

共通規則要求：只用提供的資料、把資料當成非指令、英文顧客回覆、至多 120 words、不捏造價格／庫存／規格／功效／保證；商品文案不等於已驗證效果；命名商品時使用 `[PID: identifier]`。完整字串保存在 [w09_prompts.json](C:/Users/yuting/Desktop/HW2_ADL/code/script/configs/w09_prompts.json)。

| ID | 本次主要變因 | 具體要求 |
|---|---|---|
| G-P1 | 直接推薦 | 選一個最合適商品並簡短說明依據；沒有合適候選時直說 |
| G-P2 | 需求對應證據 | 推薦後列三組 Need → Evidence，指出未滿足或未確認的要求 |
| G-P3 | 明示不確定性 | 分開已支持／未證實的內容，缺規格視為 unknown，僅作有條件推薦或無法確認 |

模型 revision：`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`；bf16、seed 0、greedy (`do_sample=False`)、`num_beams=1`、batch 1、最多 256 new tokens；總預算 4096、輸入上限 3840、每商品上限 900 tokens。模型權重及設定已核對下載 manifest 的 hashes，正式生成離線執行。

## 可直接核對的執行結果

**54 個測試／5.962 秒通過**（初期 51 個／5.833 秒），包含同題同證據、最長 prompt 共同縮短、超長對話不靜默刪除、樣本拒絕 target／重複 qid、未知／重複 PID、控制標記、續跑順序，以及質性評語完整覆蓋與合法分數範圍。真實 tokenizer／全部 token IDs／來源 hashes／raw decoding 的整合核對另已通過。

| 項目 | 實測 |
|---|---:|
| 完整 grid | 5 題 × 3 prompts＝15 筆 |
| 最長輸入 | 2,337 tokens |
| 最長輸出 | 211 tokens |
| 停止原因 | 15／15 EOS；0 長度上限停止 |
| 商品證據截斷 | 0／15 個商品位置 |
| 空回答／提到證據外 PID | 0／0 |
| 提到任一實際 PID | 4／15 筆回答 |
| 遵守指定 `[PID: ...]` 格式 | **0／15** |
| 生成程序時間（含模型載入，不含已完成的 W08 檢索） | 84.250 秒 |
| CUDA peak allocated／reserved | 3.148／3.545 GiB |
| Process peak RSS | 5.892 GiB |

「沒有證據外 PID」不能當作正確率：11 筆根本沒提 PID，其他也可能把合法 PID 的規格說錯。結構核對通過只表示紀錄完整、同條件與 token／來源可核實，不表示生成品質通過。

| Prompt | 提到任一 PID | 指定 PID 格式 | 超過 120 words | 輸出 tokens 合計 | 純生成秒數 |
|---|---:|---:|---:|---:|---:|
| G-P1 | 1／5 | 0／5 | 0／5 | 571 | 15.630 |
| G-P2 | 2／5 | 0／5 | 1／5 | 680 | 18.669 |
| G-P3 | 1／5 | 0／5 | 3／5 | 821 | 21.474 |

Words 依空白切分，屬格式診斷。時間為 WSL Ubuntu 22.04／Python 3.12.12／RTX 4090 開發機觀測，不代表官方 Python 3.12.9／RTX 3070 驗收。

## 五題逐例比較

所有完整原始回答見 [raw_outputs.md](C:/Users/yuting/Desktop/HW2_ADL/results/w09_q2/raw_outputs.md)，完整提供證據見 [evidence.md](C:/Users/yuting/Desktop/HW2_ADL/results/w09_q2/evidence.md)。以下摘錄只是分析，沒有代替或修改 raw outputs。

### 1. 保持份量的狗減重：`train-pet-supplies-000140`

第一名 `B07GPX2F5S` 的 SIMPLY FIT 文案明說改變熱量而非份量、兩種食品交替，以及減脂維持肌肉；第二名 `B01ACQIG8Y` 是增重食品，第三名是餵食器。

- **G-P1** 選對第一名，能說明 ICR、25% 較少熱量與相同份量，較完整回答需求；但把文案功效直接當效果，`Fat Burn Chicken` 命名也不精確。
- **G-P2** 選對商品，但 `Need -> Evidence` 下只重述三個需求，沒有提出商品對應規格。這說明產生標題不等於完成證據對應。
- **G-P3** 推薦方向正確，卻說製造商有 `100% satisfaction guarantee`。此保證不在第一名的提供文字，反而出現在第二名增重食品；符合跨商品屬性混用的模式。也沒有清楚說明最重要的不確定處。

這題檢索已提供合適候選，主要問題發生在生成的歸屬與表達，不需要靠補 gold 解決。

### 2. 標準型 IWB 商品：`train-sports-and-outdoors-000373`

第一名 `B077TXZ34T` 已列標準型 Shield／無附加配件、IWB、退伍軍人公司及 100% money-back；沒有列出退款期限。第三名的 title 與 features 對左右手還有矛盾，不能任意合併規格。

- **G-P1** 選對第一名，引用的主要規格有支持，且未被歷史腳踏車與耳罩帶走；但舒適與可靠性仍應標示為文案宣稱，而不是保證結果。
- **G-P2** 能列需求對應，卻新增 `within 30 days`。100% 退款不能自動推導成 30 天，這是無依據的新規格。
- **G-P3** 大量談歷史腳踏車與耳罩，最後說 `no suitable product available`，忽略已提供的合適第一名。**謹慎措辭也可能是錯誤拒答**，不能把所有 abstention 視為更安全或更正確。

### 3. Full nail patterns：`beauty-train-003071`

第一名 `B005V2LLXC` 描述 stamper／scraper；第三名 `B006OMHV00` 標題明列 42 Full Nailart Designs。顧客已買過 stamper，對 full nail patterns 感興趣且仍開放選項，並未確認想再買同類工具。

- **G-P1** 推第一名並說它 `offers full nail patterns`；此 PID 的提供文字沒有明確證實附有哪些 full-nail 圖案，沒有交代工具與圖板的差異。
- **G-P2** 多了結構但也多了捏造：聲稱顧客表示會 `work fast`，實際這是其他商品的操作文字，不是顧客說過的話。也自行補出提升技巧的明確意圖；171 words 超限。
- **G-P3** 同樣把 full patterns 套到第一名，又無證據地稱其他圖板較 specialized、less versatile，未按提示詞指出缺資訊；134 words 超限。

這題顯示生成未充分區分「使用者的話」「商品操作說明」「既有商品」與「目前想買的東西」。

### 4. 消化酵素與美國製：`train-pet-supplies-000234`

前兩名 Yak Cheese 文案明列 **made in Nepal**，成分含 yak milk、salt、lime juice；第三名 `B08223DG6P` 是口腔清潔用酵素咀嚼品，列 **Country of Origin: Mexico**，不能將牙齒／口腔酵素直接等同保留消化酵素。三項都不能由提供證據確認符合顧客全部硬限制。

- **G-P1** 說 Yak Cheese `made in the USA`，直接與其證據矛盾，也未處理單一成分、酵素與不烘焙要求。
- **G-P2** 列出天然／無添加等有支持的片段，卻以 `Unmet Requirement: None identified` 掩蓋關鍵不符。
- **G-P3** 不但仍說 USA-made，還新增 `8 oz size` 與 `ensuring digestive health`；明示 unknown 的指令沒有可靠生效。

**檢索與生成都有問題：**top-3 沒提供已確認符合全部需求的商品；生成又把不符合／未知寫成符合。不能只靠換提示詞，宣稱已補救候選缺口。

### 5. 紅色偏好與受損疑慮：`beauty-train-001206`

三個 SKU 實際色號依序為 Light Golden Brown、Dark Mahogany Brown、Medium Ash Brown。系列文案提到還有紅色，不代表此 PID 就是紅色。顧客是 `Maybe a red`，屬暫定偏好；不宜硬性禁止棕色，但推薦時應交代色號差異與效果證據缺口。

- **G-P1** 推第一名溫和配方，卻把 100% gray coverage 當成可解決 brassing 的理由；提供文字沒有建立這個關係，也沒說明實際是棕色。
- **G-P2** 列了許多規格，但把其他候選的 ColorSilk 技術名稱移植到第一名 Total Color；規格列表不等於正確歸屬或需求對應。
- **G-P3** 捏造使用者有 `very light hair tones like yours`，並推論缺 keratin／amino acids 的效果；顧客未提供目前髮色。兩個提到的 PID 都是候選內合法 ID，仍不代表規格及個人化判斷正確。

## 自訂量表：助理質性檢閱，非官方分數

本次由 **Codex 助理**在看過輸出與證據後逐例檢閱，**不是使用者人工標註、獨立評審或盲測**。五維度各 0–2：當前需求、限制／偏好、證據歸屬忠實度、無依據陳述、清楚程度。2 代表充分符合、1 代表部分／有缺口、0 代表重大錯誤；各維度的精確定義及 15 筆理由保存於 `review.json`。清楚可讀不代表內容正確，PID 格式另作客觀診斷，不重複灌進主觀分數。

| Prompt | 五題平均總分／10（主觀） | 本次觀察 |
|---|---:|---|
| G-P1 | 5.8 | 較短，已有合適候選時較能直接回覆；仍有產地、規格與效果錯誤 |
| G-P2 | 4.4 | 可出現有用的需求結構，但也會只重述需求、捏造退款期限與使用者意圖 |
| G-P3 | 3.2 | 此五例沒有可靠降低幻覺；出現錯誤拒答、規格混用及新增假設，且較冗長 |

這些數字只概括本次五例的助理判斷，**沒有官方及格線，也不能推論 P1 對其他樣本普遍較好**。三種提示詞均未達到可無檢閱使用的可靠程度；「提示詞寫得更保守」並未在本次實驗保證更忠實。

## 解讀與後續候選方向

觀察支持的失敗類型包括：跨商品屬性混用、把顧客希望當成已確認規格、把商品說明當成顧客說過的話、忽略歷史／當前需求區別，以及無根據的拒答。這是根據輸出與提供文字的分析，沒有隔離模型容量、注意力機制或提示詞長度的因果影響。

若之後追加生成實驗，可固定每個 PID 的規格表與逐條證據，明確分出 confirmed／contradicted／unknown，再測試是否改善；PID 語法可以程式檢查，但檢查到合法 ID 不能證明推薦正確。這些是後續候選，本輪沒有改提示詞、補回覆或重跑。**Q2 的成果是保留完整比較與分析，不是把不好結果改成好結果。**

W08 與其官方 retrieval 成績保持凍結。下一項是 W10 public／官方環境驗收，之後 W11 將 Q1／Q2 內容納入正式 `report.pdf` 並完成下載／重現／封裝；本輪沒有產生或宣稱已繳交正式報告。

## 程式、原始產物與重現

- [generate_report.py](C:/Users/yuting/Desktop/HW2_ADL/code/script/generate_report.py)：context、凍結計畫、離線生成及可驗證前綴續跑；不自動撰寫 report.pdf。
- [w09_generation.json](C:/Users/yuting/Desktop/HW2_ADL/code/script/configs/w09_generation.json)：固定資料／排名 hashes、模型來源及生成設定。
- [audit_w09.py](C:/Users/yuting/Desktop/HW2_ADL/code/script/audit_w09.py)：重建 chat-template／token IDs／top-3 核對、raw decoding 及匯出可讀檔。
- [analyze_generation.py](C:/Users/yuting/Desktop/HW2_ADL/code/script/analyze_generation.py)：驗證主觀評語的完整 grid／分數範圍，與直接可測的診斷分開彙整。
- `results/w09_q2/plan.json`、`manifest.json`、`outputs.jsonl`：完整輸入、原始 token／文字、版本／資源與 hashes。
- `audit.json`、`raw_outputs.md`、`evidence.md`、`review.json`、`review_summary.json`：核對、完整原文與逐例分析；相關檔案均保留供 W11 提交。

在目前 WSL 專案根目錄執行。既有產物不可覆寫；重現使用新目錄，不挑選替代輸出：

```bash
.venv/bin/python -m unittest discover -s code/script/tests -p 'test_*.py' -v
.venv/bin/python code/script/generate_report.py --stage prepare --output results/w09_reproduction
.venv/bin/python code/script/generate_report.py --stage generate --output results/w09_reproduction
.venv/bin/python code/script/audit_w09.py --directory results/w09_reproduction
```

只有在生成中斷且已有完整 JSONL 前綴時，才對同目錄加 `--resume`；會核對凍結的輸入／來源與前綴順序，不重生已保存回答。已完成 15 筆後拒絕續跑。質性評語需要另行檢閱，不能把原評語冒充針對不同輸出的新評審結果。
