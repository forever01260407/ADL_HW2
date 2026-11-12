# ADL HW2 — 多輪對話商品檢索

學號：r15922a14

根據顧客與購物助理的英文對話，從指定商品池選出前 10 件商品。
系統使用經兩階段微調的 **BAAI/bge-base-en-v1.5**，搭配 BM25，
以 **Dense : BM25 = 5 : 1** 的加權 RRF 融合排名。
方法、超參數、實驗比較與生成分析請見 `report.pdf`。

提交根目錄依投影片排列：`download.sh`、`run.sh`、`retrieval_utils.py`、
`requirements.txt`、`report.pdf`、`README.md`，其餘程式與紀錄集中於 `code/script/`。
模型、資料及輸出目錄會在執行時產生。

以下命令均在解壓後的 `r15922a14/` 目錄執行；資料路徑請換成實際位置。

## 1. 環境

作業指定 Ubuntu 22.04、Python 3.12.9、32 GB RAM、RTX 3070 8 GB。
本機開發與驗證使用 WSL Ubuntu 22.04、Python 3.12.12、RTX 4090 24 GB；
尚未在指定的實體 RTX 3070 上測試。

依助教討論區補充：助教會執行 `run.sh`；訓練沒有時間限制，但同樣須符合
8GB VRAM 限制。助教會將原始 `data/train/` 放在學號根目錄，並依提交的
`requirements.txt` 安裝套件。訓練入口已啟用 gradient checkpointing；本機完整
重現兩階段的 PyTorch reserved 峰值約 2.88／3.34 GiB，這仍不等於實體3070驗收。

使用課程環境，或在自己的 CUDA 環境中安裝套件：

```bash
python -m pip install -r requirements.txt
```

主要版本：PyTorch 2.8.0、Transformers 4.56.1、Sentence Transformers 5.1.0、
NumPy 1.26.4、CPU FAISS 1.12.0。PyTorch 須搭配可用的 CUDA 驅動；
`download.sh` 不會安裝套件或變更環境。

## 2. 下載模型

在專案根目錄執行：

```bash
bash download.sh
```

腳本下載並驗證最終模型，以及重現訓練與生成分析所需的原始 BGE-base、Qwen 模型。
最終檢索模型放在 `artifacts/training/h5_hard_negative/checkpoint`。
若只需執行檢索，可使用 `bash download.sh --skip-auxiliary`。

模型下載連結已設定在 `code/script/configs/release_assets.json`。
已在Ubuntu環境從空目錄完成遠端下載，壓縮檔與全部10個模型檔案的SHA-256核對通過。

## 3. 執行檢索與官方評分

指定待測對話與完整商品池：

```bash
python code/script/prepare_runtime.py
python retrieval.py --dialogues /path/to/dialogues.jsonl --products /path/to/products.jsonl --output ranked.json
```

第一行將隨附的原版官方入口與參考檔案複製至執行路徑，不安裝套件。
助教使用自備的官方入口即可；若檔案已存在，初始化只驗證、不覆寫不同內容。

輸出 `ranked.json`，每段對話包含 10 個有序且不重複的商品 PID。
每次執行都會對傳入商品池編碼並建立索引。正式檢索使用預設設定即可。

需要在本機評分時，使用官方提供的 `evaluate.py`：

```bash
python evaluate.py --retrieval-results ranked.json --dialogues /path/to/dialogues.jsonl --products /path/to/products.jsonl --model artifacts/models/Qwen2.5-1.5B-Instruct --batch-size 8 --output graded.json
```

## 4. 重現訓練

將課程提供的原始 Train 檔案放在以下位置；原始資料不隨提交包附送：

```text
data/train/dialogues.jsonl
data/train/products.jsonl
```

完成第 1、2 節後執行：

```bash
bash run.sh
```

`run.sh` 會先初始化隨附的固定切分與參考紀錄，再執行兩階段訓練。

固定切分為 Train 3,864 題、Dev 966 題，候選池保留完整 50,000 件商品。
訓練分成兩階段：原始 BGE-base 微調一輪，再以固定、僅由 Train 建構的難負例續訓一輪。
兩階段各為 batch 16、242 steps，學習率依序為 2e-5、5e-6，seed 42。
Dev 僅檢查切分完整性，不參與梯度或負例挖掘；訓練不使用 Public／Private。

輸出分別在 `artifacts/reproduced/base/checkpoint` 與
`artifacts/reproduced/hard/checkpoint`，不覆寫已下載的正式模型。
若輸出目錄已存在，可指定新位置：

```bash
bash run.sh --output-root artifacts/reproduced_run2
```

難負例使用隨附的固定資料 `code/script/configs/release_negatives.json`，
不必重新執行 Qwen 篩選。完整超參數與 loss 請見報告。

## 5. 生成分析與相關檔案

生成實驗使用 Qwen2.5-1.5B-Instruct，對固定 5 題各比較 3 種提示詞。
為固定比較條件，證據沿用較早的微調 BGE-small＋1:1 融合結果。
提示詞在 `code/script/configs/w09_prompts.json`，15 份原始回答與商品證據在
`code/script/records/results/w09_q2/`。重現生成可執行：

```bash
python code/script/prepare_runtime.py
python code/script/generate_report.py --stage prepare --output results/q2_reproduction
python code/script/generate_report.py --stage generate --output results/q2_reproduction
```

訓練入口為 `code/script/train_release.py`；報告產製程式為
`code/script/build_final_report.py`，額外排版套件列於
`code/script/configs/requirements-dev.txt`。正式報告已提供 PDF，執行檢索不需排版套件。
若自行重建PDF，另需 `msjh.ttc`、`msjhbd.ttc` 字型；可用
`HW2_REPORT_FONT_DIR` 指定字型目錄。字型與排版套件不影響模型訓練或檢索。
原始評分與資源紀錄集中於 `code/script/records/`，設定位於 `code/script/configs/`，
報告原稿位於 `code/script/docs/report.md`。初始化會還原程式使用的相對路徑。

較早的消融程式保留供查閱，可能需要其當時使用的舊模型；正式重現請使用上列入口，
不必依序重跑所有歷史實驗。

AI 協作：Codex 協助程式、測試與文件整理。生成分析中的主觀評分由 Codex 助理評閱，
不是獨立人工標註或官方分數；執行數據使用保存的實際輸出。
