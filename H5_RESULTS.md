# H5 — Train限定的難負例續訓

日期：2026-10-25（Asia/Taipei）。狀態：**本輪完成；H5難負例續訓模型已採用，維持5:1。完整Dev Graded Recall .893374741201／MRR .707190919846，正式966題top50重現及8GiB allocator檢查通過。沒有新Public成績，不能宣稱Boss已達成。**

本輪只進行一個事前固定的續訓配置。基準是已採用的H3微調base＋5:1；舊checkpoint及H4紀錄不覆寫。本輪不使用Public挖掘負例、產生訓練樣本、更新權重或調參，也不重跑Public、不進W11。

## 資料與篩選

- Train來源僅`data/derived/train.jsonl`的3864題與`data/train/products.jsonl`完整50000商品。Dev僅用於原有split完整性防護，以及訓練後評估；不產生訓練／挖掘樣本。
- 目前base對Train保存Dense top100；固定從第5及15名開始，最多第50名，各找一個不是gold且正規化文案不同的候選，每題兩個。
- 固定既有Qwen revision／原版relevance prompt，bf16、batch8、greedy、maxnew12、不截斷；只接受精確可解析JSON `{"relevant": false}`，每題選第一個合格者、最多一個，全部原始輸出保存。
- 這是模型篩選的pseudo-negative，可能誤判，不等同人工確認。沒有合格者不硬塞負例，仍保留該Train正例。

## 唯一訓練配置

從`artifacts/training/h3_base_epoch1/checkpoint`續訓1epoch，batch16、seed42、FP32、lr5e-6、warmup10%、AdamW wd.01、clip1、cosine scale20、512tokens；全3864正例按原sampler使用一次，預期242steps。開啟non-reentrant gradient checkpointing並設定8GiB PyTorch allocator預算；先做4步工程smoke，再正式訓練一次。

Loss保留原in-batch正例商品對比，另讓每個query至多對比自己經篩選的負例。負例不廣播為其他query的錯誤商品；若已在同batch正例流中則不重複計分。無合格負例的query仍使用原in-batch對比。

這個實驗測的是「額外續訓＋篩選難負例」整體介入，沒有等步數、不加負例的續訓對照，不能單獨聲稱提升必然由難負例造成。原H3 FP32約8.19GiB的問題不會被隱藏；H5是否能在8GiB allocator內訓練，須依實測。

## 事前驗證流程

1. **84 tests／9.130s／OK**：候選窗口、正例／重複文案排除、train qid邊界、嚴格JSON false、teacher對齊、缺少負例退化、batch正例流重複排除、獨立loss數值與gradient檢查，及既有回歸。`results/h5_tests_initial.log`。
2. Train Dense top100已保存，3864題／50k、8GiB allocator預算，冷啟動206.750447s；allocated1,111,927,808 bytes、reserved1,407,188,992 bytes、RSS2,051,526,656 bytes。固定選出7728候選配對，完整Train覆蓋；原版Qwen兩候選篩選進行中。至少25% Train題目有合格負例，否則停止、不硬造標籤。
3. Qwen完成7728配對／513.737318s／0解析失敗，最大輸入2698tokens、不截斷；嚴格篩選後3796題各取1負例，68題不加負例。全部7728候選的題號／reference／原Dense rank與train覆蓋獨立稽核通過；3796對正負例在實際512token編碼下相同者0對，避免不可辨識的矛盾訓練。來源與原始判斷均保存。
4. 四步FP32／batch16工程smoke通過：64次own-negative使用，2.533772s，allocated2,934,958,592 bytes／reserved3,244,294,144 bytes，重載最大向量差0，8GiB allocator內。
5. **唯一正式續訓完成**：全3864題／242steps／121.215269s，實際3795次own-negative對比、1次因同batch正例流重複而略過。allocated2,952,656,896 bytes／reserved3,548,381,184 bytes，FP32、batch16、8GiB allocator內；權重實際更新、離線重載最大向量差0。完整batch／loss／checkpoint manifest保存於`artifacts/training/h5_hard_negative/`。
6. 完整Dev966／50k新Dense top100已保存，冷啟動169.448134s、8GiB allocator內；固定5:1離線融合，1932份排名由獨立rank-map核對，Gold再由實際rank重算一致。hybrid Hit10多64題、MRR+.065438、Hit50多32題；Dense Hit10／MRR也提高，事前五項Gold gate全部通過。
7. 唯一原版Dev Judge完成：8868個非gold比較、**0解析失敗**，最大輸入2285tokens、未截斷，bf16／batch8／greedy／12tokens。冷啟動534.721328s，raw output／gain／rank／aggregate核對通過。Recall **.893374741201**／MRR **.707190919846**，相對基準+.022774327122／+.048892914654，事前整體品質gate通過。
8. **正式採用驗證通過**：選定預設後完整84tests／8.350s／OK；無環境覆寫的完整Dev966／50k Hybrid冷啟動175.911588s，每題每個top50全數等同已評分H5，差異0。8GiB allocator內allocated1,111,927,808 bytes／reserved1,407,188,992 bytes，RSS2,028,969,984 bytes。官方三檔SHA不變，只更換student block中的checkpoint路徑；5:1與其他檢索邏輯不變，舊H4來源另存`code/script/snapshots/h4_default/retrieval_utils.py`。

## 基準（不同資料集分開）

| 資料集／配置 | Gold Hit10 | Gold MRR10 | Gold Hit50 | Graded Recall10 | Graded MRR10 |
|---|---:|---:|---:|---:|---:|
| dev966／目前base5 | .753623 | .550325 | .886128 | .870600 | .658298 |
| dev966／H5新模型5:1 | **.819876** | **.615763** | **.919255** | **.893375** | **.707191** |

基準Dense-only的dev Gold為.747412／.556166／.884058；H5 Dense-only為.813665／.637067／.916149。固定5:1提高Hit，但MRR低於Dense-only；本輪不追加比例或另一組Judge挑分。這些是Dev比較，不是Public門檻，新H5本輪沒有Public分數。

| Dev品類 | 基準Hit10 | H5 Hit10 | 基準MRR10 | H5 MRR10 | 基準Hit50 | H5 Hit50 |
|---|---:|---:|---:|---:|---:|---:|
| beauty／511 | .645793 | .724070 | .420584 | .495313 | .814090 | .866928 |
| home-and-kitchen／104 | .894231 | .951923 | .701000 | .769723 | .990385 | 1.000000 |
| pet-supplies／102 | .852941 | .931373 | .705077 | .765138 | .950980 | .970588 |
| sports-and-outdoors／130 | .938462 | .953846 | .751825 | .791548 | .992308 | 1.000000 |
| toys-and-games／119 | .806723 | .873950 | .622999 | .678368 | .932773 | .941176 |

以上為Gold-only，下面另列官方Graded，不混用指標。

## 官方 Dev Graded 得失

| 品類 | 基準Recall | H5 Recall | 基準MRR | H5 MRR |
|---|---:|---:|---:|---:|
| beauty／511 | .825832 | .841487 | .579350 | .617135 |
| home-and-kitchen／104 | .942308 | .971154 | .764370 | .816197 |
| pet-supplies／102 | .911765 | .980392 | .751261 | .828482 |
| sports-and-outdoors／130 | .976923 | .969231 | .797573 | .853205 |
| toys-and-games／119 | .848739 | .890756 | .672779 | .735161 |

整體命中**841→863／966，淨多22題**；Recall42題改善／20題退步，MRR214改善／105退步。Sports Recall127→126／130，少1題；其餘四品類兩項改善，全部保留，不以品類加特殊規則。

第一個被接受的商品：第一名**534→593**，第2–10名307→270，前十未命中125→103。H5的792題Gold命中，另71題有被接受替代品，合計863；這是官方Graded而不只是Gold指標。

共同非gold候選5819組，其中40組Judge gain不同，原始輸出全部保留、不重評分拼接；確切組批敏感性原因未隔離。單seed、重用Dev選型、沒有等步數無負例續訓對照，不能當作獨立泛化或難負例單獨因果證據。

`h5_quality_final.json`保留品質階段當時的`quality_eligible_pending_formal_validation`；正式驗證已完成，最終採用狀態以`h5_adoption.json`及稽核後的`h5_completion.json`為準。本輪未跑Public，不能由這些Dev數字宣稱Public／Boss已達成。歷史H4 Public .933000／.747915屬於舊checkpoint，不是新H5的Public成績。

## 程式與紀錄

- `configs/h5_hard_negative.json`、`results/h5_training_plan.json`：品質配置／資料及來源hash，在新Dev結果前固定。
- `code/script/hard_negative_data.py`：Train候選、嚴格teacher篩選、逐題負例與batch對齊。
- `code/script/hard_negative_loss.py`：只加入自己驗證負例的InfoNCE。
- `code/script/train_hard_negative.py`：smoke／正式訓練、FP32 checkpointing、8GiB allocator、重載與manifest。
- `code/script/evaluate_hard_negative.py`：訓練完整性、完整Dev Gold、條件式唯一Judge與逐題比較；在第一次新Dev結果前另凍結評估來源。
- `artifacts/training/h5_hard_negative_smoke/`與`h5_hard_negative/`均已完成；正式預設為`artifacts/training/h5_hard_negative/checkpoint`，不覆寫任何既有模型。
- `code/script/adopt_hard_negative.py`只在品質gate通過後保存舊來源、切換checkpoint並測試完整正式推論；失敗會回復舊來源。`h5_adoption.json`確認已通過。
- `code/script/audit_hard_negative.py`核對原始teacher、實際Train qid／batch／loss／weight更新、Gold／Graded與歷史H4 seal；通過後保存`h5_completion.json`。這是實驗產物稽核，不是繳交封裝腳本。

## 實際執行入口

以下為已完成的實驗命令紀錄，不要求再跑；工具拒絕覆寫既有產物。歷史資料／來源hash與固定輸出在最終W11需整合成可重現的新工作目錄／輸出入口，本輪未修改`run.sh`或建立下載腳本。

```bash
.venv/bin/python -m unittest discover -s code/script/tests -v
.venv/bin/python code/script/hard_negative_data.py --stage prepare
HW2_RETRIEVER=dense HW2_MODEL_PATH=artifacts/training/h3_base_epoch1/checkpoint HW2_RERANK=0 \
  .venv/bin/python code/script/run_validated.py --budget-gib 8 \
  --output results/h5_train_dense_timing.json -- retrieval.py \
  --dialogues data/derived/train.jsonl --products data/train/products.jsonl \
  --top-k 100 --output results/h5_train_dense_top100.json
.venv/bin/python code/script/hard_negative_data.py --stage candidates
.venv/bin/python code/script/hard_negative_data.py --stage filter
.venv/bin/python code/script/train_hard_negative.py --mode smoke
.venv/bin/python code/script/train_hard_negative.py --mode train
.venv/bin/python code/script/evaluate_hard_negative.py --stage prepare
HW2_RETRIEVER=dense HW2_MODEL_PATH=artifacts/training/h5_hard_negative/checkpoint HW2_RERANK=0 \
  .venv/bin/python code/script/run_validated.py --budget-gib 8 \
  --output results/h5_dev_dense_timing.json -- retrieval.py \
  --dialogues data/derived/dev.jsonl --products data/train/products.jsonl \
  --top-k 100 --output results/h5_dev_dense_top100.json
.venv/bin/python code/script/evaluate_hard_negative.py --stage screen
.venv/bin/python code/script/evaluate_hard_negative.py --stage prepare-judge
.venv/bin/python code/script/evaluate_hard_negative.py --stage judge
.venv/bin/python code/script/evaluate_hard_negative.py --stage final
.venv/bin/python code/script/adopt_hard_negative.py
.venv/bin/python code/script/audit_hard_negative.py
```

8GiB allocator不是實體RTX3070硬體模擬；開發仍RTX4090／Python3.12.12。H5續訓本身符合預算，不代表來源H3原FP32訓練8.19GiB問題已解決。官方Dev Judge未做8GiB cap，allocated4.568GiB／reserved9.396GiB；不要把BGE訓練／推論的預算檢查延伸為Judge或整條從原始模型重現鏈已驗收。未建立report.pdf／下載或封裝腳本，`run.sh`與README最終一致性仍屬未開始的W11。
