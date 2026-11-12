# H4 — base＋5:1 正式驗證及固定 Public 確認

日期：2026-10-25（Asia/Taipei）。狀態：**本輪完成；正式預設已在新Public前採用base＋5:1。Public原版Judge為Recall 0.933000／MRR 0.747915，優於舊版、仍為Strong，未達Boss。**

使用者已同意先驗證正式推論，通過後固定設定評估一次完整 Public。D28 接受 H3 的 Pet Recall 98→93／102 取捨，繼續驗證整體 dev 更好的 base；不回寫 H3 原先分組門檻失敗的紀錄。Public 不用於超參數、模型或分品類規則選擇。

## 固定配置

- `artifacts/training/h3_base_epoch1/checkpoint`：已訓練一次，3,864 題／242 steps／1 epoch，未重訓。
- 完整角色對話，商品標題＋描述；512 tokens、一商品一向量、768 維、FAISS exact。
- Dense 與 BM25 各取 100，RRF 常數60、固定 Dense:BM25＝5:1；BM25仍完整對話、原始詞權重。
- 正式邏輯只改 `retrieval_utils.py` student block，官方三檔保留原 SHA。
- 舊 small＋1:1 來源保存於 `code/script/snapshots/w08_default/retrieval_utils.py`，原 checkpoint 不變。新程式可明確指定 `HW2_MODEL_PATH=artifacts/training/w08_epoch1/checkpoint`、`HW2_RRF_DENSE_WEIGHT=1`、`HW2_RRF_BM25_WEIGHT=1` 重現舊配置。

## 驗證順序與目前結果

1. 77 tests 通過：`results/h4_tests_initial.log`（8.670s）、`results/h4_tests_release.log`（8.921s）。包含獨立加權公式、1:1舊配置與768維商品換序／增刪／同分測試；這些索引測試使用替身，真實品質依完整dev結果。
2. 完整 dev966／50k，正式入口無模型／權重覆寫、冷啟動、8GiB PyTorch allocator budget：**185.420963s，top50全數等同H3，通過。**allocated1,111,927,808 bytes／reserved1,407,188,992 bytes／RSS2,016,149,504 bytes。重用已評分的相同排名，不重跑dev Judge。
3. 第一次worker也成功完成且966題每個top50與H3完全一致；但parent儲存相對log路徑出錯，沒有完整外層計時報告，因此保留為`h4_dev_attempt1_*`，不當作完整驗證成功。修正`resolve().relative_to(ROOT)`，新增2項回歸測試（0.003s／OK），再完成上述冷啟動重測。模型／融合／資料不變，修正紀錄見`results/h4_measurement_repair.json`。
4. 新Public回饋前已提升並凍結base＋5:1；Public1000／75k推論完成，冷啟動274.644899s，allocated1,112,026,112 bytes／reserved1,405,091,840 bytes／RSS2,355,834,880 bytes，8GiB allocator檢查通過。`results/h4_public_plan.json`記錄選型時點與依據。
5. 原版Public Judge完成：9,146個非gold比較、**0解析失敗**、最大輸入2,399 tokens、未截斷；Qwen固定revision、bf16／batch8／greedy／maxnew12。唯一一次冷啟動649.719935s，逐題raw output／gain／rank／aggregate已核對。Judge未設8GiB cap，allocated4.653GiB／reserved8.389GiB；不能把檢索的資源驗證延伸到Judge。

`h4_validation_plan_initial.json` 與初版工具另存；在任何 H4 推論前修正分組報表 helper 的參數個數。另保留第一次計時工具、計畫及完整成功worker輸出；修正僅為parent紀錄路徑，重新凍結`h4_validation_plan.json`。沒有改變配置或觀察新Public評分結果。

## 比較基準

| 資料集 | 配置 | 官方 Recall@10 | 官方 MRR@10 |
|---|---|---:|---:|
| Public1000 | W08 small＋1:1 | 0.902000 | 0.704348 |
| Public1000 | H4 base＋5:1 | **0.933000** | **0.747915** |
| dev966 | H3 base＋5:1 | 0.870600 | 0.658298 |

Public Recall淨增**0.031000**（902→933／1000），MRR增加**0.043567063492**。Recall逐題59改善／28退步／913不變，MRR249改善／143退步／608不變。Gold命中854題、另79題有Judge認可替代品；不是只有Gold Hit。

Public Boss門檻為Recall .950／MRR .800，目前分別差**.017000／.052084523810**；Strong .885／.680兩項通過。按投影片各格獨立累加估算仍為Public **6／8**課程百分點，與舊版同級，並非整份作業75分。Private最多6與Report最多6的實際得分未知。

dev與Public不同，不能用dev判斷老師Public門檻。這次Public同時改模型與融合比例，不能將全部提升單獨歸因模型變大；相同5:1下的small/base受控比較見H3。新Public沒有用於調比例、再訓練、挑選模型或分品類規則。

## Public 分組與 Gold 診斷

每品類200題；以下只描述凍結後結果，不作調參依據。

| 品類 | small1 Recall | base5 Recall | small1 MRR | base5 MRR |
|---|---:|---:|---:|---:|
| beauty | .935000 | .935000 | .733143 | .714115 |
| home-and-kitchen | .940000 | .970000 | .762476 | .845532 |
| pet-supplies | .875000 | .935000 | .679976 | .705274 |
| sports-and-outdoors | .935000 | .975000 | .777986 | .806437 |
| toys-and-games | .825000 | .850000 | .568161 | .668220 |

Beauty MRR下降.019028；其他四品類兩項均改善。Pet的Public改善不會取消H3 dev退步的事實，兩者資料不同，也不改寫事前門檻。

Gold-only：Hit10 **.854000**／MRR10 **.663237301587**／Hit50 **.946000**，舊版分別為.819000／.615436904762／.928000。Gold是診斷，不拿來換算正式得分。

兩版有4,361組共同非gold候選，其中48組的Judge gain不同；原始兩次判斷都保留，沒有重評分挑答案。候選改變會改變組批，確切原因未隔離；結果不是獨立泛化或穩定性保證。

## 重現入口與產物

以既有WSL專案`.venv/bin/python`執行；正式檢索與Judge皆離線、全候選池、固定題目順序。

```bash
# 77項正式推論／歷史回歸測試，另有2項工具回歸
.venv/bin/python -m unittest discover -s code/script/tests -v

# 已完成的正式dev驗證；量測工具拒絕覆寫，重現請用新output檔名
.venv/bin/python code/script/run_validated.py --budget-gib 8 \
  --output results/h4_base5_dev_timing.json -- retrieval.py \
  --dialogues data/derived/dev.jsonl --products data/train/products.jsonl \
  --top-k 50 --output results/h4_base5_dev_top50.json

# 本輪唯一Public推論與Judge，以下為實際命令紀錄，不要求再跑
.venv/bin/python code/script/run_validated.py --budget-gib 8 \
  --output results/h4_base5_public_timing.json -- retrieval.py \
  --dialogues data/public/dialogues.jsonl --products data/public/products.jsonl \
  --top-k 50 --output results/h4_base5_public_top50.json
.venv/bin/python code/script/run_validated.py \
  --output results/h4_base5_public_judge_timing.json -- evaluate.py \
  --retrieval-results results/h4_base5_public_top10.json \
  --dialogues data/public/dialogues.jsonl --products data/public/products.jsonl \
  --model artifacts/models/Qwen2.5-1.5B-Instruct --batch-size 8 \
  --max-new-tokens 12 --top-k 10 --device cuda \
  --output results/h4_base5_public_graded.json
```

原始排名、Gold、原版Judge、計時、分組／逐題得失保存為`results/h4_*`；`h4_summary.json`為彙總，`h4_completion.json`由最後的`code/script/audit_release.py`核對後保存SHA seal，含checkpoint與歷史H3產物核對。本輪沒有新增訓練、dev Judge、第二次Public或W11。

## 限制

- 8GiB預算限制PyTorch allocator，不包含所有CUDA context／外部程序；RTX4090速度也不等於實體RTX3070。
- H3原FP32訓練約8.19GiB，沒有宣稱可在8GB重現相同訓練。此次只驗證推論。
- `run.sh` 仍重現歷史 W08 small；base 的重現入口是已保存的 `code/script/train_base.py`。正式提交需要在之後獲准的W11統一訓練入口、README與選定checkpoint；本輪未做封裝，不能宣稱已可提交。
- Python3.12.12／RTX4090與正式Python3.12.9／RTX3070不同；正式環境驗收仍待。
- H3分組取捨、重複使用dev選型與Judge組批敏感性仍成立；Public早在W10已看過，這次不屬於未見過的獨立測試。
- 不進W11，不建立report.pdf或封裝／下載腳本。
