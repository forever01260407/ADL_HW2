# H3 — 相同 W08 訓練條件，將 BGE-small 換成 BGE-base

日期：2026-10-25（Asia/Taipei）。**實驗完成：base＋5:1 的整體官方 dev Recall／MRR 均提高，但 pet Recall 退步超過事前門檻，未通過完整採用 gate，保留 small 預設。**只訓練一個正式 base 配置、兩組官方 Judge 各一次；沒有 Public 重跑、正式預設切換或 W11。

官方 dev：small＋1:1 **0.829193／0.611836** → small＋5:1 **0.846791／0.631014** → base＋5:1 **0.870600／0.658298**（Recall10／MRR10）。提高Dense比例與換base，在此次完整dev上分別都有整體收益；不能由此預告Public／Boss分數。

完整採用gate只有 **pet Recall 保護條件**未通過：98／102 → 93／102，下降0.049020，超過預定0.02；pet MRR反而0.697670 → 0.751261。這是本專案的保守工程選擇，不是老師要求每品類都不能下降。未在看到結果後放寬門檻，因此本輪沒有接入正式權重或執行8GiB cap管線驗證，也沒有將base設為預設。

同樣5:1下，base Gold Hit10 **0.753623**／MRR10 **0.550325**／Hit50 **0.886128**，small為0.723602／0.517920／0.862319，分別提高約3.00／3.24／2.38個百分點。這是Gold診斷，不能先宣稱官方graded改善或Boss通過。

## 固定的比較方式

同一份 train 3864／dev 966、候選商品完整50000；沿用 W08 的 batch qid 組成與順序，seed42、FP32、batch16、1epoch／242steps（末批8）、AdamW lr2e-5／weight decay.01、warmup25及線性下降、clip1、MNR scale20。query instruction、完整角色對話、512-token商品前綴都不變；不加hard negatives、reranker、分段或新資料。

唯一模型變因：官方 `BAAI/bge-base-en-v1.5` revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`，768維；small為384維。四項tokenizer資產逐檔hash完全相同。舊train.py／models.lock／W08 checkpoint保留，新入口重用原資料、sampler及query處理，另存base模型鎖定與下載紀錄。

先比較 small/base 的 Dense-only 與相同 Dense:BM25＝5:1。BM25 full／末輪1／title1、兩路top100、RRF常數60與同分PID次排序固定；重建base的商品／query向量，不能重用small Dense快取。保留small＋1:1及已評分small＋6:1歷史，避免把改權重與改模型當成同一項增益。

## 預先凍結的門檻

來源：`code/script/configs/h3_experiment.json`、`results/h3_experiment_plan.json`。這些是本專案工程門檻，不是課程指定指標。

- Gold：base5相對small5，Hit10至少多1題、MRR10至少+.01、Hit50不降；Dense-only的Hit10／MRR10也不降。失敗就停止，不試另一組訓練參數。
- Gold通過後才做small5及base5各一次完整dev原版Judge，Qwen／batch8／bf16／greedy／12 tokens／不截斷，兩組事前凍結。
- 品質採用：base的graded Recall不低於small5及small6，MRR分別至少+.01；對small5各品類Recall最多退.02、MRR最多退.03。配對bootstrap只作重用dev上的探索性描述，不冒稱跨seed或未見泛化證據。
- 品質通過後才補正式全域RRF權重、完整dev top50等同已評分排名、768維商品換序／增刪測試，並做8GiB顯存預算下完整dev冷啟動（allocated／reserved符合預算、RSS<32GiB、程序<1h）。這仍不是實體RTX3070／Python3.12.9驗收；限制須另列。任一門檻失敗保留small，不改門檻或降低訓練batch／精度來掩飾。

## 已完成的訓練與測試

- 74 tests／8.372秒／OK，`results/h3_tests_initial.log`。
- 官方必要資產10檔／438900399 bytes，Git blob／LFS／SHA-256全通過；`results/h3_base_download.json`。
- 四步工程smoke：FP32／batch16，權重更新、非零gradient、離線重載通過，向量最大差0；訓練迴圈1.490秒。獨立保存在`artifacts/training/h3_base_smoke/`，不是品質候選。
- 單一正式訓練：3864題／242steps，迴圈57.869秒，allocated峰值8796487680 bytes（約8.19GiB），離線重載最大差0；完整batch／loss／checkpoint manifest位於`artifacts/training/h3_base_epoch1/`。
- 訓練用量是24GB開發卡上的FP32結果，不是正式推論8GiB驗收；也不能以loss下降直接宣稱檢索品質提高。
- 原版完整50k／966題base Dense top100冷啟動175.354秒；新排名與固定5:1各966份，共1932份以獨立rank-map公式重建一致，Gold整體／五品類／六來源也由逐題rank獨立核對，`results/h3_gold_audit.json`。
- base Dense-only Gold Hit10 **0.747412**／MRR10 **0.556166**／Hit50 **0.884058**；同一5:1融合提高Hit但MRR低於base Dense-only，不能宣稱5:1對base也是最佳。本輪固定5:1隔離模型變因，不另選Dense-only送Judge或追加權重。
- 分組Gold：toys三項提高；pet MRR提高約.09525但Hit10少1題；sports Hit10多6題但MRR略降。全部得失保留，不依品類改權重。
- 原版Judge兩組preflight均最大2285 tokens、不截斷；small5有8961／base5有8932個非gold比較，固定batch8／bf16／greedy／12tokens。兩組排名於任何本輪Judge前已凍結。

## 主要產物

| 路徑 | 用途 |
|---|---|
| `code/script/fetch_base_model.py` | 官方固定revision下載與Git／LFS校驗 |
| `code/script/configs/h3_base_model.lock.json` | base版本、768維與資產manifest |
| `code/script/train_base.py` | 原W08 helper與相同訓練迴圈的base入口 |
| `code/script/base_selection.py` | 事前固定Gold／graded gate及探索性bootstrap |
| `code/script/prepare_base_experiment.py` | 訓練前凍結來源、資料、模型、超參數與門檻 |
| `code/script/evaluate_base.py` | 訓練核對、完整Gold、條件式兩組官方Judge與結果稽核 |
| `results/h3_experiment_plan.json`／`h3_evaluation_plan.json` | 凍結計畫與來源指紋 |
| `results/h3_base_dense_top100.json` | 新checkpoint完整dev原始Dense排名 |
| `results/h3_{base_dense,base5}_{top50,top10,gold}.json` | 兩種base配置排名與Gold逐題分析 |
| `results/h3_gold_screening.json` | 同配置small/base Gold、分組、逐題得失與gate |
| `results/h3_judge_plan.json` | 若Gold通過，兩組事前固定的原版Qwen輸入與參數 |
| `results/h3_{small5,base5}_graded.json` | 若執行，完整官方原始回答／gain／逐題指標 |
| `results/h3_quality_final.json` | Gold或官方品質gate的最終核對與採用資格 |

兩組條件式Judge已實際完成；正式權重／8GiB預算驗證因品質gate失敗而未執行。當前真實狀態以本文件開頭及PROGRESS.md為準。

## Gold comparison tables

| Configuration | Gold Hit@10 | Gold MRR@10 | Gold Hit@50 |
|---|---:|---:|---:|
| small Dense-only | 0.709110 | 0.509199 | 0.851967 |
| small 5:1 | 0.723602 | 0.517920 | 0.862319 |
| base Dense-only | 0.747412 | 0.556166 | 0.884058 |
| base 5:1 | 0.753623 | 0.550325 | 0.886128 |

| Category | n | Configuration | Gold Hit@10 | Gold MRR@10 | Gold Hit@50 |
|---|---:|---|---:|---:|---:|
| beauty | 511 | small Dense-only | 0.610568 | 0.418454 | 0.788650 |
| beauty | 511 | small 5:1 | 0.618395 | 0.403699 | 0.786693 |
| beauty | 511 | base Dense-only | 0.649706 | 0.443624 | 0.816047 |
| beauty | 511 | base 5:1 | 0.645793 | 0.420584 | 0.814090 |
| home-and-kitchen | 104 | small Dense-only | 0.855769 | 0.607025 | 0.971154 |
| home-and-kitchen | 104 | small 5:1 | 0.865385 | 0.638282 | 0.990385 |
| home-and-kitchen | 104 | base Dense-only | 0.903846 | 0.700950 | 0.990385 |
| home-and-kitchen | 104 | base 5:1 | 0.894231 | 0.701000 | 0.990385 |
| pet-supplies | 102 | small Dense-only | 0.833333 | 0.528933 | 0.911765 |
| pet-supplies | 102 | small 5:1 | 0.862745 | 0.609827 | 0.950980 |
| pet-supplies | 102 | base Dense-only | 0.843137 | 0.676930 | 0.950980 |
| pet-supplies | 102 | base 5:1 | 0.852941 | 0.705077 | 0.950980 |
| sports-and-outdoors | 130 | small Dense-only | 0.869231 | 0.709881 | 0.969231 |
| sports-and-outdoors | 130 | small 5:1 | 0.892308 | 0.755275 | 0.984615 |
| sports-and-outdoors | 130 | base Dense-only | 0.915385 | 0.725342 | 0.984615 |
| sports-and-outdoors | 130 | base 5:1 | 0.938462 | 0.751825 | 0.992308 |
| toys-and-games | 119 | small Dense-only | 0.722689 | 0.577228 | 0.840336 |
| toys-and-games | 119 | small 5:1 | 0.747899 | 0.565136 | 0.865546 |
| toys-and-games | 119 | base Dense-only | 0.764706 | 0.624576 | 0.915966 |
| toys-and-games | 119 | base 5:1 | 0.806723 | 0.622999 | 0.932773 |

| Source | n | Configuration | Gold Hit@10 | Gold MRR@10 | Gold Hit@50 |
|---|---:|---|---:|---:|---:|
| beauty-train | 437 | small 5:1 | 0.574371 | 0.357213 | 0.757437 |
| beauty-train | 437 | base 5:1 | 0.597254 | 0.367528 | 0.789474 |
| train-beauty-official | 74 | small 5:1 | 0.878378 | 0.678218 | 0.959459 |
| train-beauty-official | 74 | base 5:1 | 0.932432 | 0.733896 | 0.959459 |
| train-home-and-kitchen | 104 | small 5:1 | 0.865385 | 0.638282 | 0.990385 |
| train-home-and-kitchen | 104 | base 5:1 | 0.894231 | 0.701000 | 0.990385 |
| train-pet-supplies | 102 | small 5:1 | 0.862745 | 0.609827 | 0.950980 |
| train-pet-supplies | 102 | base 5:1 | 0.852941 | 0.705077 | 0.950980 |
| train-sports-and-outdoors | 130 | small 5:1 | 0.892308 | 0.755275 | 0.984615 |
| train-sports-and-outdoors | 130 | base 5:1 | 0.938462 | 0.751825 | 0.992308 |
| train-toys-and-games | 119 | small 5:1 | 0.747899 | 0.565136 | 0.865546 |
| train-toys-and-games | 119 | base 5:1 | 0.806723 | 0.622999 | 0.932773 |

## 官方Graded完整比較

| 配置 | Recall@10 | MRR@10 | 命中題數 |
|---|---:|---:|---:|
| small1 | 0.829192547 | 0.611835749 | 801/966 |
| small5 | 0.846790890 | 0.631014411 | 818/966 |
| small6 | 0.845755694 | 0.629400818 | 817/966 |
| base5 | 0.870600414 | 0.658298005 | 841/966 |

Small 1:1 → small 5:1：淨多17題、MRR +0.019179。相同5:1下，small → base：淨多23題、MRR +0.027284。Base相對原small1共多40題、MRR +0.046462。

| 品類 | n | small5 Recall | base5 Recall | small5 MRR | base5 MRR |
|---|---:|---:|---:|---:|---:|
| beauty | 511 | 0.790607 | 0.825832 | 0.563318 | 0.579350 |
| home-and-kitchen | 104 | 0.913462 | 0.942308 | 0.700839 | 0.764370 |
| pet-supplies | 102 | 0.960784 | 0.911765 | 0.697670 | 0.751261 |
| sports-and-outdoors | 130 | 0.938462 | 0.976923 | 0.789377 | 0.797573 |
| toys-and-games | 119 | 0.831933 | 0.848739 | 0.630552 | 0.672779 |

| 來源 | n | small5 Recall | base5 Recall | small5 MRR | base5 MRR |
|---|---:|---:|---:|---:|---:|
| beauty-train | 437 | 0.762014 | 0.803204 | 0.532496 | 0.544671 |
| train-beauty-official | 74 | 0.959459 | 0.959459 | 0.745335 | 0.784138 |
| train-home-and-kitchen | 104 | 0.913462 | 0.942308 | 0.700839 | 0.764370 |
| train-pet-supplies | 102 | 0.960784 | 0.911765 | 0.697670 | 0.751261 |
| train-sports-and-outdoors | 130 | 0.938462 | 0.976923 | 0.789377 | 0.797573 |
| train-toys-and-games | 119 | 0.831933 | 0.848739 | 0.630552 | 0.672779 |

## 逐題得失與不確定性

- 相對small5：Recall 57題改善／34題退步；MRR 223題改善／168題退步。Base有728題命中gold，另113題由替代品補命中，共841題。
- 10,000次qid配對bootstrap（seed42）的探索性95% CI：Recall差[0.004141, 0.043478]，MRR差[0.009052, 0.045353]。這是已重複使用dev、單一訓練seed的描述性檢查；相關樣本可能影響區間，不能當作未見資料或跨seed保證。
- 共有4715個相同非gold配對，其中55個gain不同；候選／組批改變，原因未隔離。保留原始結果，不重跑修分。
- Pet共6題Recall退步、1題改善；6題舊模型被認可的候選均離開base前十名，並非這6題仍有同一正例卻單純翻轉Judge判斷。完整候選與原始回答見results/h3_judge_diagnostics.json。

| Pet退步qid | small5 Gold rank@50 | base5 Gold rank@50 |
|---|---:|---:|
| train-pet-supplies-000004 | 38 | 30 |
| train-pet-supplies-000194 | 7 | 12 |
| train-pet-supplies-000234 | None | None |
| train-pet-supplies-000383 | 4 | 14 |
| train-pet-supplies-000458 | 4 | 23 |
| train-pet-supplies-000540 | 3 | None |

## 資源、採用與重現範圍

- 兩組Judge均0解析失敗；small5程序562.244秒、base5程序553.435秒。全部原始回答／gain／aggregate、固定batch8／12tokens／模型指紋與排名皆通過核對。
- FP32正式訓練迴圈57.869秒、腳本總時間95.133秒，allocated峰值約8.19GiB；相同訓練需要超過8GiB可用顯存，未宣稱可在3070 8GB重現。文件明確列的是測試環境，是否也約束重新訓練未獲新官方答案，後續README需如實列訓練設備需求。
- 原版base Dense完整dev程序175.354秒，allocated約1.04GiB／reserved約1.31GiB、RSS約1.88GiB；這只是4090／Python3.12.12量測，尚未做正式5:1的8GiB cap或3070實機驗收。
- 完整採用gate只敗於Pet Recall下降0.049020，超過事前容許0.02。正式權重整合、正式全量排名一致性及8GiB cap測試依條件未執行，不得稱為已通過；retrieval_utils.py與small預設保持原SHA-256。
- Base checkpoint與全部數據保留，但本輪不提升預設，不再訓練／調比例／重跑Judge、不跑Public或W11。若後續接受Pet取捨，需明確記錄新政策再完成正式推論／資源檢查；原gate失敗紀錄不改寫。
- 本輪沒有新的Public得分，舊W08 Public 0.902000／0.704348只代表先前small1配置，不能加上dev增益或宣稱Boss已過。

## 本次實際執行命令

```bash
.venv/bin/python code/script/fetch_base_model.py
.venv/bin/python -m unittest discover -s code/script/tests -v
.venv/bin/python code/script/prepare_base_experiment.py
.venv/bin/python code/script/train_base.py --mode smoke
.venv/bin/python code/script/train_base.py --mode train
.venv/bin/python code/script/evaluate_base.py --stage prepare
HW2_RETRIEVER=dense HW2_RERANK=0 HW2_MODEL_PATH=artifacts/training/h3_base_epoch1/checkpoint \
.venv/bin/python code/script/run_measured.py --output results/h3_base_dense_timing.json -- retrieval.py \
  --dialogues data/derived/dev.jsonl --products data/train/products.jsonl --top-k 100 --output results/h3_base_dense_top100.json
.venv/bin/python code/script/evaluate_base.py --stage screen
.venv/bin/python code/script/evaluate_base.py --stage prepare-judge
.venv/bin/python code/script/evaluate_base.py --stage judge --candidate small5
.venv/bin/python code/script/evaluate_base.py --stage judge --candidate base5
.venv/bin/python code/script/evaluate_base.py --stage final
```

所有既存產物拒絕覆寫。重跑需另立實驗路徑，不能直接覆蓋這次checkpoint、排名或Judge；不把smoke當成第二個品質候選。
