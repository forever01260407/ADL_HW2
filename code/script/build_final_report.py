#!/usr/bin/env python3
"""Build report.pdf and editable report.md from saved, audited experiments.

Requires reportlab; no retrieval, generation, Judge or training is executed.
"""
from __future__ import annotations

import argparse
import json
import os
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether)

ROOT=Path(__file__).resolve().parents[2]
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def metrics(path):return read(path)['metrics']
def fmt(x):return f'{x:.6f}'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT,help='Write report.pdf and report.md here')
    args=parser.parse_args()
    output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=True)
    fonts=Path(os.environ.get('HW2_REPORT_FONT_DIR', 'C:/Windows/Fonts'))
    pdfmetrics.registerFont(TTFont('JhengHei',str(fonts/'msjh.ttc'),subfontIndex=0))
    pdfmetrics.registerFont(TTFont('JhengHei-Bold',str(fonts/'msjhbd.ttc'),subfontIndex=0))
    pdfmetrics.registerFontFamily('JhengHei',normal='JhengHei',bold='JhengHei-Bold',italic='JhengHei',boldItalic='JhengHei-Bold')
    styles=getSampleStyleSheet()
    common=dict(fontName='JhengHei',wordWrap='CJK',textColor=colors.HexColor('#233047'))
    styles.add(ParagraphStyle('TitleZH',fontName='JhengHei-Bold',fontSize=23,leading=31,spaceAfter=14,textColor=colors.HexColor('#123856')))
    styles.add(ParagraphStyle('H1ZH',fontName='JhengHei-Bold',fontSize=16,leading=23,spaceBefore=10,spaceAfter=10,textColor=colors.HexColor('#123856'),keepWithNext=True))
    styles.add(ParagraphStyle('H2ZH',fontName='JhengHei-Bold',fontSize=12,leading=18,spaceBefore=11,spaceAfter=6,textColor=colors.HexColor('#176877'),keepWithNext=True))
    styles.add(ParagraphStyle('BodyZH',fontSize=10.3,leading=16,spaceAfter=8,**common))
    styles.add(ParagraphStyle('SmallZH',fontSize=8.5,leading=12,spaceAfter=6,**common))
    styles.add(ParagraphStyle('RawZH',fontSize=9,leading=12.5,spaceAfter=9,borderColor=colors.HexColor('#d6e2ea'),borderWidth=.5,borderPadding=7,backColor=colors.HexColor('#f6f8fa'),**common))
    styles.add(ParagraphStyle('RawAppendix',fontSize=8.6,leading=11.6,spaceAfter=5,borderColor=colors.HexColor('#d6e2ea'),borderWidth=.5,borderPadding=4,backColor=colors.HexColor('#f6f8fa'),**common))
    styles.add(ParagraphStyle('H2Appendix',fontName='JhengHei-Bold',fontSize=10.5,leading=14,spaceBefore=6,spaceAfter=4,textColor=colors.HexColor('#176877'),keepWithNext=True))
    story=[];md=[]
    def text(value,style='BodyZH'):
        story.append(Paragraph(escape(value).replace('\n','<br/>'),styles[style]));md.append(value+'\n')
    def h(value,level=1):
        story.append(Paragraph(escape(value),styles['H1ZH' if level==1 else 'H2ZH']));md.append('#'*(level+1)+' '+value+'\n')
    def page():story.append(PageBreak())
    def table(headers,rows,widths=None):
        cells=[[Paragraph(escape(str(c)),styles['SmallZH']) for c in row] for row in [headers,*rows]]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5f0f3')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fa')]),
            ('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#8eafb9')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
            ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.append(t);story.append(Spacer(1,8))
        md.extend(['| '+' | '.join(map(str,headers))+' |','|'+'---|'*len(headers),*['| '+' | '.join(map(str,r))+' |' for r in rows],''])
    width=174*mm
    text('ADL HW2: Conversational Product Retrieval','TitleZH')
    text('Retrieval & Reranking Strategy / Generation Analysis','H2ZH')
    text('學號：r15922a14　日期：2026-10-25\n最終配置：Train-only 難負例續訓 BGE-base + BM25，加權 RRF 5:1。')
    h('摘要與任務界線')
    text('本作業從顧客與助理的多輪購物對話中，辨識當前商品需求與限制，對指定商品池輸出十個有序、有效且不重複的 PID。主要評分對象是檢索排名；Q2 才使用固定 Qwen 模型，依檢索商品生成回覆。對話處理、檢索模型及融合策略由本專案設計，官方 Judge 與評分規則保持不變。')
    dev=metrics('results/h5_base5_graded.json');pub=metrics('results/h6_h5_public_graded.json')
    table(['資料集','題數／商品池','Graded Recall@10','Graded MRR@10'],[
        ['Dev','966 / 50,000',fmt(dev['graded_recall@10']),fmt(dev['graded_mrr@10'])],
        ['Public','1,000 / 75,000',fmt(pub['graded_recall@10']),fmt(pub['graded_mrr@10'])]], [36*mm,44*mm,47*mm,47*mm])
    text('本機 Public 的兩項均高於 Boss .950 / .800，Recall 餘裕為2題、MRR約.002404；Private 尚無成績。模型選擇與負例建構使用 Train／Dev，Public 用於固定配置後的測試。較早的小模型及一般微調base也曾測過Public，因此此結果並非首次接觸該測試集的獨立驗證。')
    h('資料與切分',2)
    text('原始 Train 有4,830段對話、50,000商品。以seed42進行群組切分：3,864題訓練、966題驗證；相同target PID、正規化商品文字及正規化對話透過傳遞群組保持同一分區。Dev不做梯度或負例挖掘，訓練載入器只額外讀取Dev檢查切分完整性。檢索與訓練始終保留完整50k商品池。Public為1,000題／75k，Private為2,000題／100k但本地不可得。')
    text('商品輸入是 title + 換行 + description。訓練資料使用課程提供的Train，預訓練權重來自官方BAAI；Qwen用於負例篩選、評分與生成分析。未加入外部商品訓練資料。')
    page()
    h('Q1.1 系統架構與多輪對話')
    table(['階段','處理流程','設計理由'],[
        ['Offline indexing','PID排序 → title+description → BGE向量 → L2正規化 → FAISS IndexFlatIP；另建立BM25倒排表','語意與詞面訊號互補；不部署外部向量資料庫。'],
        ['Online retrieval','完整角色對話 → BGE query / BM25 query → 各top100 → 5:1 RRF → top10 PID','只用rank融合，避免不同分數尺度直接相加。'],
        ['正式執行','每次針對傳入商品池重建索引，完成後釋放','離線開發快取只作消融，正式推論不依賴歷史測試排名。'],
        ['Q2 generation','固定較早小模型的top3 → 相同證據的3種prompt → Qwen回覆','比較生成提示詞時固定檢索與證據。']], [30*mm,80*mm,64*mm])
    text('「Offline」表示可與query處理分離的商品編碼／建索引階段；正式作業的冷啟動仍包含這個階段，沒有省略建索引時間。FAISS IndexFlatIP是精確內積搜尋，正規化後等同cosine；沒有近似索引造成的召回損失。')
    h('對話與商品表示',2)
    text('保留原有 user: / assistant: 角色及全部回合。Query前加 BGE英文檢索指令 Represent this sentence for searching relevant passages: 。若超過512 tokens，保留原對話開頭最多128 tokens及可容納的結尾，反覆檢查實際token預算；不做LLM摘要或改寫。Dev query實測最長200 tokens，因此本輪Dev沒有觸發截斷。')
    text('每個商品產生一個768維向量，讀取前512 tokens，沒有額外分段。Train商品約26.976%超過此長度，尾端規格可能被Dense忽略。BM25讀取完整文字，可補充詞面訊號，但仍可能漏掉需要語意理解的尾端條件。')
    h('BM25與融合',2)
    text('BM25採k1=1.5、b=.75、正值IDF log(1+(N-df+.5)/(df+.5))；大小寫統一，保留否定詞、小數及連字號型號，不加stemmer或停用詞。完整對話每個不同詞項計一次，最後user與title權重都為1。零詞面命中不獲得BM25融合票。')
    text('最終分數 RRF(p) = 5/(60 + rank_dense(p)) + 1/(60 + rank_bm25(p))，只累加該商品實際出現的路線。兩路各取100，輸出top10；商品去重，融合同分依PID排序，Dense邊界同分先全部納入再決定PID次序。5:1是開發集選定的全域比例，沒有依Public品類切換參數。')
    page()
    h('Q1.2 訓練資料、難負例與loss')
    text('訓練分兩階段，使用同一組3,864題Train及相同組批順序。第一階段從原始BGE-base微調一輪，第二階段由該模型加入經篩選的難負例，再續訓一輪。Qwen權重保持固定。')
    table(['超參數','第一階段：一般微調','第二階段：難負例續訓'],[
        ['唯一Train題數 / steps','3,864 / 242','3,864 / 242'],['batch / seed','16 / 42','16 / 42'],
        ['epochs / precision','1 / FP32','1 / FP32'],['learning rate','2e-5','5e-6'],
        ['optimizer / weight decay','AdamW / .01','AdamW / .01'],['warmup / schedule','25 steps / linear decay','25 steps / linear decay'],
        ['cosine scale / grad clipping','20 / 1','20 / 1'],['max tokens / dimension','512 / 768','512 / 768'],
        ['negative policy','其他batch正例；排除同PID／同文案','相同in-batch + 每題自己的合格難負例']], [61*mm,56*mm,57*mm])
    text('固定seed42打亂並組批，每題每階段使用一次；同batch不含同PID或正規化同商品文案的正例，末批8題。這只降低明顯重複假負例，不代表所有其他商品都確定不相關。')
    h('Train限定的難負例建構',2)
    text('以第一階段微調模型對3,864題Train及50,000商品取得Dense top100。從第5與15名起各找第一個非gold、非重複PID／文案候選，最多查到第50名，共7,728配對。以原版Qwen prompt、bf16、batch8、greedy及12 tokens上限篩選；只有可解析且完整JSON為 {"relevant": false} 的候選可保留，每題最多取第一個。負例挖掘僅使用Train。')
    text('3,796題有合格pseudo-negative，68題沒有但仍保留原in-batch訓練。每題的難負例只加在該query的logits，不廣播成其他query的負例；若已出現在batch正例流則mask，實際3,795次難負例使用／1次重複略過。教師判斷不等於人工真負例，原始512-token正負輸入完全相同的配對為0。')
    text('令 s(q,p)=20*cos(q,p)，batch正例為p_j，則 L_i = -log[ exp(s(q_i,p_i)) / (sum_j exp(s(q_i,p_j)) + I_i*exp(s(q_i,n_i))) ]。第一階段I_i=0；第二階段僅有合格且未重複的own-negative時I_i=1。目的是把容易混淆的替代候選拉開，同時避免把未驗證的跨query難負例套給整批。')
    page()
    h('Q1.3 開發集比較與策略取捨')
    paths=[('原始BGE-small，Dense','w06_b0_dev_graded.json'),('BM25 only','h1_bm25_dev_graded.json'),
        ('raw small + BM25 1:1','h1_hybrid_dev_graded.json'),('MiniLM重排Dense top50','w07_a1_dev_graded.json'),
        ('fine-tuned small + 1:1','w08_epoch1_dev_graded.json'),('fine-tuned small + 5:1','h3_small5_graded.json'),
        ('fine-tuned base + 5:1','h3_base5_graded.json'),('難負例續訓base + 5:1','h5_base5_graded.json')]
    table(['完整Dev966／50k商品','Graded Recall@10','Graded MRR@10'],[[label,fmt(metrics('results/'+p)['graded_recall@10']),fmt(metrics('results/'+p)['graded_mrr@10'])] for label,p in paths],[90*mm,42*mm,42*mm])
    text('BM25-only的MRR高於原Dense，但Recall稍低；Hybrid對原Dense兩項都改善，支持兩種訊號互補。通用MiniLM cross-encoder重排退步，故不採用；這不代表所有reranker都無效。該次pair中45.04%商品被截斷，但可完整放入的gold分組也退步，因此不能把失敗全部歸因於截斷。')
    text('另比較BM25只讀user回合、短指代補前一assistant語境，以及末輪需求／title加權。通過Gold快篩的user-only＋末輪2倍配置，在官方評分中Recall少2題、MRR提高；因未同時改善，最終保留完整對話。')
    h('權重消融及模型容量',2)
    text('以同一微調BGE-small的Dense與BM25 top100重組排名，先確認1:1融合能重現原排名，再於完整Dev比較Dense-only及Dense權重0.75、1、1.1、1.5、2、3、4、5、6、8、10。5:1的Gold Hit@10最高，為699/966；6:1的Gold MRR@10最高，為.518419；8:1與10:1回落。這是已測設定中的比較，未做連續參數最佳化。')
    text('官方Dev評分中，小模型6:1為.845756／.629401，5:1為.846791／.631014，因此固定5:1比較模型容量。一般微調base為.870600／.658298，整體改善；但pet Recall由98降至93/102，採用時保留此分組取捨。')
    text('加入難負例續訓後，Dev Recall提高.022774、MRR提高.048893，命中由841增至863題，第一名被接受由534增至593題。Sports Recall少1題但MRR提高，其餘四類兩項皆提高。因未設置相同步數、但不加難負例的續訓對照，此改善反映額外續訓與難負例的整體效果。')
    page()
    h('Q1.4 評估協定、Public與資源')
    text('Gold Hit@10只檢查原target是否在top10；Gold MRR是target在top10的倒數名次。官方把gold的gain設1，其他候選由固定Qwen判斷0/1。每題Graded Recall=max(gain)，MRR=max(gain/rank)；因此官方可接受替代品，分數不低於同排名的Gold。兩者不可混用。')
    table(['固定Public配置','Recall@10','MRR@10','Gold Hit@10 / Hit@50'],[
        ['微調small，1:1','0.902000','0.704348','0.819 / 0.928'],
        ['一般微調base，5:1','0.933000','0.747915','0.854 / 0.946'],
        ['難負例續訓base，5:1','0.952000','0.802404','0.904 / 0.968']], [58*mm,32*mm,32*mm,52*mm])
    text('最終模型對1,000題及75,000商品執行一次完整檢索與原版Judge。9,096組非gold比較皆可解析，最長prompt為2,399 tokens，未截斷；904題命中gold，另48題由合格替代品補足，共952題。相較一般微調base，Recall改善29題、退步10題，MRR改善196題、退步83題。原始排名、Judge回答與彙整數據均保存。')
    groups=read('results/h6_summary.json')['by_category_vs_h4']
    table(['Public品類（各200題）','一般微調 Recall / MRR','難負例續訓 Recall / MRR'],[[c.removeprefix('public-test-'),fmt(v['baseline']['graded_recall@10'])+' / '+fmt(v['baseline']['graded_mrr@10']),fmt(v['candidate']['graded_recall@10'])+' / '+fmt(v['candidate']['graded_mrr@10'])] for c,v in groups.items()],[65*mm,54*mm,55*mm])
    text('Dev beauty占511/966，約52.9%；Public為20%。題目與品類比例不同，Dev、Public與Private成績不可直接換算。兩次Public評分共同的5,970組非gold配對中，有54組判斷不同；候選與組批差異的影響尚未隔離。實驗僅使用單一訓練seed，並反覆使用同一Dev，泛化與Judge穩定性仍有限制。')
    table(['量測／RTX4090開發機','秒數','allocated / reserved GiB','RSS GiB'],[
        ['難負例續訓迴圈','121.215','2.750 / 3.305','見原manifest'],
        ['完整Public檢索','272.102','1.036 / 1.309','2.190'],
        ['官方Qwen評分','633.660','4.653 / 9.096','6.134']], [62*mm,28*mm,57*mm,27*mm])
    text('開發為WSL Ubuntu22.04／Python3.12.12／RTX4090，正式要求為Ubuntu22.04／Python3.12.9／RTX3070 8GB。檢索在8GiB PyTorch allocator內、冷啟動<1h、RSS<32GiB；此cap不涵蓋CUDA context／其他程序，不是實體3070驗收。Judge未cap，不宣稱其8GB驗收通過。')
    repro_path=ROOT/'results/w11_reproduction_comparison.json'
    if repro_path.exists():
        repro=read('results/w11_reproduction_comparison.json')
        text('另從原始BGE-base完整重現兩階段訓練，各3,864題、242 steps；使用相同組批、FP32與超參數，並啟用gradient checkpointing降低記憶體需求。兩階段各199個參數張量的最大差均為0，模型檔SHA-256也一致。正式推論仍使用原本已測試的checkpoint。')
    else:
        text('提交的run.sh另存兩階段重訓權重，不覆寫已測H5。重現版第一階段增加non-reentrant gradient checkpointing以降低原H3 FP32約8.19GiB的記憶體需求；重現驗證詳細數據見隨附W11紀錄，未聲稱實體3070或bitwise重現。')
    page()
    h('Q2.1 生成實驗：固定證據比較三種prompt')
    text('生成分析使用固定的Qwen2.5-1.5B-Instruct。五題在生成前由Dev依固定seed與選取規則挑選，涵蓋3／5則訊息、否定、歷史與不確定語句；標記只表示詞面線索。此實驗沿用較早微調BGE-small＋BM25 1:1的top3，不是最終base的檢索結果。同題三種prompt使用相同對話與商品證據，輸入不含gold或Judge結果。')
    text('模型revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306；bf16、batch1、seed0、greedy、最多256 new tokens、總預算4096／輸入3840、商品上限900 tokens。本次15/15 EOS、最長input2337／output211 tokens、0商品證據截斷、0空答。15筆全部原文列在附錄，完整對話與商品證據亦隨檔提交。')
    prompts=read('code/script/configs/w09_prompts.json')
    prompt_names={'G-P1':'直接推薦','G-P2':'需求與證據對照','G-P3':'明示不確定性'}
    h('共通system prompt（完整原文）',2)
    text(prompts['system'],'RawZH')
    for prompt in prompts['prompts']:
        h(prompt_names[prompt['id']]+' / '+prompt['label'],2);text(prompt['instruction'],'RawZH')
    text('三種提示詞均以JSON提供英文對話與三件商品，並要求英文回答。比較重點為需求掌握、商品規格歸屬與無依據內容。')
    page()
    h('Q2.2 客觀診斷與質性比較')
    review=read('results/w09_q2/review_summary.json')
    table(['prompt','任一PID／5','指定PID格式／5','>120 words／5','主觀均分／10'],[
        [prompt_names[k],str(v['measured_diagnostics']['outputs_mentioning_any_pid']),str(v['measured_diagnostics']['outputs_with_requested_pid_format']),str(v['measured_diagnostics']['over_120_words']),str(v['subjective_mean_total_out_of_10'])] for k,v in review['by_prompt'].items()], [28*mm,32*mm,39*mm,39*mm,36*mm])
    text('「沒有證據外PID」不能當正確率：11/15根本沒提PID，其他也可能把合法PID的規格說錯。主觀量表由Codex助理在看過證據後評閱，非使用者人工標註、獨立盲測或官方生成分數。五維各0-2：當前需求、限制／偏好、歸屬忠實度、無依據陳述、清楚程度；精確rubric及15筆理由在review.json。')
    analyses=[
        ('狗減重且維持份量','train-pet-supplies-000140','第一名SIMPLY FIT文案有相同份量、ICR與減脂維持肌肉；第二名卻是增重食品。P1直接推薦方向合理，但把文案功效當效果。P2只是重述需求，沒有完成Need→Evidence。P3把第二名的100% satisfaction guarantee移植到第一名。候選已足夠時，跨商品事實歸屬仍可能失敗。'),
        ('標準Shield、IWB與無配件','train-sports-and-outdoors-000373','第一名明列相符型號、IWB、退伍軍人公司與100%退款，但沒有退款期限。P1較能掌握目前需求；P2憑空增加within 30 days；P3被歷史腳踏車／耳罩干擾，錯稱no suitable product available。更保守的語氣不等於更正確的拒答。'),
        ('已買stamper，想看full nail patterns','beauty-train-003071','第一名是stamper/scraper，第三名標題明列42 Full Nailart Designs。P1把full patterns套到第一名而未交代工具／圖板差異。P2把商品操作說明work fast當成顧客說的話。P3亦未指出規格缺口，還無依據貶低其他候選。問題涉及商品功能角色及話語來源。'),
        ('美國製、單一成分與消化酵素','train-pet-supplies-000234','前兩名Yak Cheese文案明列Nepal且含milk/salt/lime；第三名是口腔酵素且原產Mexico。三者不能確認全部符合。P1誤稱USA；P2聲稱Unmet Requirement: None identified；P3再捏造8 oz與消化健康效果。候選缺口與生成捏造同時存在，提示詞不能憑空補回缺少的商品。'),
        ('暫定紅色、brassy與受損疑慮','beauty-train-001206','三個SKU實際為不同棕色色號，系列有紅色不代表該PID是紅色；Maybe a red是偏好而非已確認硬限制。P1把gray coverage當成改善brassing理由。P2把其他商品ColorSilk技術移到Total Color。P3捏造顧客very light hair tones like yours。合法PID不保證規格或個人化推論正確。')]
    for title,qid,analysis in analyses:
        for short,label in [('P1','直接推薦'),('P2','需求對照'),('P3','不確定性提示')]:
            analysis=analysis.replace(short,label)
        h(title+' / '+qid,2);text(analysis)
    text('直接推薦在本次五例的主觀均分最高，但樣本少且為非盲評閱，無法推論普遍最佳。需求對照的結構仍可能填入無依據規格，不確定性提示也未可靠抑制幻覺。後續可測試逐項綁定PID與規格來源，再區分已支持、相矛盾與未知；本次分析保留15筆原始回答。')
    page()
    h('附錄A：五個樣本的15筆完整原始回答')
    text('以下均為保存的模型原文，包含錯誤與未遵守指令之處。此處引用不代表認可其商品功效、規格或建議。完整原始token、prompt與evidence另見results/w09_q2。')
    plan=read('results/w09_q2/plan.json')
    outputs=[json.loads(line) for line in (ROOT/'results/w09_q2/outputs.jsonl').read_text(encoding='utf-8').splitlines()]
    for index,(title,qid,analysis) in enumerate(analyses):
        if index:page()
        h(f'A{index+1}. '+title)
        context=next(c for c in plan['contexts'] if c['qid']==qid)
        text(qid,'SmallZH')
        if qid=='train-sports-and-outdoors-000373':
            h('對話重點（完整原文隨附於plan.json）',2)
            text('歷史提到Schwinn upright bike與Howard Leight earmuffs；目前要為標準Smith & Wesson M&P Shield找IWB concealed-carry holster，無laser／light等配件，偏好美國或退伍軍人公司與退款保證。最後user回合原文：Veteran-owned is a plus, and yes, it must fit a standard Shield with no laser or light attached.','SmallZH')
        else:
            h('原始多輪對話',2);text(context['query'],'SmallZH')
        table(['檢索排名','PID','商品title'],[[e['rank'],e['pid'],e['text'].split('\n',1)[0]] for e in context['evidence']],[20*mm,38*mm,116*mm])
        for prompt in prompts['prompts']:
            row=next(o for o in outputs if o['qid']==qid and o['prompt_id']==prompt['id'])
            title=prompt_names[prompt['id']]+'：完整原始回答'
            story.append(Paragraph(escape(title),styles['H2Appendix']));md.append('### '+title+'\n')
            text(row['raw_output'],'RawAppendix')
    page()
    h('附錄B：可重現性與來源索引')
    text('正式推論載入已保存的難負例續訓模型。download.sh下載並驗證模型檔案，run.sh另存重訓結果。下載模型與重現訓練的步驟見README；實驗的原始排名、評分與生成回答依下列路徑查閱。')
    text('AI協作說明：本專案使用Codex協助程式、測試、紀錄及報告整理；Q2的主觀評語亦由Codex助理產生，已明示其非人工／盲測性質。所有執行成績來自實際保存的程序與原始輸出，未捏造結果。')
    refs=[
        '[1] 課程提供的ADL2026-HW2.pptx.pdf，pp.9-11、13-24，以及作業說明錄音。',
        '[2] BAAI official model card: https://huggingface.co/BAAI/bge-base-en-v1.5 ; pinned revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a。',
        '[3] Qwen official model card: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct ; pinned revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306。',
        '[4] Cormack, Clarke, Buettcher (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. SIGIR, 758-759. https://doi.org/10.1145/1571941.1572114 。本專案採加權變體。',
        '[5] 原始評分：results/*_graded.json；最終Dev為h5_base5_graded.json，Public為h6_h5_public_graded.json；分組比較見h6_summary.json。檔名前綴是開發期識別碼。',
        '[6] Q2：code/script/configs/w09_prompts.json、w09_generation.json；results/w09_q2/{plan,manifest,audit,review,review_summary}.json、outputs.jsonl、raw_outputs.md、evidence.md。',
        '[7] 訓練：code/script/train.py、train_base.py、train_hard_negative.py（歷史）；train_release.py（可攜重現）；configs/release_negatives.json與release_assets.json。',
        '[8] 版本及操作：README.md、requirements.txt、code/script/configs/requirements-dev.txt。訓練重現比對見results/w11_reproduction_comparison.json，乾淨目錄檢索驗證見results/w11_clean_validation.json。']
    for ref in refs:text(ref,'SmallZH')
    def decoration(canvas,doc):
        canvas.saveState();canvas.setFont('JhengHei',8);canvas.setFillColor(colors.HexColor('#62788b'))
        canvas.drawString(18*mm,287*mm,'ADL HW2 | r15922a14 | Retrieval and Generation Analysis')
        canvas.setStrokeColor(colors.HexColor('#bdcdd6'));canvas.line(18*mm,284*mm,192*mm,284*mm)
        canvas.drawString(18*mm,11*mm,'2026-10-25 | BGE-base + BM25 | Weighted RRF 5:1')
        canvas.drawRightString(192*mm,11*mm,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(output/'report.pdf'),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,
        topMargin=21*mm,bottomMargin=19*mm,title='ADL HW2 - r15922a14',author='r15922a14')
    doc.build(story,onFirstPage=decoration,onLaterPages=decoration)
    (output/'report.md').write_text('# ADL HW2 - r15922a14\n\n'+'\n'.join(md),encoding='utf-8')
    print(json.dumps({'pdf':str(output/'report.pdf'),'markdown':str(output/'report.md'),'bytes':(output/'report.pdf').stat().st_size}))


if __name__=='__main__':main()
