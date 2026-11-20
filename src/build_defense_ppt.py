# -*- coding: utf-8 -*-
"""生成答辩 PPT（白底 + 微软雅黑 + 4 张核心图），输出 结题材料/答辩PPT.pptx。

设计约束（用户偏好）：微软雅黑、白底、图文并茂；只保留 4 张核心图：
  1) 三架构对比（rmse_comparison + bbbp_comparison）
  2) 特征工程收益（feature_gain）
  3) 融合边界（scaffold_multiseed）
  4) 解释一致性（explanation_consistency）
"""
import os

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt
from lxml import etree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "reports", "figures")
OUT_DIR = os.path.join(ROOT, "结题材料")
OUT = os.path.join(OUT_DIR, "答辩PPT.pptx")

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
BLUE = RGBColor(0x2E, 0x6D, 0xA4)
ORANGE = RGBColor(0xF5, 0x85, 0x18)
GRAY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xF2, 0xF5, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Microsoft YaHei"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def _set_run_font(run, size=16, bold=False, color=GRAY, font=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    # 同时写入东亚字体，避免中文回退到宋体
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find("{http://schemas.openxmlformats.org/drawingml/2006/main}ea")
    if ea is None:
        ea = etree.SubElement(rPr, "{http://schemas.openxmlformats.org/drawingml/2006/main}ea")
    ea.set("typeface", font)


def add_text(slide, x, y, w, h, text, size=16, bold=False, color=GRAY,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line_spacing=1.1):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = line
        _set_run_font(r, size=size, bold=bold, color=color)
    return box


def add_bullets(slide, x, y, w, h, items, size=15, color=GRAY, gap=6, bullet_color=BLUE):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.15
        p.space_after = Pt(gap)
        r1 = p.add_run()
        r1.text = "•  "
        _set_run_font(r1, size=size, bold=True, color=bullet_color)
        r2 = p.add_run()
        r2.text = item
        _set_run_font(r2, size=size, color=color)
    return box


def add_title(slide, kicker, title, subtitle=None):
    add_text(slide, 0.75, 0.42, 11.9, 0.3, kicker, size=11, bold=True, color=ORANGE)
    add_text(slide, 0.75, 0.72, 11.9, 0.62, title, size=27, bold=True, color=NAVY)
    # 细线
    line = slide.shapes.add_shape(1, Inches(0.75), Inches(1.36), Inches(11.85), Inches(0.035))
    line.fill.solid()
    line.fill.fore_color.rgb = NAVY
    line.line.fill.background()
    if subtitle:
        add_text(slide, 0.75, 1.48, 11.9, 0.42, subtitle, size=13, color=GRAY)


def add_footer(slide, idx, total, label="工程实践结题答辩 · 基于图神经网络的分子性质预测"):
    add_text(slide, 0.75, 7.12, 9.5, 0.25, label, size=9, color=RGBColor(0x8C, 0x8C, 0x8C))
    add_text(slide, 11.7, 7.12, 0.9, 0.25, f"{idx}/{total}", size=9, color=RGBColor(0x8C, 0x8C, 0x8C),
             align=PP_ALIGN.RIGHT)


def add_picture_fit(slide, path, x, y, w, h, border=True):
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    px = x + (w - pw) / 2
    py = y + (h - ph) / 2
    pic = slide.shapes.add_picture(path, Inches(px), Inches(py), Inches(pw), Inches(ph))
    if border:
        pic.line.color.rgb = RGBColor(0xD9, 0xD9, 0xD9)
        pic.line.width = Pt(0.75)
    return pic


def add_card(slide, x, y, w, h, title, body, accent=BLUE, fill=LIGHT, title_size=14, body_size=12):
    box = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.color.rgb = RGBColor(0xE3, 0xE8, 0xEE)
    box.line.width = Pt(0.75)
    # 顶部色条
    bar = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(0.06))
    bar.fill.solid()
    bar.fill.fore_color.rgb = accent
    bar.line.fill.background()
    add_text(slide, x + 0.16, y + 0.16, w - 0.3, 0.3, title, size=title_size, bold=True, color=NAVY)
    add_text(slide, x + 0.16, y + 0.52, w - 0.3, h - 0.6, body, size=body_size, color=GRAY, line_spacing=1.12)


def add_kpi(slide, x, y, w, h, value, label, accent=BLUE):
    box = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = WHITE
    box.line.color.rgb = RGBColor(0xDD, 0xE3, 0xEA)
    box.line.width = Pt(0.75)
    add_text(slide, x, y + 0.12, w, 0.5, value, size=24, bold=True, color=accent, align=PP_ALIGN.CENTER)
    add_text(slide, x, y + 0.62, w, 0.3, label, size=10, color=GRAY, align=PP_ALIGN.CENTER)


def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def compact_table(shape, header_h=0.42, row_h=0.36):
    n = len(shape.table.rows)
    for i, row in enumerate(shape.table.rows):
        row.height = Inches(header_h if i == 0 else row_h)
    shape.height = Inches(header_h + row_h * (n - 1))


TOTAL = 12

# ------------------------------ 1 封面 ------------------------------
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = WHITE
bg.line.fill.background()

add_text(s, 0.9, 1.15, 11.5, 0.35, "中国科学技术大学软件学院 · 工程实践", size=13, bold=True, color=ORANGE)
add_text(s, 0.9, 1.62, 11.6, 1.1, "基于图神经网络的\n分子性质预测", size=40, bold=True, color=NAVY, line_spacing=1.05)
add_text(s, 0.9, 3.02, 11.5, 0.75,
         "GCN / GAT / GraphSAGE 的架构对比、特征工程、异质集成与解释一致性",
         size=17, color=GRAY)
line = s.shapes.add_shape(1, Inches(0.9), Inches(3.92), Inches(1.6), Inches(0.05))
line.fill.solid()
line.fill.fore_color.rgb = ORANGE
line.line.fill.background()

add_text(s, 0.9, 4.18, 5.6, 0.35, "答辩人：李乐", size=14, bold=True, color=NAVY)
add_text(s, 0.9, 4.56, 5.6, 0.35, "2026 年 11 月 20 日", size=13, color=GRAY)

for i, (v, lab, c) in enumerate([
    ("4", "MoleculeNet 数据集", BLUE),
    ("310", "checkpoint 总数（主实验 144）", BLUE),
    ("4", "训练种子复验", ORANGE),
    ("120", "解释一致性分子", ORANGE),
]):
    add_kpi(s, 7.15 + (i % 2) * 2.55, 4.15 + (i // 2) * 1.25, 2.3, 0.98, v, lab, accent=c)

add_text(s, 0.9, 6.85, 11.6, 0.3,
         "代码仓库：gnn-molecular-property-prediction　|　冻结版本 tag v1.0-freeze-144ckpt　|　复验分支 exp/scaffold-multiseed",
         size=9, color=RGBColor(0x99, 0x99, 0x99))
add_notes(s, "各位老师好，我是李乐，本次工程实践的题目是《基于图神经网络的分子性质预测》。"
             "我会用 10 分钟左右汇报四件事：三个架构的对比、特征工程的收益、融合到底在什么条件下有效，以及模型解释是否可信。"
             "所有数字都来自仓库里的 JSON 结果文件，可以逐项复核。")

# ------------------------------ 2 任务与方案 ------------------------------
s = prs.slides.add_slide(BLANK)
add_title(s, "01 / 任务与方案", "分子是图：把性质预测做成一个可控的小型 GNN 实验",
          "4 个 MoleculeNet 数据集 · 3 种架构 · 2 种划分 · 从训练到 Web 的完整闭环")

# 左：任务表
add_text(s, 0.75, 1.78, 6.4, 0.3, "数据集与任务", size=15, bold=True, color=NAVY)
rows = [
    ("数据集", "任务", "样本", "指标"),
    ("ESOL", "水溶解度 logS", "1128", "RMSE / R²"),
    ("FreeSolv", "水合自由能", "642", "RMSE / R²"),
    ("BBBP", "血脑屏障渗透", "2039", "AUC / F1"),
    ("Clintox", "临床毒性", "1480", "AUC / F1"),
]
tbl_shape = s.shapes.add_table(len(rows), 4, Inches(0.75), Inches(2.12), Inches(6.1), Inches(2.2))
tbl = tbl_shape.table
for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = tbl.cell(r, c)
        cell.text = val
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY if r == 0 else (WHITE if r % 2 else LIGHT)
        cell.margin_left = Inches(0.08)
        cell.margin_right = Inches(0.08)
        cell.margin_top = Inches(0.03)
        cell.margin_bottom = Inches(0.03)
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
            for run in p.runs:
                _set_run_font(run, size=11, bold=(r == 0), color=WHITE if r == 0 else GRAY)

compact_table(tbl_shape)
# 左下：两种划分
add_text(s, 0.75, 4.55, 6.4, 0.3, "两种数据划分", size=15, bold=True, color=NAVY)
add_bullets(s, 0.75, 4.9, 6.1, 1.5, [
    "随机划分 8:1:1（seed=42）：与文献可比的主口径",
    "Murcko 骨架划分：同一骨架只进一个集合，检验分布外泛化",
    "骨架划分下 FreeSolv 的 val 仅 64 条 —— 模型选择本身成为变量",
], size=12)

# 右：流程图（5 步）
add_text(s, 7.15, 1.78, 5.4, 0.3, "实验链路", size=15, bold=True, color=NAVY)
steps = [
    ("① 特征工程", "BASE 9 维 → RDKit 27 维 → +键特征 → +MACCS"),
    ("② 架构对比", "GCN / GAT / GraphSAGE，统一 3 层 + 64 隐维"),
    ("③ 融合研究", "同构 / 同架构×特征档 / 异质 all9，4 种子复验"),
    ("④ 解释可信度", "9 配置 × 40 分子一致性 + 官能团聚合"),
    ("⑤ Web 演示", "单条预测 / 批量 CSV / 降级 / 版本清单"),
]
y = 2.15
for i, (t, d) in enumerate(steps):
    box = s.shapes.add_shape(1, Inches(7.15), Inches(y), Inches(5.35), Inches(0.82))
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT if i % 2 == 0 else WHITE
    box.line.color.rgb = RGBColor(0xDD, 0xE3, 0xEA)
    box.line.width = Pt(0.75)
    add_text(s, 7.28, y + 0.08, 1.45, 0.3, t, size=11, bold=True, color=BLUE)
    add_text(s, 8.6, y + 0.12, 3.8, 0.6, d, size=10.5, color=GRAY)
    y += 0.92

add_text(s, 0.75, 6.62, 11.9, 0.35,
         "统一评估纪律：只看验证集选模型，test 只评估一次；分类 AUC 用 softmax 正类概率口径。",
         size=11, bold=True, color=ORANGE)
add_footer(s, 2, TOTAL)
add_notes(s, "先交代任务：四个 MoleculeNet 数据集，回归两个、分类两个；三种 GNN 架构结构完全一致，只换卷积算子，保证对比公平。"
             "两种划分是关键：随机划分看文献可比性，骨架划分看真正的分布外泛化。"
             "整个链路从特征、架构、融合、解释一直到 Web，全部有脚本和 JSON 记录。")

# ------------------------------ 3 核心图 1：三架构 ------------------------------
s = prs.slides.add_slide(BLANK)
add_title(s, "02 / 核心图 1", "没有单一架构全面胜出：任务特性决定归纳偏置",
          "左：回归 RMSE（越低越好）　右：BBBP 分类指标（越高越好）")
add_picture_fit(s, os.path.join(FIG, "rmse_comparison.png"), 0.65, 1.8, 6.0, 4.0)
add_picture_fit(s, os.path.join(FIG, "bbbp_comparison.png"), 6.75, 1.8, 6.0, 4.0)
add_bullets(s, 0.75, 5.95, 11.8, 1.0, [
    "ESOL：GAT 最优（RMSE 0.748 / R² 0.867）——注意力擅长小分子局部精细结构",
    "FreeSolv：GraphSAGE 最优（RMSE 1.384）——均值聚合在少样本、宽分布下更稳；BBBP 同样 SAGE 领先（AUC 0.876）",
    "Clintox 正类仅 7.6%，三种架构 AUC 0.64–0.72 —— 后续涉及 Clintox 的结论一律谨慎解读",
], size=10.5, gap=3)
add_footer(s, 3, TOTAL)
add_notes(s, "这张图回答第一个问题：换架构有没有用。结论是没有全能选手。"
             "ESOL 上 GAT 最好，因为分子小、局部环境重要；FreeSolv 样本只有 642 条、标签跨度大，GraphSAGE 的均值聚合更稳。"
             "BBBP 也是 SAGE 领先。Clintox 正样本太少，AUC 波动大，所以后面不把它当作主结论依据。")

# ------------------------------ 4 核心图 2：特征工程 ------------------------------
s = prs.slides.add_slide(BLANK)
add_title(s, "03 / 核心图 2", "特征工程的收益远大于换架构",
          "基线 = 9 维原子特征下最强架构；标注为最优特征档配置")
add_picture_fit(s, os.path.join(FIG, "feature_gain.png"), 0.9, 1.75, 8.0, 4.55)
add_card(s, 9.25, 1.95, 3.35, 1.45, "FreeSolv", "RMSE 1.384 → 1.008\n下降 27.1%（GCN_featc）", accent=ORANGE)
add_card(s, 9.25, 3.55, 3.35, 1.45, "Clintox", "AUC 0.724 → 0.898\n提升 17.4 点（GAT_feata）", accent=BLUE)
add_card(s, 9.25, 5.15, 3.35, 1.45, "结论", "RDKit 原子特征是主贡献；\nMACCS 对回归有效、对分类有限", accent=NAVY, body_size=11)
add_bullets(s, 0.9, 6.4, 11.6, 0.6, [
    "特征档之间犯的错不同 —— 这是后面\"跨特征档集成\"有效的物理基础，也是比\"换架构\"更值钱的多样性来源",
], size=11)
add_footer(s, 4, TOTAL)
add_notes(s, "第二个问题：加特征和换架构哪个更值钱？答案是加特征。"
             "FreeSolv 的 RMSE 从 1.384 降到 1.008，降了 27%；Clintox 的 AUC 从 0.724 提到 0.898。"
             "更关键的是：不同特征档犯的错不一样，这为后面的跨特征档集成提供了多样性。")

# ------------------------------ 5 融合诊断 ------------------------------
s = prs.slides.add_slide(BLANK)
add_title(s, "04 / 融合研究（一）", "先诊断：项目原来的\"集成\"其实是伪融合",
          "144 个 checkpoint 的 test 残差相关矩阵（越低越互补）")

rows = [
    ("数据集", "同配置跨种子", "同架构跨特征档", "跨架构"),
    ("ESOL", "0.918", "0.570", "0.606"),
    ("FreeSolv", "0.914", "0.712", "0.750"),
    ("BBBP", "0.952", "0.844", "0.871"),
    ("Clintox", "0.829", "0.740", "0.752"),
]
tbl_shape = s.shapes.add_table(len(rows), 4, Inches(0.75), Inches(2.0), Inches(5.9), Inches(2.4))
tbl = tbl_shape.table
for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = tbl.cell(r, c)
        cell.text = val
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY if r == 0 else (WHITE if r % 2 else LIGHT)
        cell.margin_left = Inches(0.08)
        cell.margin_top = Inches(0.03)
        cell.margin_bottom = Inches(0.03)
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
            for run in p.runs:
                _set_run_font(run, size=11, bold=(r == 0), color=WHITE if r == 0 else GRAY)

compact_table(tbl_shape)
add_bullets(s, 0.75, 4.35, 5.9, 1.4, [
    "同配置跨种子相关 0.91–0.95：成员几乎犯同样的错，平均不产生增益",
    "同架构跨特征档 0.57–0.84：特征档改变了模型看到的输入，误差开始分化",
    "跨架构 0.61–0.87：归纳偏置不同，但相关仍高于特征档（ESOL 例外）",
], size=11)

add_card(s, 7.15, 1.95, 5.35, 1.3, "伪融合证据", "Clintox GCN 同构集成 AUC 0.7087\n低于最好单成员 0.714 → 平均反而伤害", accent=ORANGE)
add_card(s, 7.15, 3.45, 5.35, 1.3, "有效方案", "9 配置 × 4 种子等权平均（all9）\n随机划分四数据集全部为正增益", accent=BLUE)
add_card(s, 7.15, 4.95, 5.35, 1.3, "但随机划分增益只有", "+0.6 / +0.8 / +2.1 / +0.4 点\n因为 train/test 共享骨架，单模型已\"见过\"相似分子", accent=NAVY, body_size=11)

add_footer(s, 5, TOTAL)
add_notes(s, "做融合之前先诊断。我们把 144 个模型的 test 残差算相关：同一配置换种子，相关 0.92 以上，说明它们犯同样的错，平均没有意义。"
             "跨特征档相关降到 0.57 到 0.84，跨架构 0.61 到 0.87，多样性才出现。"
             "所以项目原来的同构集成是伪融合；真正有效的是跨配置等权。但随机划分下增益很小，只有 0.6 到 2 个点，这促使我们去看骨架划分。")

# ------------------------------ 6 核心图 3：融合边界 ------------------------------
s = prs.slides.add_slide(BLANK)
add_title(s, "05 / 核心图 3", "骨架划分 + 4 种子复验：增益真实存在，但不是\"架构越多越好\"",
          "均值 ± 标准差；右侧为 all9（圆点）与同架构 12 成员（方块）相对最强单模型的 95% bootstrap CI")
add_picture_fit(s, os.path.join(FIG, "scaffold_multiseed.png"), 0.55, 1.78, 12.2, 3.95)
add_bullets(s, 0.75, 5.95, 11.9, 1.0, [
    "ESOL +3.0 点 CI[+0.2, +6.5]、FreeSolv +21.9 点 CI[+9.9, +44.0]、BBBP +7.9 点 CI[+1.5, +14.5]：相对部署基线的 CI 均不含 0，可称\"稳定提升\"（ESOL/BBBP 为边界显著）",
    "但主要增益来自特征档：同架构 3 档 × 4 种子（12 成员）在 FreeSolv/BBBP 已拿到 +20.1/+8.1 点；BBBP 上 all9 反而比 12 成员低 0.2 点，CI 含 0 → 无提升",
    "BBBP 不提高上限：事后最优单配置 GCN_feata 为 0.8339，高于所有集成 → 集成的价值是降方差、免调模，不是刷新最好成绩",
], size=10.5)
add_footer(s, 6, TOTAL)
add_notes(s, "这是本次复验的核心。单种子时我们以为 all9 全面领先；补到 4 个种子后，方向仍然成立，但结构变了。"
             "相对部署基线，三个数据集都是正的、CI 不含 0；但 ESOL 和 BBBP 把基线换成每个种子各自的最优单模型，就有一半左右的配对 CI 含 0，所以幅度不能外推。"
             "更重要的发现是：同架构、3 个特征档、4 个种子这 12 个成员已经拿到大部分收益；跨架构在 ESOL 再加 1.4 点，FreeSolv 只有趋势性，BBBP 甚至略微为负。所以\"架构越多越好\"不成立，覆盖特征档比堆架构更重要。")

# ------------------------------ 7 负结果 ------------------------------
s = prs.slides.add_slide(BLANK)
add_title(s, "06 / 融合研究（二）", "三种\"更高级\"的融合全部失败，但失败原因可归因",
          "保留负结果，是为了说明融合的边界在哪里")

add_card(s, 0.75, 1.85, 3.85, 2.45, "① 架构级融合 HybridAgg",
         "层内 GCN+GAT+SAGE 可学习加权\nESOL R² 0.855（对照 0.921）\nBBBP AUC 0.921（对照 0.929）\n三层权重全部退化到 0.33/0.33/0.33",
         accent=ORANGE, body_size=11)
add_card(s, 4.75, 1.85, 3.85, 2.45, "② NNLS 静态加权",
         "在 val 上拟合 9 个成员权重\n仅 ESOL 有效（0.29/0.15/0.56）\nFreeSolv / Clintox 权重坍缩为 1.0/0/0\n小验证集上拟合权重必然过拟合",
         accent=BLUE, body_size=11)
add_card(s, 8.75, 1.85, 3.85, 2.45, "③ 知识蒸馏",
         "教师 12 模型 → 学生单模型\nESOL 学生 0.675 vs 教师 0.591\nBBBP AUC 0.9125 vs 教师 0.9183\n单模型容量表达不了集成边界",
         accent=NAVY, body_size=11)

add_text(s, 0.75, 4.65, 11.9, 0.35, "共同机理", size=15, bold=True, color=ORANGE)
add_bullets(s, 0.75, 5.05, 11.9, 0.9, [
    "在同一个计算图里做融合 → 分支共享输入与梯度 → 表示趋同 → 多样性被摧毁；跨模型、独立训练轨迹的等权才是有效层级",
], size=11)
add_card(s, 0.75, 5.85, 11.85, 0.85, "方法学价值", "负结果不是失败，而是给出边界：融合必须发生在模型之间，不能发生在计算图内部。",
         accent=BLUE, body_size=11, title_size=12)
add_footer(s, 7, TOTAL)
add_notes(s, "我们也试了三种看起来更高级的方案，全部失败。"
             "HybridAgg 在层内并联三个算子，结果三层权重都退化成三分之一，等于等权，但性能还不如模型级集成；加残差和 LayerNorm 的 v2 更差，ESOL R² 直接崩到 0.41。"
             "NNLS 在验证集上拟合权重，小验证集会坍缩成单专家；蒸馏把 12 个模型压成一个，容量不够。"
             "共同原因：在同一个计算图里融合，梯度共享会把多样性抹掉。")

# ------------------------------ 8 核心图 4：解释一致性 ------------------------------
s = prs.slides.add_slide(BLANK)
add_title(s, "07 / 核心图 4", "解释一致性：关键原子\"集合\"高度一致，\"排序\"只有中等一致",
          "每数据集 40 个代表性分子 × 9 配置 × 36 对两两比较；阈值在看到结果前预设")
add_picture_fit(s, os.path.join(FIG, "explanation_consistency.png"), 0.75, 1.75, 8.55, 4.7)

add_card(s, 9.55, 1.85, 3.05, 1.35, "ESOL", "Top-10 原子 Jaccard 0.758\nSpearman 0.358 → 高", accent=BLUE, body_size=11)
add_card(s, 9.55, 3.35, 3.05, 1.35, "FreeSolv", "Top-10 原子 Jaccard 0.963\nSpearman 0.348 → 高", accent=ORANGE, body_size=11)
add_card(s, 9.55, 4.85, 3.05, 1.35, "BBBP", "Top-10 原子 Jaccard 0.489\nSpearman 0.365 → 中", accent=NAVY, body_size=11)

add_bullets(s, 0.75, 6.55, 11.9, 0.55, [
    "结论：\"哪些原子重要\"共识高，\"谁排第一\"分歧大 → Web 必须展示一致性等级，不能把单一模型排序当化学真值",
], size=11)
add_footer(s, 8, TOTAL)
add_notes(s, "第四个问题是解释可不可信。"
             "同一分子换 9 个配置跑 GNNExplainer，ESOL 和 FreeSolv 的 Top-10 原子集合重合度分别是 0.76 和 0.96，判为高；BBBP 只有 0.49，判为中。"
             "但排序相关只有 0.35 左右，说明\"哪些原子重要\"有共识，\"谁排第一\"各说各话。所以 Web 上给一致性等级，而不是把一张热图当唯一答案。")

# ------------------------------ 9 官能团 ------------------------------
s = prs.slides.add_slide(BLANK)
add_title(s, "08 / 官能团解释", "把原子重要性聚合到化学语言，但三层职责必须分开",
          "同一个分子：模型归因、化学识别、自然语言复述，不能混为一谈")

# 三层流程图
layers = [
    ("模型归因", "GNNExplainer\n9 配置 seed42 掩码平均", "给出原子/键重要性数值", BLUE, 0.75),
    ("化学识别", "RDKit SMARTS + 环感知", "判断官能团存在性与成员原子", ORANGE, 4.65),
    ("自然语言", "规则模板 / DeepSeek LLM", "只复述证据，不得决定谁重要", NAVY, 8.55),
]
for title, src, role, color, x in layers:
    box = s.shapes.add_shape(1, Inches(x), Inches(1.9), Inches(3.6), Inches(1.55))
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = color
    box.line.width = Pt(1.2)
    add_text(s, x + 0.15, 2.0, 3.3, 0.3, title, size=14, bold=True, color=color)
    add_text(s, x + 0.15, 2.35, 3.3, 0.5, src, size=10.5, bold=True, color=GRAY)
    add_text(s, x + 0.15, 2.92, 3.3, 0.4, role, size=10, color=GRAY)

# 示例
add_text(s, 0.75, 3.75, 6.0, 0.3, "示例：ESOL #963（y=−6.02，pred=−5.07）", size=14, bold=True, color=NAVY)
add_text(s, 0.75, 4.12, 6.1, 0.55, "Fc1cccc(F)c1C(=O)NC(=O)Nc2ccc(Cl)cc2", size=10.5, color=GRAY)
rows = [
    ("Top-3 官能团", "原子数", "重要性均值", "份额"),
    ("芳香环", "12", "0.451", "0.678"),
    ("卤素 (Cl)", "1", "0.354", "0.044"),
    ("酰胺", "6", "0.316", "0.237"),
]
tbl_shape = s.shapes.add_table(len(rows), 4, Inches(0.75), Inches(4.72), Inches(6.2), Inches(1.55))
tbl = tbl_shape.table
for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = tbl.cell(r, c)
        cell.text = val
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY if r == 0 else (WHITE if r % 2 else LIGHT)
        cell.margin_left = Inches(0.08)
        cell.margin_top = Inches(0.03)
        cell.margin_bottom = Inches(0.03)
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
            for run in p.runs:
                _set_run_font(run, size=10.5, bold=(r == 0), color=WHITE if r == 0 else GRAY)

compact_table(tbl_shape, header_h=0.38, row_h=0.33)
add_bullets(s, 7.3, 3.78, 5.25, 2.55, [
    "唯一原子分配：羧基 > 酯基 > 酰胺 > 磺酰基 > …… > 卤素 > 芳香环，避免羟基被羧基重复计数",
    "芳香环允许与取代基在连接原子处重叠；份额 = 组内重要性 / 全分子重要性",
    "覆盖有限：FreeSolv 平均仅 1.2 个可识别官能团，7/40 零检出；零检出时回退原子级解释，不编造",
], size=9.5, gap=4)
add_footer(s, 9, TOTAL)
add_notes(s, "原子级解释对化学老师不够直观，所以我们聚合到官能团。"
             "但必须把三层分开：重要性数值只能来自 GNNExplainer；RDKit 只负责识别基团是否存在；LLM 只能复述，不能自己决定哪个基团重要。"
             "这个分子 Top-3 是芳香环、氯、酰胺。局限也要讲清楚：FreeSolv 平均只有 1.2 个可识别基团，7 个分子零检出，这时就回退到原子级，不能编。")

# ------------------------------ 10 Web ------------------------------
s = prs.slides.add_slide(BLANK)
add_title(s, "09 / Web 演示系统", "从一条 SMILES 到一批分子：可用、可降级、可追溯",
          "Flask + 原生前端；五种分析方式 + 批量 CSV + 导出 + 规则模板兜底")

add_card(s, 0.75, 1.8, 3.85, 2.15, "单条预测", "SMILES → 预测值 + 单位\nGNNExplainer 结构着色\nMC-Dropout / 同构 / 异质 all9\n一致性等级 + Top-3 官能团", accent=BLUE, body_size=11)
add_card(s, 4.75, 1.8, 3.85, 2.15, "批量预测", "CSV（smiles/id，≤200 行）\n逐行预测、逐行错误\n一键导出 CSV（UTF-8 BOM）\n非法 SMILES 不中断整批", accent=ORANGE, body_size=11)
add_card(s, 8.75, 1.8, 3.85, 2.15, "降级与追溯", "无 Key / 超时 / 断网 → 规则模板\n页脚显示模型 tag 与成员数\n空输入/超长/缺模型 → 明确 400/503", accent=NAVY, body_size=11)

add_text(s, 0.75, 4.22, 11.9, 0.3, "接口与验收", size=14, bold=True, color=NAVY)
add_bullets(s, 0.75, 4.58, 11.9, 1.1, [
    "/api/predict、/api/batch、/api/export、/api/explain、/api/meta 五个端点覆盖单条、批量、导出、解释与版本",
    "35 项 unittest + 17 项 Web 端点冒烟全部通过；从干净环境按 README 可启动",
    "示例批量文件：示例批量分子.csv（含 1 条非法 SMILES，用于演示逐行错误提示）",
], size=10.5, gap=3)

add_card(s, 0.75, 5.9, 11.85, 0.8, "现场演示顺序",
         "启动 python src/app.py → 打开 http://127.0.0.1:5000 → 输入阿司匹林 SMILES → 异质集成 + 官能团 → 切到批量的 示例批量分子.csv → 观察错误行 → 导出 CSV",
         accent=ORANGE, body_size=10, title_size=12)
add_footer(s, 10, TOTAL)
add_notes(s, "Web 不是摆设，答辩时我会现场跑。"
             "单条能做五种分析；批量支持 200 行 CSV，错误逐行返回，结果能导出；没有 API Key 或断网时自动降级成规则模板，不会崩。"
             "页脚会显示模型版本 tag 和 checkpoint 数量，保证演示结果和报告对得上。35 项单元测试和 17 项端点测试都通过。")

# ------------------------------ 11 局限与展望 ------------------------------
s = prs.slides.add_slide(BLANK)
add_title(s, "10 / 局限与展望", "把不确定的地方说清楚，比多报一个点更有价值",
          "结论的边界 = 适用范围；不能外推的地方逐条标注")

add_text(s, 0.75, 1.95, 5.85, 0.3, "必须承认的局限", size=15, bold=True, color=ORANGE)
add_bullets(s, 0.75, 2.32, 5.85, 3.3, [
    "Clintox 骨架 test 仅 5 个正样本 → 多种子只覆盖 ESOL/FreeSolv/BBBP",
    "FreeSolv 的 MC-Dropout 区间覆盖率低于名义 95% → 不能当严格置信区间",
    "解释一致性只测了随机划分的 seed42 主模型；GNNExplainer 60 epochs 仍有随机性",
    "官能团 SMARTS 覆盖有限：FreeSolv 7/40 零检出，Top-3 常不满额",
    "BBBP 上集成不提高上限：GCN_feata 0.8339 > all9 0.8182",
    "CPU 训练可行但不适合大规模搜索：BBBP-GAT 约 15 分钟/模型",
], size=11, gap=5)

add_text(s, 7.0, 1.95, 5.6, 0.3, "下一步", size=15, bold=True, color=BLUE)
add_bullets(s, 7.0, 2.32, 5.6, 3.3, [
    "架构扩展 GIN/MPNN：先验证\"架构多样性\"的边际收益是否真的存在",
    "不确定性校准：温度缩放 / conformal prediction，修正 FreeSolv 覆盖率",
    "骨架级 OOD 判别 + 拒答：把\"陌生骨架\"识别出来再路由",
    "解释评估：加入 deletion/insertion faithfulness，与 Jaccard 联合评估",
    "部署：ONNX 导出与流式批量，降低 9 模型 CPU 推理成本",
], size=11, gap=5)
add_card(s, 0.75, 5.85, 11.85, 0.85, "结论边界（写进报告的措辞纪律）",
         "CI 不含 0 才写\"稳定提升\"；含 0 一律写\"趋势性提升\"；点估计 ≤0 且 CI 含 0 写\"无提升\"——本报告已按此规则逐条执行。",
         accent=ORANGE, body_size=11, title_size=12)
add_footer(s, 11, TOTAL)
add_notes(s, "这一页专门讲局限。"
             "Clintox 骨架划分正样本只有 5 个，所以多种子只做三个数据集；FreeSolv 的 MC-Dropout 区间覆盖率不足，我们不当严格置信区间用；解释一致性只覆盖随机划分的 seed42 模型；官能团覆盖也有限。"
             "BBBP 上集成不提高上限，这一点必须说，否则就是过度声明。下一步优先做不确定性的校准和真正的 OOD 拒答。")

# ------------------------------ 12 结论 ------------------------------
s = prs.slides.add_slide(BLANK)
add_title(s, "11 / 结论", "四句话总结", "所有数字可在 reports/*.json 与 src/*.py 中逐项复核")

items = [
    ("① 架构", "没有全能选手：ESOL 看 GAT，FreeSolv/BBBP 看 GraphSAGE；Clintox 样本太少，不做主结论。", BLUE),
    ("② 特征", "特征工程收益大于换架构：FreeSolv RMSE −27.1%，Clintox AUC +17.4 点。", ORANGE),
    ("③ 融合", "有效层级是模型级；骨架划分增益主要来自特征档多样性。all9 相对部署基线 +3.0/+21.9/+7.9 点（CI 不含 0），但 BBBP 不如 12 成员，也不如事后最优单模型。", NAVY),
    ("④ 解释", "Top-10 原子集合 Jaccard 0.49–0.96，但排序相关仅约 0.35；Web 展示一致性等级，官能团解释三层分离。", BLUE),
]
y = 1.98
for tag, text, color in items:
    add_text(s, 0.75, y, 1.35, 0.4, tag, size=15, bold=True, color=color)
    add_text(s, 2.15, y - 0.02, 10.4, 0.8, text, size=12.5, color=GRAY, line_spacing=1.1)
    y += 0.78

add_text(s, 0.75, 5.25, 11.9, 0.4,
         "交付：12 页答辩 PPT · 3–5 分钟演示脚本 · 35 项单元测试 · 所有结论可溯源到 reports/*.json",
         size=12, bold=True, color=ORANGE)
add_text(s, 0.75, 6.25, 11.9, 0.45, "谢谢各位老师，欢迎提问。", size=18, bold=True, color=NAVY)
add_text(s, 0.75, 6.72, 11.9, 0.3,
         "代码与数据：gnn-molecular-property-prediction　|　报告：reports/项目报告.md　|　复现：结题材料/交付与复现说明.md",
         size=9, color=RGBColor(0x99, 0x99, 0x99))
add_footer(s, 12, TOTAL)
add_notes(s, "最后四句话。"
             "第一，架构没有全能选手，要按任务选。第二，特征工程比换架构更值钱。"
             "第三，融合的有效层级是模型级，骨架划分下增益主要来自特征档多样性；all9 相对部署基线是稳定提升，但 BBBP 上不提高上限。"
             "第四，解释的集合一致性高、排序一致性中等，所以 Web 给等级、官能团分三层。所有数字都在仓库 JSON 里，欢迎老师抽查。")

os.makedirs(OUT_DIR, exist_ok=True)
prs.save(OUT)
print("PPT 已保存 ->", OUT)
