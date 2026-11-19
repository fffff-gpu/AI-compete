# -*- coding: utf-8 -*-
"""企业赛答辩 PPT（1.5）生成脚本。

    python build_deck_15.py

用 python-pptx 手排 16:9，无模板依赖；每页只表达一个判断，正文不超过 6 条。
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls
from pptx.util import Emu, Inches, Pt

OUT = Path(__file__).with_name("答辩PPT-1.5.pptx")

INK = RGBColor(0x1F, 0x29, 0x37)
MUTED = RGBColor(0x6B, 0x72, 0x80)
ACCENT = RGBColor(0x0F, 0x76, 0x6E)
AMBER = RGBColor(0xB4, 0x53, 0x09)
RED = RGBColor(0xB4, 0x1C, 0x1C)
PANEL = RGBColor(0xF1, 0xF5, 0xF9)
PANEL2 = RGBColor(0xEC, 0xFD, 0xF5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0xCB, 0xD5, 0xE1)
FONT = "Microsoft YaHei"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
PAGES = []


def style_run(run, size=14, bold=False, color=INK):
    f = run.font
    f.name = FONT
    f.size = Pt(size)
    f.bold = bold
    f.color.rgb = color
    rpr = run._r.get_or_add_rPr()
    rpr.append(parse_xml(f'<a:ea {nsdecls("a")} typeface="{FONT}"/>'))


def textbox(slide, x, y, w, h, lines, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
            space_after=6, line_spacing=1.22):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(0.02))
    for i, (text, opt) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = opt.get("align", align)
        p.space_after = Pt(opt.get("space_after", space_after))
        p.line_spacing = line_spacing
        run = p.add_run()
        run.text = text
        style_run(run, opt.get("size", 14), opt.get("bold", False), opt.get("color", INK))
    return box


def rect(slide, x, y, w, h, fill=WHITE, edge=None, radius=False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    if radius:
        shape.adjustments[0] = 0.06
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    if edge:
        shape.line.color.rgb = edge
        shape.line.width = Pt(0.9)
    else:
        shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def line(slide, x1, y1, x2, y2, color=LINE, width=1.1, dash=False):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                      Inches(x2), Inches(y2))
    conn.line.color.rgb = color
    conn.line.width = Pt(width)
    if dash:
        from pptx.enum.dml import MSO_LINE_DASH_STYLE
        conn.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    return conn


def new_slide(kicker=None, heading=None, sub=None, footer_text=None):
    s = prs.slides.add_slide(BLANK)
    rect(s, 0, 0, 13.333, 7.5, WHITE)
    if kicker:
        textbox(s, 0.62, 0.30, 6.0, 0.32, [(kicker, dict(size=11, bold=True, color=ACCENT))])
    if heading:
        textbox(s, 0.62, 0.58, 12.1, 0.62, [(heading, dict(size=26, bold=True))])
    if sub:
        textbox(s, 0.64, 1.20, 12.1, 0.42, [(sub, dict(size=12, color=MUTED))])
    rect(s, 0.62, 1.62, 0.62, 0.045, ACCENT)
    if footer_text:
        textbox(s, 0.62, 7.05, 10.5, 0.28, [(footer_text, dict(size=9, color=MUTED))])
    textbox(s, 11.4, 7.05, 1.35, 0.28, [("%d / 16" % len(PAGES), dict(size=9, color=MUTED,
                                                                          align=PP_ALIGN.RIGHT))])
    PAGES.append(s)
    return s


def bullets(slide, items, x=0.72, y=1.95, w=12.0, size=14, gap=0.52, color=INK):
    for i, item in enumerate(items):
        yy = y + i * gap
        rect(slide, x, yy + 0.09, 0.075, 0.075, ACCENT)
        textbox(slide, x + 0.20, yy - 0.04, w - 0.20, gap, [(item, dict(size=size, color=color))])


def table(slide, rows, x=0.72, y=1.95, w=11.9, col_ratios=None, size=11.5,
          header_fill=PANEL, row_h=0.40, header_h=0.42):
    cols = len(rows[0])
    ratios = col_ratios or [1.0 / cols] * cols
    total = sum(ratios)
    widths = [w * r / total for r in ratios]
    yy = y
    for ri, row in enumerate(rows):
        h = header_h if ri == 0 else row_h
        if ri == 0:
            rect(slide, x, yy, w, h, header_fill)
        elif ri % 2 == 0:
            rect(slide, x, yy, w, h, RGBColor(0xFA, 0xFB, 0xFD))
        cx = x
        for ci, cell in enumerate(row):
            textbox(slide, cx + 0.06, yy + 0.02, widths[ci] - 0.10, h - 0.02,
                    [(str(cell), dict(size=size, bold=(ri == 0 or ci == 0),
                                      color=INK if ri else MUTED))],
                    anchor=MSO_ANCHOR.MIDDLE, space_after=0)
            cx += widths[ci]
        line(slide, x, yy + h, x + w, yy + h, LINE, 0.7)
        yy += h
    return yy


def kpi(slide, items, y=5.9, x=0.72, w=11.9, h=1.0, gap=0.14):
    n = len(items)
    cw = (w - gap * (n - 1)) / n
    for i, (value, label, color) in enumerate(items):
        xx = x + i * (cw + gap)
        rect(slide, xx, y, cw, h, PANEL2, edge=None)
        textbox(slide, xx + 0.16, y + 0.10, cw - 0.3, 0.44, [(value, dict(size=20, bold=True, color=color))])
        textbox(slide, xx + 0.16, y + 0.58, cw - 0.3, 0.34, [(label, dict(size=10, color=MUTED))])


# ---------------------------------------------------------------- 1 封面
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, 13.333, 7.5, WHITE)
rect(s, 0, 0, 0.22, 7.5, ACCENT)
textbox(s, 1.05, 1.55, 11.0, 0.4, [("赛题 9 · 工业制造场景下的模型训练与超参数优化（CPU 赛道）",
                                   dict(size=13, bold=True, color=ACCENT))])
textbox(s, 1.05, 2.05, 11.2, 1.5, [("工业模型自进化 Harness", dict(size=44, bold=True)),
                                   ("监测归因 → 封闭动作 → 门禁采纳 → 可回滚", dict(size=20, color=MUTED))],
        space_after=10)
line(s, 1.05, 3.95, 6.2, 3.95, ACCENT, 2.4)
textbox(s, 1.05, 4.20, 11.0, 1.2, [
    ("队伍：智训引擎　｜　交付版本 zhixun-harness 1.5　｜　2026-11-19", dict(size=14)),
    ("一句话：在 2 核 CPU、零新增标注、每轮一次模型调用的预算内，把三个数据集的漂移流都跑成受控闭环。",
     dict(size=13, color=MUTED))], space_after=8)
kpi(s, [("0.7140 → 0.7258", "A 气体六分类 Macro-F1", INK),
        ("681.16 → 497.75", "B 轴承 RUL RMSE（−26.9%）", ACCENT),
        ("52.87 → 51.49", "C 发动机 RUL RMSE（−2.61%）", ACCENT),
        ("0 条 / 5–6k", "新增标注 / 10 轮 token", AMBER)], y=6.05, h=0.95)
PAGES.append(s)

# ---------------------------------------------------------------- 2 我们怎么读评分
s = new_slide("评分拆解", "评分公式决定了“能做多少”，也决定了什么不能做",
              "客观分 35：漂移流 15 + 最差阶段与恢复 10 + 稳定性 5 + 资源效率 5", "指标口径见《项目说明书》§1")
bullets(s, [
    "G = (系统 − 静态基线) / (每阶段全量重训上界 − 静态基线)：分母是重训上界，纯静态与纯微调都拿不到分",
    "最差阶段单独计分：只把平均做好、末段崩掉的方案会被扣——所以门禁里“最差段不退”是硬条件",
    "资源效率按全体排名分档：算力、标注配额、大模型 token 三项都要省，省得越多越靠前",
    "稳定性要求历史工况回测保持率降幅 ≤5%：被拒轮次绝不能改写下一轮起点",
], y=2.05, gap=0.78)
rect(s, 0.72, 5.35, 11.9, 1.35, PANEL, edge=None)
textbox(s, 0.95, 5.50, 11.4, 1.1, [
    ("反过来读：这条公式奖励的是“每轮真的吸收新数据，但不许把旧能力弄丢”的系统。",
     dict(size=14, bold=True)),
    ("我们的回答：受控续训 + 封闭动作 + 门禁采纳，把“能吸收”和“不丢旧能力”同时写进不变量。",
     dict(size=13, color=MUTED))], space_after=6)

# ---------------------------------------------------------------- 3 路线选择
s = new_slide("技术路线", "三条候选路线，我们选了最不炫但最可控的那条",
              "判断依据：动作空间是否可枚举、坏动作能否被拦住、离线能否复现", "《项目说明书》§1.2")
table(s, [
    ["路线", "做法", "问题", "结论"],
    ["全量重训", "每轮从零训练", "算力最高、无历史保持、无自进化叙事", "只作对照上界"],
    ["自由超参搜索", "每轮让 LLM 给任意超参", "动作空间无界，坏动作累积损伤，离线不可复现", "否"],
    ["封闭配方 + 受控续训", "预挖掘 4 个配方，LLM 只选“哪个 + 几轮”", "需先做数据挖掘；动作空间小", "采用"],
], y=1.95, col_ratios=[2.2, 3.4, 4.6, 2.0], row_h=0.62, size=12)
textbox(s, 0.72, 5.15, 11.9, 1.3, [
    ("关键设计：LLM 的输出只有 recipe 与 epochs 能生效，未知配方一律回默认；",
     dict(size=13.5, bold=True)),
    ("任何一次模型回复都能被安全执行，模型的“自由”被限制在四个已验证配方之内。",
     dict(size=13, color=MUTED))], space_after=4)

# ---------------------------------------------------------------- 4 架构
s = new_slide("系统架构", "两层：平台持有数据、计量与隐藏标签；镜像只做三件事",
              "describe / train / predict，协议 contest.harness.workspace.v2，SDK 1.0.4", "《系统运行说明》§3")
rect(s, 0.72, 1.95, 11.9, 0.72, PANEL)
textbox(s, 0.95, 2.10, 11.4, 0.5, [
    ("平台侧：每轮注入 /data（公开投影）、/previous（父产物）、/state；提供宿主模型网关；持有隐藏验证集",
     dict(size=12.5))])
for i, (t, d) in enumerate([
        ("/app/harness.py", "协议入口：describe / train / predict，不写业务"),
        ("/app/contest_submission.py", "选手模块：续训 + 一次模型调用选配方"),
        ("rulkit/", "投影强校验 + 865~4326 参数 MLP + 配方开关"),
        ("monitor / control / governance", "KS 监测归因、门禁 v1/v2、六不变量"),
        ("memory / records", "跨轮记忆、三本账、审计链"),
        ("/workspace", "唯一可写；/data、/previous 全程只读")]):
    y = 2.92 + i * 0.56
    rect(s, 0.72, y, 4.35, 0.46, PANEL2 if i < 3 else PANEL, edge=None)
    textbox(s, 0.86, y + 0.06, 4.1, 0.36, [(t, dict(size=11.5, bold=True, color=ACCENT))])
    textbox(s, 5.30, y + 0.06, 7.3, 0.36, [(d, dict(size=11.5, color=INK))])
textbox(s, 0.72, 6.62, 11.9, 0.34, [
    ("依赖只有 numpy 1.26.4；线程=1；无 GPU、无外网、无密钥——离线环境可直接跑。",
     dict(size=11, color=MUTED))])

# ---------------------------------------------------------------- 5 一轮时序
s = new_slide("一轮的时序", "八步之内，所有状态变化都留痕",
              "每轮一个容器、一次真实模型调用、一个候选产物", "《项目说明书》§2.2")
steps = [
    ("① 收到 start 帧", "round_index / dataset_key / task"),
    ("② 读父检查点", "校验 sha256，不符直接拒跑"),
    ("③ 读投影并校验", "schema 必须与父模型一致"),
    ("④ 测当轮指标 + 模型调用", "验证集或 probe；temperature=0，max_tokens≤600"),
    ("⑤ 应用配方续训", "LR/L2/SWA 都由配方固化"),
    ("⑥ 验证集 before/after", "1.5.1 起同时作为下一轮决策输入"),
    ("⑦ 提交候选 + finish", "必带 model_call_id"),
    ("⑧ 宿主评分采纳/拒绝", "被拒不改写下一轮起点"),
]
for i, (t, d) in enumerate(steps):
    col = i % 4
    row = i // 4
    x = 0.72 + col * 3.02
    y = 2.05 + row * 2.05
    rect(s, x, y, 2.86, 1.72, PANEL if row == 0 else PANEL2, edge=None)
    textbox(s, x + 0.14, y + 0.12, 2.6, 0.36, [(t, dict(size=13, bold=True, color=ACCENT))])
    textbox(s, x + 0.14, y + 0.52, 2.6, 0.95, [(d, dict(size=11, color=INK))])
rect(s, 0.72, 6.32, 11.9, 0.62, WHITE, edge=LINE)
textbox(s, 0.90, 6.42, 11.5, 0.44, [
    ("I1 不变量：被拒轮次只影响 1 轮——坏动作的影响半径被门禁封死，历史回测保持率不下降。",
     dict(size=12.5, bold=True, color=AMBER))])

# ---------------------------------------------------------------- 6 漂移监测
s = new_slide("漂移监测与诊断", "监测不是一个报警灯，而是决定“这轮该不该花”的输入",
              "参考窗 = 验证集前 25%；统计量 = 逐特征两样本 KS 的 Top-3 均值", "《项目说明书》§3")
table(s, [
    ["环节", "做法", "为什么这样做"],
    ["参考窗", "每轮取公开验证集前 25%，ref_stop = max(32, n//4)", "不引入外部参考数据，随投影自适应"],
    ["统计量", "逐特征 KS × sqrt(nm/(n+m))，取 Top-3 均值", "少数特征真漂不被大量稳定特征稀释"],
    ["阈值", "参考窗自举 24 次：warn=max(q95×1.05,0.05)，alarm=max(q99.5×1.05,1.5×warn)", "阈值随数据规模变，不写死"],
    ["诊断", "特征 KS 显著→covariate；标签位移→concept；同时成立→mixed", "把“输入变了”和“规则变了”分开"],
    ["接线", "covariate 偏重加权，concept 偏重续训与末端加权，mixed 走保守配方", "监测结果直接进动作选择"],
], y=1.95, col_ratios=[1.4, 5.6, 4.9], row_h=0.62, size=11.5)
textbox(s, 0.72, 5.64, 11.9, 1.2, [
    ("判定结果进 records 账本，成为下一轮记忆的输入；但记忆只能影响候选，不能越过门禁。",
     dict(size=12.5, color=MUTED))])

# ---------------------------------------------------------------- 7 监测基准
s = new_slide("监测基准", "A、B 全对，C 只对 2/4——这个结果我们写进材料，没有藏",
              "合成漂移流：每类 4 种漂移构造（特征缩放、噪声、比例/工况切换、混合）", "《项目说明书》§3.3–3.4")
for i, (name, hit, note, ok) in enumerate([
        ("A 气体六分类", "4 / 4", "四类漂移全部判对", True),
        ("B 轴承 RUL", "4 / 4", "四类漂移全部判对", True),
        ("C 发动机 RUL", "2 / 4", "工况切换与 P(x) 漂移在行级 KS 上同值（1.85）", False)]):
    x = 0.72 + i * 4.02
    rect(s, x, 1.95, 3.82, 2.05, PANEL2 if ok else RGBColor(0xFE, 0xF2, 0xF2), edge=None)
    textbox(s, x + 0.20, 2.08, 3.4, 0.34, [(name, dict(size=13, bold=True))])
    textbox(s, x + 0.20, 2.44, 3.4, 0.62, [(hit, dict(size=30, bold=True, color=ACCENT if ok else RED))])
    textbox(s, x + 0.20, 3.14, 3.45, 0.76, [(note, dict(size=10.5, color=MUTED))])
rect(s, 0.72, 4.25, 11.9, 1.05, PANEL, edge=None)
textbox(s, 0.95, 4.36, 11.4, 0.85, [
    ("C 的失败不是阈值没调好：只看单特征边际分布时，“工况切换”和“P(x) 漂移”在行级 KS 上不可区分。",
     dict(size=12.5, bold=True)),
    ("要分开必须引入条件分布或时序证据，超出 2 核 / 4 GiB 预算下可稳定实现的范围——保留为已知局限。",
     dict(size=12, color=MUTED))], space_after=4)
textbox(s, 0.72, 5.55, 11.9, 0.9, [
    ("结论：监测在 A/B 上足以驱动动作；在 C 上我们选择“归因不确定就走保守配方”，",
     dict(size=12.5)),
    ("宁可少赚一轮，也不让错误归因把模型带偏。", dict(size=12.5, bold=True, color=ACCENT))], space_after=3)

# ---------------------------------------------------------------- 8 门禁
s = new_slide("模型更新与受控自改进", "动作空间封闭，采纳判定独立于决策模型",
              "4 配方 × epochs 1..16；门禁同时看平均增益与最差段", "《项目说明书》§4")
table(s, [
    ["门禁", "规则", "效果"],
    ["v1", "gain ≥ min_gain（A 0.002 / B 5.0 / C 0.5）且 worst_gain ≥ −0.02×|静态|", "简单可解释，挡住末段崩坏"],
    ["v2", "多轨迹均值 − k·se 下界 + 配对 bootstrap 下界 > 0", "小验证集上降低误采纳"],
    ["无门禁对照", "每轮都采纳", "A 掉到 0.6296（−9.6 点），B 末段 −108.6"],
], y=1.90, col_ratios=[1.6, 6.6, 3.9], row_h=0.58, size=11.5)
textbox(s, 0.72, 4.16, 11.9, 0.34, [("六条不变量（治理层强制，违反即拒跑）", dict(size=13, bold=True))])
inv = [("I1", "被拒轮次不作下一轮起点"), ("I2", "父检查点摘要不符即拒"), ("I3", "训练只读 /data"),
       ("I4", "候选只写 /workspace"), ("I5", "每轮恰好一次真实调用"), ("I6", "finish 必带 call_id")]
for i, (k, v) in enumerate(inv):
    x = 0.72 + (i % 3) * 4.02
    y = 4.50 + (i // 3) * 0.74
    rect(s, x, y, 3.82, 0.66, PANEL2, edge=None)
    textbox(s, x + 0.14, y + 0.10, 0.6, 0.42, [(k, dict(size=13, bold=True, color=ACCENT))])
    textbox(s, x + 0.72, y + 0.12, 3.0, 0.44, [(v, dict(size=11, color=INK))])
textbox(s, 0.72, 6.06, 11.9, 0.6, [
    ("自改进的边界：记忆只影响“候选”，不影响“采纳”。1.5.1 起配方空间按任务门控，执行异常走“默认配方→official”三级回退。",
     dict(size=12, color=MUTED))])

# ---------------------------------------------------------------- 9 结果
s = new_slide("实验结果", "三个数据集，两种结局：B 是压倒性，A 是“不乱动”",
              "同一份公开投影、同一隐藏验证集、同一评分公式", "《项目说明书》§5")
table(s, [
    ["数据", "指标", "静态基线", "本系统 1.5", "增益", "显著性", "最差段"],
    ["A 气体六分类", "Macro-F1 ↑", "0.7140", "0.7258", "+0.0086", "0.44σ（不显著）", "0（不可达）"],
    ["B 轴承 RUL", "RMSE ↓", "681.1598", "497.7463", "+183.41（+26.9%）", "≈20σ", "+169.76"],
    ["C 发动机 RUL", "RMSE ↓", "52.8686", "51.4884", "−2.61%", "bootstrap 下界>0", "+1.25"],
], y=1.90, col_ratios=[1.9, 1.7, 1.5, 1.6, 1.9, 2.1, 1.4], row_h=0.60, size=11)
kpi(s, [("+183.41", "B 的 RMSE 改进（≈20σ）", ACCENT),
        ("+169.76", "B 最差段增益（未退化）", ACCENT),
        ("1/10", "B 采纳率：首轮即最优", INK),
        ("0.44σ", "A 的增益：不显著，如实标注", AMBER)], y=4.55, h=0.95)
textbox(s, 0.72, 5.75, 11.9, 1.1, [
    ("B 的增益来自三件事叠加：中心化损失（同 batch 按轴承去均值、梯度 stop-gradient）+ 末端谱能量加权 + 末 2 epoch SWA。",
     dict(size=12)),
    ("A 的价值不在分数，而在证明系统在信息不足时不会乱动——这也是工业场景更需要的性质。",
     dict(size=12, color=MUTED))], space_after=4)

# ---------------------------------------------------------------- 10 对照
s = new_slide("对照分析", "把门禁和监测拿掉，平均分看着还行，最差段直接崩",
              "同一数据集、同样轮次，只改采纳规则", "《项目说明书》§5.4")
table(s, [
    ["配置", "A Macro-F1", "B RMSE", "C RMSE", "结论"],
    ["静态基线（不更新）", "0.7140", "681.16", "52.87", "基准"],
    ["每轮全采纳（无门禁）", "0.6296", "648.88", "52.14", "平均尚可，最差段 −108.6"],
    ["有监测 + 有门禁（本系统）", "0.7258", "497.75", "51.49", "分段与最差段都稳"],
], y=1.95, col_ratios=[2.9, 1.9, 1.7, 1.6, 3.8], row_h=0.66, size=11.5)
rect(s, 0.72, 4.45, 11.9, 1.20, PANEL, edge=None)
textbox(s, 0.95, 4.55, 11.4, 1.0, [
    ("门禁单独贡献 9.6 个点（A：0.7258 vs 0.6296）。", dict(size=14, bold=True)),
    ("它拦下的不是“差方案”，而是“平均好看但把末段弄坏”的方案——这正是评分里最差阶段那一项要防的。",
     dict(size=12, color=MUTED))], space_after=4)
textbox(s, 0.72, 5.85, 11.9, 1.0, [
    ("C 的无门禁版本看起来也不差（52.14），但它的采纳轮次更多、方差更大；",
     dict(size=12)),
    ("在只有 10 轮的预算里，“少而准”比“多而噪”更有价值。", dict(size=12, bold=True, color=ACCENT))])

# ---------------------------------------------------------------- 11 成本
s = new_slide("资源成本", "三本账：算力几乎免费，标注为零，token 是唯一线性成本",
              "评分按全体排名分档，所以“省”本身就是分", "《项目说明书》§6")
table(s, [
    ["账", "用量", "口径"],
    ["算力", "2 核 CPU / 4 GiB；10 轮训练总耗时十几秒到二十秒级", "平台 cost_units：常规轮 0，v2 bootstrap 轮 63–227"],
    ["标注", "新增人工标注 0 条", "只用公开投影的 validation 前 25% 作参考窗"],
    ["Token", "每轮 1 次调用，10 轮合计 5–6k", "1.5 单轮输入≈610 字符、输出≤600 token"],
], y=1.95, col_ratios=[1.3, 5.4, 5.2], row_h=0.72, size=11.5)
textbox(s, 0.72, 4.62, 11.9, 0.34, [("与 1.4 的对照（同口径，10 轮）", dict(size=12.5, bold=True))])
table(s, [
    ["版本", "调用次数", "输入 token", "输出 token", "降级次数"],
    ["1.4 修复前 a_llm2", "10", "6,156", "5,779", "2"],
    ["1.4 修复后 a_llm_v1fix", "10", "9,521", "1,389", "0"],
    ["1.5（结构压缩）", "10", "≈5–6k 合计", "—", "0"],
], y=4.95, col_ratios=[3.0, 1.6, 1.9, 1.9, 1.5], row_h=0.40, header_h=0.40, size=10.5)
textbox(s, 0.72, 6.62, 11.9, 0.34, [
    ("降级次数是“模型回复不可用”的次数：修复前 2/10，修复后 0/10——提示词结构与校验口径一起改的。",
     dict(size=10.5, color=MUTED))])

# ---------------------------------------------------------------- 12 微调证据
s = new_slide("决策模型后训练", "微调过，有效，但只当加分证据，不当默认决策器",
              "Qwen3.8-27B LoRA：rank 32、2 epoch、93 条、73,649 token ≈ ¥0.66", "《项目说明书》§6.3")
table(s, [
    ["臂（旧 gas 流，同 seed）", "acc", "G", "单位增益", "token"],
    ["规则", "0.7174", "0.7088", "0.0492", "—"],
    ["基座 Qwen3.8-27B", "0.7053", "0.6914", "0.1223", "9,044"],
    ["LoRA 微调（P5）", "0.6851", "0.6503", "0.2451", "7,604"],
], y=1.95, col_ratios=[4.2, 1.6, 1.6, 1.9, 1.7], row_h=0.52, size=11.5)
kpi(s, [("2.0–2.7×", "微调 vs 基座的单位算力增益", ACCENT),
        ("93%", "保留基座增益的比例", INK),
        ("0.00 → 0.70", "离线留出集 exact_ok", ACCENT),
        ("1.5–2×", "token 反而更多（更长的 JSON）", AMBER)], y=4.35, h=0.95)
textbox(s, 0.72, 5.55, 11.9, 1.2, [
    ("为什么不进评分路径：① 训练数据是旧项目的 gas 流，不是本赛道 A/B/C 投影，直接引用不诚实；② temperature=0 仍有远端抖动（两次复跑 acc 0.6851 → 0.6934）；",
     dict(size=11.5)),
    ("③ 平台是否允许 trio:// sampler 作为模型身份尚未确认。定位：方法是成立的，证据是局部的，所以只放加分项。",
     dict(size=11.5, color=MUTED))], space_after=3)

# ---------------------------------------------------------------- 13 安全
s = new_slide("安全可控", "七类风险，七条控制，全部可离线验证",
              "安全不是文档里的形容词，是镜像里可检查的约束", "《项目说明书》§7.1")
table(s, [
    ["风险", "控制措施"],
    ["坏动作损伤模型", "门禁 + I1：坏动作影响半径 = 1 轮，历史回测保持率不下降"],
    ["标签泄漏 / 隐藏集", "训练只读 /data/train；禁止 rul 等目标派生字段；隐藏集不可见"],
    ["越权改环境", "镜像内无密钥无端点；/app 只读；产物只写 /workspace"],
    ["父产物被篡改", "checkpoint_sha256 不符即拒跑（PARENT_DIGEST_MISMATCH）"],
    ["不可解释的自改进", "动作空间封闭（4×16）；记忆只影响候选，不影响采纳"],
    ["故障不可恢复", "三级回滚：回配方（秒级）→ 回检查点（单轮）→ 回镜像（分钟级）"],
    ["评测不可复现", "线程固定 1、无时间戳、temperature=0、每轮账本落盘"],
], y=1.92, col_ratios=[2.6, 9.3], row_h=0.56, size=11.5)
textbox(s, 0.72, 6.42, 11.9, 0.5, [
    ("离线自检：docker load → describe（返回协议 v2）→ predict（挂本地 checkpoint）→ sha256 校验。",
     dict(size=11, color=MUTED))])

# ---------------------------------------------------------------- 14 工业价值
s = new_slide("工业落地价值", "产线愿意放权的三个前提：省、可解释、失败可承受",
              "这三点恰好也是评分主观项里的“工业可控与落地”", "《项目说明书》§7.2")
items = [
    ("迁移成本低", "换产线只换 /data 投影与父检查点；配方与门禁参数按新数据重标定，不重训大模型"),
    ("边际成本近零", "每轮 = 秒级续训 + 一次极短调用；可以“每班一轮”，不必“每季度重训”"),
    ("决策链可读", "每轮都有 hypothesis / change_summary / reason / model_call_id，工艺工程师能读懂"),
    ("失败可承受", "最差情况是“这轮不采纳”，不是“模型被带坏”——7×24 产线能不能上线的分水岭"),
]
for i, (t, d) in enumerate(items):
    y = 1.95 + i * 1.06
    rect(s, 0.72, y, 11.9, 0.92, PANEL if i % 2 == 0 else PANEL2, edge=None)
    textbox(s, 0.95, y + 0.10, 2.8, 0.34, [(t, dict(size=13.5, bold=True, color=ACCENT))])
    textbox(s, 3.85, y + 0.10, 8.6, 0.74, [(d, dict(size=11.5))])
textbox(s, 0.72, 6.30, 11.9, 0.6, [
    ("边界同样写清楚：C 的归因受限于边际统计量；v2 门禁的 bootstrap 会抬高 cost_units；",
     dict(size=11, color=MUTED)),
    ("高维图像/振动场景需要重新验证。", dict(size=11, color=MUTED))])

# ---------------------------------------------------------------- 15 结论
s = new_slide("结论", "一句话：把“每轮续训”做成可审计、可回滚的闭环",
              "不是训练一个更强的模型，而是让模型在漂移里稳定地变好")
bullets(s, [
    "B：681.16 → 497.75（−26.9%，≈20σ），三段全正，最差段 +169.76",
    "A：0.7140 → 0.7258（+0.0086，0.44σ）——并证明最差阶段在合规空间内不可达",
    "C：52.87 → 51.49（−2.61%），配对 bootstrap 下界 > 0",
    "成本：0 条新增标注，10 轮 5–6k token，2 核 CPU 秒级/轮",
    "安全：六条不变量 + 三级回滚 + 全链路账本，离线可复现",
], y=1.95, gap=0.66, size=14)
rect(s, 0.72, 5.55, 11.9, 1.15, PANEL2, edge=None)
textbox(s, 0.95, 5.68, 11.4, 0.9, [
    ("我们认为真正的产出是两个否定结论：A 的最差阶段不可达、C 的归因在边际统计量上不可分。",
     dict(size=13, bold=True, color=ACCENT)),
    ("把它们写进材料，比多刷 0.5 个点更能说明这套系统在工业现场会发生什么。",
     dict(size=12, color=MUTED))], space_after=4)

# ---------------------------------------------------------------- 16 材料
s = new_slide("交付清单", "七项材料全部对齐 1.5.1 版本", "镜像 sha256 9ff79a47…3345，12 层，linux/amd64")
table(s, [
    ["材料", "内容"],
    ["模型配置（txt）", "协议、模型身份、rnd.model 参数、四配方、门禁与降级路径"],
    ["系统镜像（网盘 tar）", "zhixun-harness-1.5.tar，65,976,320 B；超 50 MB 走网盘，另附 docx 下载说明"],
    ["云端实验记录（txt）", "SwanLab 项目 university_AI_innovate，40 条 run 链接，覆盖 A/B/C"],
    ["系统运行说明（pdf）", "环境、依赖、启动、配置、输入输出、离线、复现、异常、回滚"],
    ["项目说明书（pdf）", "架构、漂移监测与诊断、模型更新与受控自改进、三本账、安全、落地"],
    ["答辩 PPT（pptx）", "本文件"],
    ["源代码（zip）", "harness 1.5 + engine 1.4 + scripts/tests + finetune-legacy + 许可证"],
], y=1.95, col_ratios=[3.0, 9.0], row_h=0.56, size=11)
textbox(s, 0.72, 6.45, 11.9, 0.5, [
    ("所有材料不含隐藏测试集、不含账号密钥；开源组件与许可证见源码包 THIRD_PARTY.md。",
     dict(size=10.5, color=MUTED))])

import re

total = len(prs.slides._sldIdLst)
for i, sl in enumerate(prs.slides, 1):
    for shape in sl.shapes:
        if not shape.has_text_frame:
            continue
        text = shape.text_frame.text.strip()
        if re.fullmatch(r"\d+\s*/\s*\d+", text):
            for para in shape.text_frame.paragraphs:
                for run in para.runs:
                    run.text = "%d / %d" % (i, total)
                    break
                break

prs.save(str(OUT))
print("OK", OUT, "slides=", len(prs.slides.__iter__.__self__._sldIdLst))