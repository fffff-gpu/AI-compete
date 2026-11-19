# -*- coding: utf-8 -*-
"""把 PPTX 画成近似的版式草图（PNG），用于本机无 Office/LibreOffice 时的目检。

本机两个 Python 环境互补：DSH 自带 python 有 python-pptx 无 matplotlib，
系统 python（D:\\andcond）有 matplotlib 无 python-pptx，所以拆两步：

    # 1) 导出形状（用 DSH 自带 python）
    python _sketch_deck.py export 答辩PPT-1.5.pptx deck.json
    # 2) 画图（用系统 python）
    python _sketch_deck.py render deck.json _preview_deck

不是渲染器：只还原色块、文字行位置与换行，用来判断“挤不挤、压不压、对齐歪不歪”。
"""
import json
import sys
import unicodedata
from pathlib import Path


def units(ch):
    if ch == " ":
        return 0.30
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return 1.0
    if ch in "iljtfr.,:;'|!":
        return 0.30
    return 0.55


def color_of(shape):
    try:
        fill = shape.fill
        if fill.type is not None and fill.fore_color and fill.fore_color.rgb is not None:
            rgb = fill.fore_color.rgb
            return "#%02X%02X%02X" % (rgb[0], rgb[1], rgb[2])
    except Exception:
        pass
    return None


def export(src, dst):
    from pptx import Presentation

    prs = Presentation(src)
    slides = []
    for slide in prs.slides:
        shapes = []
        for shape in slide.shapes:
            item = {
                "x": shape.left / 914400, "y": shape.top / 914400,
                "w": shape.width / 914400, "h": shape.height / 914400,
                "fill": color_of(shape), "paras": [],
            }
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = "".join(r.text for r in para.runs) or para.text
                    if not text.strip():
                        continue
                    size = max((r.font.size.pt for r in para.runs if r.font.size), default=14)
                    bold = any(r.font.bold for r in para.runs)
                    rgb = None
                    for r in para.runs:
                        if r.font.color and r.font.color.type is not None:
                            try:
                                rgb = r.font.color.rgb
                                break
                            except Exception:
                                rgb = None
                    item["paras"].append({
                        "text": text, "size": size, "bold": bool(bold),
                        "color": "#%02X%02X%02X" % (rgb[0], rgb[1], rgb[2]) if rgb else "#1F2937",
                        "space_after": (para.space_after.pt if para.space_after else 0.0),
                    })
            if item["fill"] or item["paras"]:
                shapes.append(item)
        slides.append(shapes)
    Path(dst).write_text(json.dumps({"w": prs.slide_width / 914400,
                                     "h": prs.slide_height / 914400,
                                     "slides": slides}, ensure_ascii=False), encoding="utf-8")
    print("exported", dst, "slides=", len(slides))


def wrap(text, size_pt, width_in):
    if not text:
        return [""]
    budget = width_in * 72.0
    lines, cur, line = [], 0.0, ""
    for ch in text:
        w = units(ch) * size_pt
        if cur + w > budget and line:
            lines.append(line)
            line, cur = ch, w
        else:
            line += ch
            cur += w
    lines.append(line)
    return lines


def render(src, outdir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    data = json.loads(Path(src).read_text(encoding="utf-8"))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for idx, shapes in enumerate(data["slides"], 1):
        fig = plt.figure(figsize=(data["w"], data["h"]), dpi=105)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, data["w"])
        ax.set_ylim(data["h"], 0)
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0), data["w"], data["h"], facecolor="white", edgecolor="none"))
        for item in shapes:
            if item["fill"]:
                ax.add_patch(Rectangle((item["x"], item["y"]), item["w"], item["h"],
                                       facecolor=item["fill"], edgecolor="none", alpha=0.9))
        for item in shapes:
            cy = item["y"] + 0.04
            for para in item["paras"]:
                for line in wrap(para["text"], para["size"], max(item["w"] - 0.06, 0.3)):
                    ax.text(item["x"] + 0.03, cy + para["size"] / 72.0 * 0.78, line,
                            fontsize=para["size"], color=para["color"],
                            fontweight="bold" if para["bold"] else "normal",
                            va="baseline", ha="left", family="Microsoft YaHei", zorder=5)
                    cy += para["size"] * 1.30 / 72.0
                cy += para["space_after"] / 72.0
        fig.savefig(out / ("slide-%02d.png" % idx), dpi=105, facecolor="white")
        plt.close(fig)
    print("rendered", out, "pages=", len(data["slides"]))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "export":
        export(sys.argv[2], sys.argv[3])
    elif mode == "render":
        render(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("usage: _sketch_deck.py export|render ...")