"""Markdown → 矢量 PDF（纯 matplotlib，不依赖 Edge/LibreOffice/网络）。

为什么不用 _md_to_pdf.py：本机沙箱禁止父子进程开命名管道（CreatePipe → WinError 5），
Edge 无头打印在 profile 初始化时被拒（network_sandbox 拒绝访问），写不出 PDF。
本机也没有 LibreOffice、reportlab、weasyprint。matplotlib 3.10 + 系统字体（SimHei）
能直接输出矢量文本 PDF，可选中、可放大、体积小。

    python _md_pdf.py <in.md> [--output out.pdf] [--title "..."]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib import lines as mlines, patches as mpatches  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.font_manager import FontProperties, fontManager  # noqa: E402

PAGE_W, PAGE_H = 8.27, 11.69          # A4 英寸
ML, MR, MT, MB = 0.72, 0.72, 0.62, 0.66
BODY = 9.2
LINE = 1.62

CJK_FONT = Path(r"C:\Windows\Fonts\simhei.ttf")
MONO_FONT = Path(r"C:\Windows\Fonts\consola.ttf")
BOLD_FONT = Path(r"C:\Windows\Fonts\msyhbd.ttc")


def _register():
    for path in (CJK_FONT, MONO_FONT, BOLD_FONT):
        if path.exists():
            try:
                fontManager.addfont(str(path))
            except Exception:
                pass
    return fontManager.get_font_names()


NAMES = _register()


def fp(size, bold=False, mono=False):
    if mono:
        family = ["Consolas", "SimHei", "DejaVu Sans Mono"]
    elif bold:
        family = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    else:
        family = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    return FontProperties(family=family, size=size, weight="bold" if bold else "normal")


_probe = Figure(figsize=(PAGE_W, PAGE_H), dpi=144)
FigureCanvasAgg(_probe)
_probe.canvas.draw()
_RENDERER = _probe.canvas.get_renderer()


def text_w(text, font):
    if not text:
        return 0.0
    return _RENDERER.get_text_width_height_descent(text, font, ismath=False)[0] / 144.0


# ------------------------------------------------------------------ inline rich
TOKEN = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def rich(text):
    """→ [(chunk, bold, mono)]；CJK 逐字可断，英文按词。”
    """
    out = []
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            out.append((part[2:-2], True, False))
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            out.append((part[1:-1], False, True))
        else:
            out.append((part, False, False))
    return out


def wrap(segments, width, size):
    """把 rich 段按字符/词断成多行，返回 [[(text,bold,mono), ...], ...]。"""
    lines, cur, cur_w = [], [], 0.0
    for text, bold, mono in segments:
        font = fp(size, bold, mono)
        tokens = re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]|[^\s\u4e00-\u9fff]+|\s+", text)
        for tok in tokens:
            w = text_w(tok, font)
            if cur and cur_w + w > width and not tok.isspace():
                lines.append(cur)
                cur, cur_w = [], 0.0
            if cur_w + w > width and not cur:          # 超长单词：硬切
                cut = ""
                for ch in tok:
                    if cur_w + text_w(cut + ch, font) > width:
                        break
                    cut += ch
                if cut:
                    lines.append([(cut, bold, mono)])
                    rest = tok[len(cut):]
                    cur_w = text_w(rest, font)
                    cur = [(rest, bold, mono)] if rest else []
                continue
            if tok.isspace() and not cur:
                continue
            cur.append((tok, bold, mono))
            cur_w += w
    if cur:
        lines.append(cur)
    return lines or [[]]


# ---------------------------------------------------------------------- blocks
def parse(md: str):
    blocks = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        if not line.strip():
            i += 1
            continue
        if line.lstrip().startswith("```"):
            body = []
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("```"):
                body.append(lines[i])
                i += 1
            blocks.append(("code", "\n".join(body)))
            i += 1
            continue
        if line.lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not re.match(r"^[\s:|-]+$", "".join(cells)):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows))
            widths = {len(r) for r in rows}
            if len(widths) > 1:
                print("WARN ragged table near line %d: cell counts %s"
                      % (i, sorted(widths)), file=sys.stderr)
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            blocks.append(("h%d" % len(m.group(1)), m.group(2).strip()))
            i += 1
            continue
        if re.match(r"^\s*([-*+]|\d+\.)\s+", line):
            indent = (len(raw) - len(raw.lstrip())) // 2
            text = re.sub(r"^\s*([-*+]|\d+\.)\s+", "", line)
            blocks.append(("li", (indent, text)))
            i += 1
            continue
        if line.lstrip().startswith(">"):
            blocks.append(("quote", line.lstrip()[1:].strip()))
            i += 1
            continue
        if re.match(r"^\s*([-*_])\1{2,}\s*$", line):
            blocks.append(("hr", None))
            i += 1
            continue
        blocks.append(("p", line.strip()))
        i += 1
    return blocks


# 第三项是“段后间距”，单位是当前字号的多少倍行高
STYLE = {
    "h1": (16.0, True, 1.5),
    "h2": (12.4, True, 1.0),
    "h3": (10.4, True, 0.75),
    "h4": (9.6, True, 0.6),
    "p": (BODY, False, 0.5),
    "quote": (BODY, False, 0.7),
    "code": (7.9, False, 0.6),
}


class Pager:
    def __init__(self, title):
        self.title = title
        self.pages = []
        self.new_page()

    def new_page(self):
        self.fig = Figure(figsize=(PAGE_W, PAGE_H))
        self.pages.append(self.fig)
        self.y = PAGE_H - MT
        self.first = True

    def ensure(self, height):
        if self.y - height < MB:
            self.new_page()

    def text(self, x, y, segs, size, mono=False):
        cx = x
        for chunk, bold, is_mono in segs:
            font = fp(size, bold, mono or is_mono)
            self.fig.text(cx / PAGE_W, y / PAGE_H, chunk, fontproperties=font,
                          va="top", ha="left", color="#111111", parse_math=False)
            cx += text_w(chunk, font)

    def heading_rule(self, y):
                self.fig.add_artist(mlines.Line2D([ML / PAGE_W, (PAGE_W - MR) / PAGE_W], [y / PAGE_H, y / PAGE_H],
                                   color="#c8ccd4", linewidth=0.7))

    def emit(self, block):
        kind, payload = block
        width = PAGE_W - ML - MR
        if kind in STYLE and kind != "li":
            size, bold, gap = STYLE[kind]
            if kind == "code":
                font = fp(size, mono=True)
                avail = width - 0.26
                wrapped = []
                for raw_line in ((payload or "").splitlines() or [""]):
                    cur = ""
                    for ch in raw_line:
                        if cur and text_w(cur + ch, font) > avail:
                            wrapped.append(cur)
                            cur = ch
                        else:
                            cur += ch
                    wrapped.append(cur)
                top = self.y
                for line in wrapped:
                    h = size * 1.32 / 72.0
                    self.ensure(h)
                    self.fig.text((ML + 0.10) / PAGE_W, self.y / PAGE_H, line or " ",
                                  fontproperties=font, va="top", ha="left", color="#1f2328",
                                  parse_math=False)
                    self.y -= h
                self.fig.add_artist(mlines.Line2D(
                    [(ML + 0.03) / PAGE_W, (ML + 0.03) / PAGE_W],
                    [(top + 0.02) / PAGE_H, (self.y + 0.04) / PAGE_H],
                    color="#b9c0cc", linewidth=2.2))
                self.y -= gap * size * 1.32 / 72.0
                return
            lines = wrap(rich(payload), width, size)
            for line in lines:
                self.ensure(size * LINE / 72.0)
                self.text(ML, self.y, line, size)
                self.y -= size * LINE / 72.0
            self.y -= gap * size * LINE / 72.0
            if kind in ("h1", "h2"):
                self.heading_rule(self.y + 0.03)
            return

        if kind == "li":
            indent, text = payload
            x = ML + 0.22 + indent * 0.24
            size = BODY
            lines = wrap(rich(text), width - (x - ML), size)
            for n, line in enumerate(lines):
                h = size * LINE / 72.0
                self.ensure(h)
                if n == 0:
                    self.fig.text((x - 0.17) / PAGE_W, self.y / PAGE_H, "•",
                                  fontproperties=fp(size), va="top", ha="left", color="#555555")
                self.text(x, self.y, line, size)
                self.y -= h
            return

        if kind == "quote":
            size = BODY
            lines = wrap(rich(payload), width - 0.3, size)
            for n, line in enumerate(lines):
                h = size * LINE / 72.0
                self.ensure(h)
                self.fig.add_artist(mlines.Line2D(
                    [ML / PAGE_W, ML / PAGE_W], [self.y / PAGE_H, (self.y - h) / PAGE_H],
                    color="#9aa4b2", linewidth=2.0))
                self.text(ML + 0.16, self.y, line, size)
                self.y -= h
            return

        if kind == "hr":
            self.ensure(0.16)
            self.heading_rule(self.y)
            self.y -= 0.16
            return

        if kind == "table":
            self.table(payload, width)
            return

    def table(self, rows, width):
        if not rows:
            return
        cols = max(len(r) for r in rows)
        size = 7.4
        # 列宽按内容实测；窄列给足，剩余等分给超宽列（water-filling），
        # 避免等比压缩把短列挤成竖排（v1.5.1 修）。
        need = [max(text_w(r[c] if c < len(r) else "", fp(size, bold=(n == 0)))
                    for n, r in enumerate(rows)) + 0.30 for c in range(cols)]
        alloc = [0.0] * cols
        pending = list(range(cols))
        remaining = width
        while pending:
            share = remaining / len(pending)
            narrow = [i for i in pending if need[i] <= share]
            if not narrow:
                for i in pending:
                    alloc[i] = share
                break
            for i in narrow:
                alloc[i] = need[i]
                remaining -= need[i]
            pending = [i for i in pending if i not in narrow]
        xs, acc = [], ML
        for n in alloc:
            xs.append(acc)
            acc += n
        line_h = size * 1.5 / 72.0
        for ri, row in enumerate(rows):
            cells = [wrap(rich(row[c] if c < len(row) else ""), max(alloc[c] - 0.16, 0.35), size)
                     for c in range(cols)]
            h = max(len(c) for c in cells) * line_h + 0.06
            self.ensure(h)
            for c in range(cols):
                for li, line in enumerate(cells[c]):
                    segs = [(t, True, mo) for t, _, mo in line] if ri == 0 else line
                    self.text(xs[c], self.y - 0.04 - li * line_h, segs, size)
            if ri == 0:
                self.fig.add_artist(mlines.Line2D(
                    [ML / PAGE_W, (PAGE_W - MR) / PAGE_W],
                    [(self.y - h + 0.02) / PAGE_H, (self.y - h + 0.02) / PAGE_H],
                    color="#9aa4b2", linewidth=0.9))
            self.y -= h
        self.fig.add_artist(mlines.Line2D(
            [ML / PAGE_W, (PAGE_W - MR) / PAGE_W], [self.y / PAGE_H, self.y / PAGE_H],
            color="#c8ccd4", linewidth=0.7))
        self.y -= 0.10


def build(md_path: Path, out_pdf: Path, title: str):
    md = md_path.read_text(encoding="utf-8")
    # 去掉 HTML 注释
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    blocks = parse(md)
    pager = Pager(title)
    for block in blocks:
        pager.emit(block)

    with PdfPages(out_pdf, metadata=PdfInfo(title, md_path.name)) as pdf:
        for fig in pager.pages:
            pdf.savefig(fig)
    return len(pager.pages)


class PdfInfo(dict):
    def __init__(self, title, source):
        super().__init__({
            "Title": title, "Author": "智训引擎", "Subject": "企业赛提交材料",
            "Creator": "official/_md_pdf.py", "Producer": "matplotlib", "Keywords": source,
        })


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--output", default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)
    src = Path(args.source).resolve()
    out = Path(args.output).resolve() if args.output else src.with_suffix(".pdf")
    title = args.title or src.stem
    pages = build(src, out, title)
    raw = out.read_bytes()
    images = raw.count(b"/Subtype /Image")
    print(f"OK {out} pages={pages} bytes={len(raw)} image_objects={images}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())