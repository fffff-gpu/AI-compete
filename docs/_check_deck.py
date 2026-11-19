"""Deck geometry / overflow sanity check (no renderer available on this box).

    python _check_deck.py [deck.pptx]

估算口径：CJK 字宽 ≈ 字号，西文 ≈ 字号×0.55；行高 ≈ 字号×1.32。
只是启发式，用来抓“明显塞不下/压在一起”的框；最终仍需目检。
"""
import sys
import unicodedata
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu


def char_units(ch):
    if ch == " ":
        return 0.30
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return 1.0
    if ch in "iljtfr.,:;'|!":
        return 0.30
    return 0.55


def est_lines(text, size_pt, width_in):
    if not text:
        return 0
    unit = size_pt / 72.0
    budget = width_in
    lines, cur = 1, 0.0
    for ch in text:
        w = char_units(ch) * unit
        if cur + w > budget:
            lines += 1
            cur = w
        else:
            cur += w
    return lines


def text_of(shape):
    if not shape.has_text_frame:
        return ""
    return "\n".join(p.text for p in shape.text_frame.paragraphs)


def main(path):
    prs = Presentation(path)
    sw, sh = prs.slide_width / 914400, prs.slide_height / 914400
    problems = 0
    print("slides=%d size=%.2fx%.2fin" % (len(prs.slides), sw, sh))
    for idx, slide in enumerate(prs.slides, 1):
        boxes = []
        for shape in slide.shapes:
            x, y = shape.left / 914400, shape.top / 914400
            w, h = shape.width / 914400, shape.height / 914400
            if shape.has_text_frame and shape.text_frame.text.strip():
                total_h = 0.0
                for para in shape.text_frame.paragraphs:
                    text = "".join(r.text for r in para.runs)
                    if not text:
                        total_h += 0.12
                        continue
                    size = max((r.font.size.pt for r in para.runs if r.font.size), default=14)
                    lines = est_lines(text, size, max(w - 0.05, 0.3))
                    total_h += lines * size * 1.32 / 72.0
                    if para.space_after is not None:
                        total_h += para.space_after.pt / 72.0
                if total_h > h + 0.06:
                    problems += 1
                    print("[overflow] p%d %.2fx%.2f needs %.2f | %s"
                          % (idx, w, h, total_h, text_of(shape)[:60].replace("\n", " / ")))
                boxes.append((x, y, w, h, text_of(shape)[:22].replace("\n", " ")))
            if x < -0.02 or y < -0.02 or x + w > sw + 0.02 or y + h > sh + 0.02:
                problems += 1
                print("[bounds] p%d x=%.2f y=%.2f w=%.2f h=%.2f | %s"
                      % (idx, x, y, w, h, str(shape.shape_type)))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ax, ay, aw, ah, at = boxes[i]
                bx, by, bw, bh, bt = boxes[j]
                ox = max(0, min(ax + aw, bx + bw) - max(ax, bx))
                oy = max(0, min(ay + ah, by + bh) - max(ay, by))
                if ox > 0.25 and oy > 0.18:
                    problems += 1
                    print("[overlap] p%d (%.2f,%.2f) x (%.2f,%.2f) = %.2fx%.2f | %s <> %s"
                          % (idx, ax, ay, bx, by, ox, oy, at, bt))
    print("TOTAL issues=%d" % problems)
    return 1 if problems else 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("答辩PPT-1.5.pptx")
    sys.exit(main(str(target)))