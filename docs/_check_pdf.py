"""Render every page and flag content outside the printable box."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image
import _md_pdf

pdf = Path(sys.argv[1])
pages = int(sys.argv[2]) if len(sys.argv) > 2 else 0
# use the renderer's own pager to get figures
import re
src = Path(sys.argv[3])
md = re.sub(r"<!--.*?-->", "", src.read_text(encoding="utf-8"), flags=re.S)
pager = _md_pdf.Pager(src.stem)
for block in _md_pdf.parse(md):
    pager.emit(block)
DPI = 150
left, right = _md_pdf.ML * DPI - 6, (_md_pdf.PAGE_W - _md_pdf.MR) * DPI + 6
top, bottom = 0.25 * DPI, (_md_pdf.PAGE_H - 0.25) * DPI
bad = 0
for i, fig in enumerate(pager.pages, 1):
    fig.set_dpi(DPI)
    tmp = Path("docs/_preview_pdf") / f"_p{i}.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(tmp, dpi=DPI)
    a = np.array(Image.open(tmp).convert("L"))
    h, w = a.shape
    rows = [y for y in range(h) if (a[y] < 150).any()]
    cols = [x for x in range(w) if (a[:, x] < 150).any()]
    if not rows:
        continue
    issues = []
    if min(cols) < left:
        issues.append(f"left {min(cols)}<{left:.0f}")
    if max(cols) > right:
        issues.append(f"right {max(cols)}>{right:.0f}")
    if min(rows) < top:
        issues.append(f"top {min(rows)}<{top:.0f}")
    if max(rows) > bottom:
        issues.append(f"bottom {max(rows)}>{bottom:.0f}")
    if issues:
        bad += 1
        print(f"page {i}: " + ", ".join(issues))
print(f"pages={len(pager.pages)} out_of_box={bad}")
