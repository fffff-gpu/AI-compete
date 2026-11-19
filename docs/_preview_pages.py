import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _md_pdf

src, outdir, pages = Path(sys.argv[1]), Path(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")]
md = re.sub(r"<!--.*?-->", "", src.read_text(encoding="utf-8"), flags=re.S)
pager = _md_pdf.Pager(sys.argv[4] if len(sys.argv) > 4 else src.stem)
for block in _md_pdf.parse(md):
    pager.emit(block)
outdir.mkdir(parents=True, exist_ok=True)
for n in pages:
    fig = pager.pages[n - 1]
    fig.set_dpi(110)
    target = outdir / f"{src.stem}-p{n}.png"
    fig.savefig(target)
    print("wrote", target, "of", len(pager.pages))
