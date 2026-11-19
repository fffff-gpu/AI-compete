"""把本地 SwanLab 记录逐条同步到云端，并收集 run 链接。

    python scripts/sync_records.py                 # 同步所有还没上云的记录
    python scripts/sync_records.py --only a-a_10r  # 只同步指定实验名
    python scripts/sync_records.py --status        # 只看现状

链接落在 `swanlog/cloud_links.json`：实验名 → {url, run_dir, source_run, rows}。
这个文件是提交材料里"云端实验记录链接或查询依据"的来源，也是评审能逐条核对的入口，
所以同步结果必须落盘、必须可重跑（已同步过的跳过，不重复产生云端副本）。

前提：`swanlab login` 已在本机完成（凭据在你的用户目录，本脚本不接触 key）。
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINKS = ROOT / "swanlog" / "cloud_links.json"
URL_RE = re.compile(r"https://swanlab\.cn/@[^\s/]+/[^/\s]+/runs/[A-Za-z0-9]+")


def local_records() -> dict:
    """实验名 → 离线目录：先实时记录（run 的 _swanlog 里最新那个），再回填索引。"""
    found = {}
    index_path = ROOT / "swanlog" / "backfill_index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for name, meta in index.items():
            if meta.get("run_dir"):
                found[name] = {"run_dir": meta["run_dir"], "source_run": meta["source"],
                               "rows": meta.get("rows"), "backfilled": not meta.get("skipped")}
    for entry in sorted((ROOT / "runs").iterdir()):
        if entry.name.startswith("_"):        # _rec_/_smoke_/_parity_/_probe 这类一次性目录不上云
            continue
        inner = sorted((entry / "_swanlog").glob("run-*")) if (entry / "_swanlog").is_dir() else []
        if not inner:
            continue
        name = f"{entry.name[:1].lower()}-{entry.name}"
        profile = None
        try:
            profile = json.loads((entry / "results.json").read_text(encoding="utf-8")).get("profile")
        except (OSError, ValueError):
            pass
        if profile:
            name = f"{profile.lower()}-{entry.name}"
        found[name] = {"run_dir": str(inner[-1]), "source_run": str(entry),
                       "rows": None, "backfilled": False}
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="只同步名字包含该串的实验")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--force", action="store_true", help="已记录链接的也重推")
    args = parser.parse_args(argv)

    records = local_records()
    links = json.loads(LINKS.read_text(encoding="utf-8")) if LINKS.is_file() else {}
    todo = []
    for name, meta in sorted(records.items()):
        if args.only and args.only not in name:
            continue
        if name in links and not args.force:
            continue
        todo.append((name, meta))

    print(f"本地记录 {len(records)} 条，已上云 {len(links)} 条，待同步 {len(todo)} 条")
    if args.status:
        for name, meta in sorted(records.items()):
            mark = links.get(name, {}).get("url", "—")
            print(f"  {name:22s} {'回填' if meta['backfilled'] else '实时'}  {meta['run_dir'][-34:]}  {mark}")
        return 0
    if not todo:
        print("没有需要同步的记录")
        return 0

    failures = []
    for name, meta in todo:
        proc = subprocess.run([sys.executable, "-m", "swanlab", "sync", meta["run_dir"]],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        blob = (proc.stdout or "") + (proc.stderr or "")
        urls = URL_RE.findall(blob)
        if proc.returncode == 0 and urls:
            links[name] = {"url": urls[-1], "run_dir": meta["run_dir"],
                           "source_run": meta["source_run"], "backfilled": meta["backfilled"]}
            LINKS.parent.mkdir(exist_ok=True)
            LINKS.write_text(json.dumps(links, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"  ✓ {name:22s} {urls[-1]}")
        else:
            tail = " | ".join(x for x in blob.strip().splitlines()[-3:] if x)
            failures.append((name, tail))
            print(f"  ✗ {name:22s} rc={proc.returncode} {tail[:160]}")

    LINKS.write_text(json.dumps(links, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n成功 {len(links)} / {len(records)}；失败 {len(failures)}")
    for name, detail in failures:
        print(f"  - {name}: {detail[:200]}")
    print(f"链接清单 → {LINKS}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
