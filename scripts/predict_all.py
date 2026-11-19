"""把某个数据包的**全部公开输入**跑一遍预测，验证「预测检查」不会中途失败。

平台的预测检查失败时只给一句「未能完成全部公开输入预测」，不说哪一条。
这个脚本就是本地版的那项检查：逐条走与 `python -m harness.predict` **完全相同**的
代码路径（同一套 resolve_checkpoint / 官方 load_model / 官方 predict API），
统计成功条数、耗时、以及第一处报错的位置和内容。

为什么在进程内跑而不是 `subprocess`：本机沙箱禁止子进程管道
（`PermissionError: [WinError 5]` on `_winapi.CreatePipe`），进程内等价，
且能精确定位到行号。CLI 的 stdin 通路另有 `scripts/predict_cli_check.sh` 验证。

用法（在 official/ 目录下）：
    python scripts/predict_all.py --profile B --data "../人工设计工业模型/任务B/package"
    python scripts/predict_all.py --profile C --data "../人工设计工业模型/任务C/package" --limit 200
    python scripts/predict_all.py --profile A --data "../人工设计工业模型/任务A/package"

退出码：0 全部成功；1 有失败；2 前置条件缺失。
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from harness import baseline  # noqa: E402
from harness import predict as predict_mod  # noqa: E402


def rows_of(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description="把全部公开输入跑一遍预测")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--data", required=True, help="数据包目录（含 public/validation.jsonl）")
    ap.add_argument("--artifact", default=None, help="检查点路径；默认 artifacts/<P>/checkpoint.npz")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 = 全部）")
    args = ap.parse_args(argv)

    profile = args.profile.upper()
    data = Path(args.data)
    if not data.is_absolute():
        data = (Path.cwd() / data).resolve()
    source = data / "public" / "validation.jsonl"
    if not source.is_file():
        print(f"[predict_all] 找不到 {source}")
        return 2

    checkpoint = Path(args.artifact) if args.artifact else (ROOT / "artifacts" / profile / "checkpoint.npz")
    if not checkpoint.is_absolute():
        checkpoint = (Path.cwd() / checkpoint).resolve()

    ctx = {"_source": f"file:{checkpoint}"}
    ckpt, source_kind = predict_mod.resolve_checkpoint(str(checkpoint), profile, ctx)
    meta = baseline.checkpoint_meta(ckpt) or {}
    prof = str(meta.get("profile", "")).upper()
    predict_api = baseline.predict_api(prof)
    model = baseline.load_model(prof, ckpt)
    print(f"[predict_all] checkpoint={ckpt} profile={prof} source={source_kind} "
          f"task={meta.get('task')} steps={meta.get('steps')}", file=sys.stderr)

    started = time.time()
    sent = ok = 0
    first_error = None
    samples = []
    for record in rows_of(source):
        if args.limit and sent >= args.limit:
            break
        sent += 1
        try:
            payload = predict_mod._inputs_of(record)
            out = predict_api.predict(model, payload)
            json.dumps({"outputs": out}, ensure_ascii=False)  # 与 CLI 的序列化口径一致
            ok += 1
            if len(samples) < 5:
                samples.append(out)
        except Exception as exc:  # noqa: BLE001 逐条隔离，定位第一处失败
            first_error = f"第 {sent} 条（id={record.get('id')}）：{type(exc).__name__}: {exc}"
            break

    elapsed = time.time() - started
    print(json.dumps({
        "profile": profile,
        "source": str(source),
        "sent": sent,
        "ok": ok,
        "seconds": round(elapsed, 3),
        "per_row_ms": round(1000.0 * elapsed / max(sent, 1), 3),
        "sample": samples,
        "error": first_error,
    }, ensure_ascii=False, indent=2))
    return 0 if (ok == sent and sent > 0) else 1


if __name__ == "__main__":
    sys.exit(main())