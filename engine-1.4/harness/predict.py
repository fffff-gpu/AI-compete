"""独立预测入口（平台公告 11-13 明写镜像须「提供独立预测入口」）。

    python -m harness.predict                       # 从 stdin 读请求，结果写 stdout
    python -m harness.predict --data rows.jsonl -o out.jsonl
    python -m harness.predict --checkpoint /data/parent/checkpoint.npz --profile A
    python -m harness.predict --selfcheck

请求格式沿用官方 baselines/predict.py：每行 {"task": ..., "inputs": {...}}，
输出 {"outputs": ...}。本模块不重新实现任何预测逻辑——schema 校验、特征展平、
softmax/回归还原全部走官方代码，只是多了三件事：

1. **不给参数也能跑**：检查点从 `--parent-ckpt`/环境变量/产物目录/数据根里发现；
   profile 从 checkpoint 的 metadata 读，不靠目录名猜。
2. **只吃 inputs，绝不看标签**：批量模式里 `answer`/`meta`/`id` 一律丢弃，
   所以这个入口不可能泄漏隐藏标签（红线 A 类）。
3. **容忍未知参数**：平台若追加自己的命令行开关，这里警告后忽略，而不是 argparse
   退出码 2 把整场评测搞挂。
"""
import argparse
import json
import os
import sys
from pathlib import Path

from . import ROOT, baseline


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _inputs_of(record: dict):
    """取出一条记录的 inputs：既支持预测请求，也支持数据集行——只看 inputs，标签一律丢弃。"""
    if "inputs" not in record:
        raise ValueError("记录缺少 inputs 字段（预测请求只需 task 与 inputs）")
    return record["inputs"]


# 平台把「这次预测用什么模型、输入在哪」写成上下文交给容器；名字按平台文档与常见约定都收。
CONTEXT_ENVS = ("CONTEST_PREDICTION_CONTEXT", "CONTEST_CONTEXT", "PREDICTION_CONTEXT")
ARTIFACT_ENVS = ("ARTIFACT_PATH", "CONTEST_ARTIFACT_PATH", "ZHIXUN_ARTIFACT_DIR",
                 "ARTIFACT_DIR", "RESULT_DIR", "SUBMISSION_DIR")
BUNDLED = ROOT / "artifacts"


def read_context() -> dict:
    """读平台给的预测上下文：JSON 文件 / 目录 / 内联 JSON 都认；读不到就返回空 dict。

    目录形态下会顺手找 checkpoint.npz 与 inputs.jsonl，这样平台只给一个目录也能跑。
    """
    for key in CONTEXT_ENVS:
        raw = os.environ.get(key)
        if not raw:
            continue
        path = Path(raw)
        if path.is_file():
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except Exception:                                  # noqa: BLE001 上下文坏掉不该让预测整场挂
                continue
            if isinstance(obj, dict):
                obj.setdefault("_source", f"env:{key}:{path}")
                return obj
        if path.is_dir():
            info = {"_source": f"env:{key}:{path}"}
            for name in ("context.json", "prediction.json", "task.json"):
                cand = path / name
                if cand.is_file():
                    try:
                        loaded = json.loads(cand.read_text(encoding="utf-8"))
                    except Exception:                          # noqa: BLE001
                        loaded = None
                    if isinstance(loaded, dict):
                        info.update(loaded)
            for name in ("inputs.jsonl", "rows.jsonl", "public.jsonl"):
                cand = path / name
                if cand.is_file():
                    info.setdefault("inputs", str(cand))
                    break
            for name in (baseline.ARTIFACT_NAME, "model.npz"):
                cand = path / name
                if cand.is_file():
                    info.setdefault("checkpoint", str(cand))
                    break
            return info
        try:
            obj = json.loads(raw)
        except Exception:                                     # noqa: BLE001
            continue
        if isinstance(obj, dict):
            obj.setdefault("_source", f"env:{key}:inline")
            return obj
    return {}


def _artifact_hits():
    """平台给的产物路径（文件或目录）→ checkpoint 候选。

    目录形态有三种可能，都要认：直接放着 checkpoint.npz；按 profile 分目录
    （artifacts/B/checkpoint.npz，镜像自带的就是这种）；或者再套一层任意子目录。
    """
    hits = []
    for raw in (os.environ.get(key) for key in ARTIFACT_ENVS):
        if not raw:
            continue
        for chunk in raw.split(os.pathsep):
            chunk = chunk.strip().strip('"').strip("'")
            if not chunk:
                continue
            path = Path(chunk)
            if path.is_file():
                hits.append(path)
                continue
            if not path.is_dir():
                continue
            direct = path / baseline.ARTIFACT_NAME
            if direct.is_file():
                hits.append(direct)
                continue
            for prof in baseline.PROFILES:
                cand = path / prof / baseline.ARTIFACT_NAME
                if cand.is_file():
                    hits.append(cand)
            for cand in sorted(path.glob("*/" + baseline.ARTIFACT_NAME)):
                if cand.is_file() and cand not in hits:
                    hits.append(cand)
    return hits


def _bundled(profile=None):
    """镜像内自带的交付检查点：/app/artifacts/<P>/checkpoint.npz。"""
    profiles = [profile] if profile else list(baseline.PROFILES)
    for prof in profiles:
        cand = BUNDLED / str(prof).upper() / baseline.ARTIFACT_NAME
        if cand.is_file():
            return cand, "bundled:" + str(prof).upper()
    return None


def resolve_checkpoint(explicit=None, profile=None, context=None):
    """定位要用的检查点：显式 > 平台上下文 > 产物环境变量 > 自带产物 > 父检查点发现 > 包内基线。"""
    context = context or {}
    if explicit:
        path = Path(explicit)
        if path.is_dir():
            path = path / baseline.ARTIFACT_NAME
        if not path.is_file():
            raise FileNotFoundError(f"检查点不存在：{explicit}")
        return path, "cli"
    for key in ("checkpoint", "artifact", "model", "checkpoint_path", "artifact_path"):
        raw = context.get(key)
        if isinstance(raw, str) and raw:
            path = Path(raw)
            if path.is_dir():
                path = path / baseline.ARTIFACT_NAME
            if path.is_file():
                return path, "context:" + key
    env_hits = _artifact_hits()
    prof_hint = str(context.get("profile") or context.get("task") or profile or "").upper()[:1]
    if env_hits:
        if prof_hint:
            for hit in env_hits:
                if hit.parent.name.upper() == prof_hint:
                    return hit, "env:artifact"
            for hit in env_hits:
                try:
                    meta = baseline.checkpoint_meta(hit) or {}
                except Exception:                             # noqa: BLE001 读不了就跳过
                    meta = {}
                if str(meta.get("profile", "")).upper() == prof_hint:
                    return hit, "env:artifact"
        return env_hits[0], "env:artifact"
    bundled = _bundled(prof_hint or None)
    if bundled:
        return bundled
    if profile:
        return baseline.parent_checkpoint(profile, explicit=None)
    hits = [hit for hit in baseline.discover_parent_checkpoints() if hit["path"].is_file()]
    if hits:
        best = max(hits, key=lambda hit: hit["path"].stat().st_mtime)
        return best["path"], best["profile"] + ":" + best["sha256"][:8]
    for prof, directory in sorted(baseline.discover_packages().items()):
        own = directory / baseline.ARTIFACT_NAME
        if own.is_file():
            return own, f"package:{prof}"
    raise FileNotFoundError(
        f"没找到任何 checkpoint.npz。搜过的数据根：{[str(r) for r in baseline.data_roots()]}。"
        f"请显式传 --checkpoint，或设 ARTIFACT_PATH=/path/to/checkpoint.npz。")


def run(records, checkpoint=None, profile=None, quiet=False, context=None):
    """对 records（可迭代 dict）逐条预测，产出官方格式的响应 dict。"""
    ctx = read_context() if context is None else context
    ckpt, source = resolve_checkpoint(checkpoint, profile, ctx)
    meta = baseline.checkpoint_meta(ckpt) or {}
    prof = str(meta.get("profile", profile or "")).upper()
    if prof not in baseline.PROFILES:
        raise ValueError(f"检查点 metadata 里没有可用的 profile：{ckpt}（{meta.get('profile')!r}）")
    predict_mod = baseline.predict_api(prof)
    model = baseline.load_model(prof, ckpt)
    if not quiet:
        print(f"[predict] checkpoint={ckpt} profile={prof} source={source} "
              f"task={meta.get('task')} steps={meta.get('steps')}", file=sys.stderr)
    for record in records:
        yield predict_mod.predict(model, _inputs_of(record))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", dest="checkpoint", default=None,
                        help="要用的 checkpoint.npz（省略则自动发现：ZHIXUN_CKPT/产物目录/数据根）")
    parser.add_argument("--profile", choices=list(baseline.PROFILES), default=None)
    parser.add_argument("--data", "-i", dest="data", default=None,
                        help="批量输入 JSONL；省略则从 stdin 逐行读")
    parser.add_argument("--output", "-o", dest="output", default=None,
                        help="批量输出 JSONL；省略则写 stdout")
    parser.add_argument("--limit", type=int, default=0, help="最多预测多少条（调试用）")
    parser.add_argument("--selfcheck", action="store_true",
                        help="只报告能不能找到检查点并加载，不预测")
    parser.add_argument("--describe", action="store_true",
                        help="输出镜像原生描述（等价 python -m harness.describe --json）")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"[predict] 忽略未知参数：{unknown}", file=sys.stderr)
    if args.describe:
        from . import describe as describe_mod
        print(json.dumps(describe_mod.build(), ensure_ascii=False, indent=2))
        return 0

    context = read_context()
    if context:
        print(f"[predict] 上下文：{context.get('_source', 'unknown')}", file=sys.stderr)
    checkpoint = args.checkpoint or (context.get("checkpoint") if isinstance(context.get("checkpoint"), str) else None)
    profile = args.profile or str(context.get("task") or context.get("profile") or "")[:1] or None
    data = args.data or (context.get("inputs") if isinstance(context.get("inputs"), str) else None)

    if args.selfcheck:
        try:
            ckpt, source = resolve_checkpoint(checkpoint, profile, context)
            meta = baseline.checkpoint_meta(ckpt) or {}
            prof = str(meta.get("profile", "")).upper()
            baseline.predict_api(prof)
            baseline.load_model(prof, ckpt)
            print(json.dumps({"ok": True, "checkpoint": str(ckpt), "source": source,
                              "profile": prof, "task": meta.get("task"),
                              "features": len(meta.get("paths", [])),
                              "steps": meta.get("steps"), "epochs": meta.get("epochs"),
                              "context": context.get("_source")},
                             ensure_ascii=False))
            return 0
        except Exception as exc:                              # noqa: BLE001 自检要报告一切失败
            print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                             ensure_ascii=False))
            return 1

    source = _read_jsonl(data) if data else (json.loads(line) for line in sys.stdin
                                             if line.strip())
    out = Path(args.output).open("w", encoding="utf-8") if args.output else sys.stdout
    # 平台用 CONTEST_PREDICTION_CONTEXT 把输入文件塞给我们时，往往既读 stdout 也看目录里
    # 有没有结果文件——两边都给，省掉"猜平台怎么收结果"这一轮返工。
    tee = None
    if not args.output and isinstance(data, str) and context.get("inputs"):
        try:
            tee = (Path(data).parent / "predictions.jsonl").open("w", encoding="utf-8")
            print(f"[predict] 同时写 {tee.name}（{Path(data).parent}）", file=sys.stderr)
        except OSError as exc:                                # noqa: BLE001 写不了就只走 stdout
            print(f"[predict] 结果文件不可写，仅 stdout：{exc}", file=sys.stderr)
            tee = None
    n = 0
    try:
        for response in run(source, checkpoint, profile, context=context):
            line = json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n"
            out.write(line)
            if tee is not None:
                tee.write(line)
            n += 1
            if args.limit and n >= args.limit:
                break
    finally:
        if args.output:
            out.close()
        if tee is not None:
            tee.close()
    if not args.output:
        sys.stderr.write(f"[predict] {n} 条\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
