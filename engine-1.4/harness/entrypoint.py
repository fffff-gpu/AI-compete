"""统一入口：平台不给任何参数也要能把闭环跑完。

    python -m harness.entrypoint                      # 自动发现并串跑 A/B/C
    python -m harness.entrypoint --profiles A,B --rounds 10
    python -m harness.entrypoint --selfcheck
    python -m harness.entrypoint --data /data --out /runs

为什么要这一层（都是 0 分风险，不是分数风险）：

- 赛规只说官方数据包「上传至指定平台固定目录下」，没说目录名、没说入口命令、
  没说父检查点放哪。规则第 69 行的门槛①是「无人工干预地完成**全部阶段**并按时输出」，
  而 `harness.run` 的 `--profile`/`--out` 都是必填、一个进程只跑一个 profile：
  平台不传参 → 只跑 A；平台传自己的参数 → argparse 退出码 2 → 全部为 0。
- 所以这里：`parse_known_args`（未知参数警告后忽略）、`--profile`/`--out` 可选、
  扫 `ZHIXUN_DATA`/`/data`/工程根识别包（认 baseline-info.json 的 dataset_key，不认目录名）、
  **每个 profile 起一个子进程**（官方 `baselines` 模块名相同，同进程混跑会串味）、
  单个 profile 崩了记录错误继续下一个、离线时默认 rule 臂（0 token，不碰网络）。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import ROOT, baseline, provenance
from .journal import write_json

DEFAULT_ROUNDS = 10
DEFAULT_MAX_SECONDS = 1800          # 单 profile 墙钟上限：本机 10 轮实测 A 28s/B 39s/C 117s


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".zhixun-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def default_out() -> Path:
    """输出目录：环境变量 > 镜像里的 /runs 约定（仅 POSIX）> 工程内 ./runs > 临时目录。

    Windows 上 `/runs` 会解析成当前盘根（D:\\runs），所以非 POSIX 不把它当默认候选。
    """
    for raw in baseline._env_paths(("ZHIXUN_OUT", "OUTPUT_DIR", "RUNS_DIR")):
        if _writable(raw):
            return raw
    candidates = ([Path("/runs")] if os.name != "nt" else []) + [ROOT / "runs"]
    candidates.append(Path(os.environ.get("TMPDIR") or os.environ.get("TEMP") or "/tmp"))
    for candidate in candidates:
        try:
            if _writable(Path(candidate)):
                return Path(candidate)
        except OSError:
            continue
    return ROOT / "runs"


def selfcheck(profiles=None) -> dict:
    """把「平台会怎么看待这个镜像」的判定依据全部打印出来：不训练、不联网。"""
    report = {"ok": True, "python": sys.version.split()[0], "workspace_roots": [],
              "packages": {}, "parents": {}, "errors": []}
    try:
        import numpy as np

        report["numpy"] = np.__version__
    except ImportError as exc:
        report["ok"] = False
        report["errors"].append(f"numpy 不可用：{exc}")
    report["workspace_roots"] = [str(root) for root in baseline.data_roots()]
    report["recipe_env_defaults"] = {key: os.environ.get(key, "(unset)") for key in baseline.RECIPE_ENV}
    out = default_out()
    report["out_dir"] = str(out)
    report["out_writable"] = _writable(out)
    wanted = profiles or list(baseline.PROFILES)
    discovered = baseline.discover_packages()
    for profile in wanted:
        try:
            package = baseline.package(profile)
            cfg = baseline.config(profile)
            ckpt, source = baseline.parent_checkpoint(profile)
            meta = baseline.checkpoint_meta(ckpt) or {}
            val = baseline.dataset(profile, "validation")
            report["packages"][profile] = {
                "package": str(package), "task": cfg.get("task"), "hidden": cfg.get("hidden"),
                "config_seed": cfg.get("seed"), "parent": str(ckpt), "parent_source": source,
                "parent_steps": meta.get("steps"), "parent_epochs": meta.get("epochs"),
                "parent_profile": meta.get("profile"), "rows_validation": len(val["y"]),
                "continuation_compatible": (meta.get("profile") == profile
                                            and meta.get("hidden") == cfg.get("hidden")
                                            and meta.get("seed") == cfg.get("seed")),
            }
        except Exception as exc:                          # noqa: BLE001 自检要把失败原样报出来
            report["ok"] = report["ok"] and profile not in discovered
            report["packages"][profile] = {"error": f"{type(exc).__name__}: {exc}"}
            report["errors"].append(f"{profile}: {type(exc).__name__}: {exc}")
    return report


def run_profile(profile: str, out: Path, rounds: int, extra: list, max_seconds: int) -> dict:
    """在一个子进程里跑完一个 profile；返回状态记录（失败不抛，保证「全部阶段」跑到底）。"""
    started = time.monotonic()
    run_dir = out / f"{profile.lower()}_{rounds}r"
    cmd = [sys.executable, "-m", "harness.run", "--profile", profile,
           "--rounds", str(rounds), "--out", str(run_dir), "--fresh"] + list(extra)
    env = dict(os.environ)
    env.setdefault("ZHIXUN_WORKSPACE", str(baseline.WORKSPACE))
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, timeout=max_seconds,
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        status, tail = ("ok" if proc.returncode == 0 else f"exit={proc.returncode}"), \
                       (proc.stdout or "")[-2000:] + (proc.stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        status, tail = "timeout", f">{max_seconds}s 墙钟上限，已终止"
    except OSError as exc:
        status, tail = f"launch_error: {type(exc).__name__}", str(exc)
    record = {"profile": profile, "status": status, "seconds": round(time.monotonic() - started, 1),
              "run_dir": str(run_dir), "command": cmd[-6:]}
    results = run_dir / "results.json"
    if results.is_file():
        try:
            payload = json.loads(results.read_text(encoding="utf-8"))
            record.update(baseline=payload.get("baseline"), best=payload.get("best"),
                          accepted_rounds=payload.get("accepted_rounds"),
                          worst_stage_gain=payload.get("worst_stage_gain"),
                          gate=payload.get("gate"), ledger_cost_units=(payload.get("ledger") or {}).get("cost_units"))
            src = run_dir / "best" / baseline.ARTIFACT_NAME
            if src.is_file():                             # 交付约定：产物就是 checkpoint.npz，字段不变
                art = baseline.artifact_dir() or run_dir
                dst = Path(art) / profile
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst / baseline.ARTIFACT_NAME)
                record["artifact"] = str(dst / baseline.ARTIFACT_NAME)
        except (OSError, ValueError) as exc:
            record["results_error"] = str(exc)
    else:
        record["output_tail"] = tail
    return record


def boot_banner(profiles: list, out: Path) -> None:
    """把"平台到底给了我们什么"无条件打到 stdout：封闭评测的唯一远程诊断通道。

    为什么必须有：评测在指定设备的离线环境里跑，我们看不到现场；一旦运行失败，
    能拿到的只有容器 stdout。没有这段，报错只会是一句 FileNotFoundError，
    既看不出挂载点长什么样、也看不出父检查点放在哪——下一次还是猜。
    """
    env_keys = ("ZHIXUN_DATA", "ZHIXUN_DATA_STRICT", "ZHIXUN_WORKSPACE", "ZHIXUN_OUT",
                "ZHIXUN_CKPT", "CHECKPOINT_PATH", "MODEL_PATH", "ZHIXUN_ARTIFACT_DIR",
                "OUTPUT_DIR", "RUNS_DIR", "PYTRIO_BASE_URL", "SWANLAB_PROJECT")
    # 每一行都先给 ASCII 的 key=value，中文放在后面。原因：这段文本的唯一用途是被原样贴回来，
    # 而它要穿过容器日志、平台网页、PowerShell 5.1 控制台三层解码器——本机实测最后那层会把
    # UTF-8 按 GBK 解成「杩愯幆澧冭嚜帮」这种乱码。关键字保住，乱码就不致命。
    print("[entrypoint] ==== boot_banner 运行环境自述（失败时请把这一段发回来）====", flush=True)
    print(f"[entrypoint] python={sys.version.split()[0]} platform={os.name} "
          f"cwd={Path.cwd()} argv={sys.argv[1:]}")
    try:
        import numpy as np

        print(f"[entrypoint] numpy={np.__version__}")
    except ImportError as exc:
        print(f"[entrypoint] numpy=MISSING 依赖缺失：{exc}")
    for key in env_keys:
        value = os.environ.get(key)
        if value is not None and "KEY" not in key:
            print(f"[entrypoint] env {key}={value}")
    roots = baseline.data_roots()
    print(f"[entrypoint] data_roots={[str(r) for r in roots]}  （候选数据根；空列表=一个都不存在）")
    for root in roots[:4]:
        try:
            entries = sorted((p.name + ("/" if p.is_dir() else "")) for p in root.iterdir())[:40]
            print(f"[entrypoint] ls {root} n={len(entries)}: {entries}")
        except OSError as exc:
            print(f"[entrypoint] ls {root} FAILED 无法列出：{exc}")
    for profile in profiles:
        try:
            package = baseline.package(profile)
            train_file, val_file = baseline.data_files(profile)
            ckpt, source = baseline.parent_checkpoint(profile)
            print(f"[entrypoint]   profile={profile} package={package} train={train_file.name} "
                  f"parent={ckpt} parent_source={source} "
                  f"package_source={baseline.package_reason(profile)}")
            # 这两行是为平台预检那类"两个版本的官方训练器"准备的：报上来就知道现场执行的到底是
            # 哪一份 train.py（A/C 包发的 77043a0a… 对 B 一律要标签来源声明；B 包自己发的
            # 6257f0b1… 只在 classification 时要）。没有这两行，下一轮还是只能靠猜。
            tp = baseline.source_dir(profile) / "baselines" / "train.py"
            print(f"[entrypoint]   profile={profile} train_py={tp} "
                  f"train_py_sha256={provenance.sha256_file(tp)[:24] if tp.exists() else 'MISSING'}")
            pv = provenance.preview(profile, package)
            print(f"[entrypoint]   profile={profile} label_provenance required={pv['required']} "
                  f"kind={pv['kind']} status={pv['status']} "
                  f"official_label_status={pv['official_label_status']} "
                  f"citable={pv['citable']} complete={pv['complete']}")
        except Exception as exc:                          # noqa: BLE001 诊断要把失败原样打出来
            print(f"[entrypoint]   profile={profile} ERROR {type(exc).__name__}: {exc}")
    print(f"[entrypoint] out_dir={out} writable={_writable(out)}  （输出目录）")
    art = baseline.artifact_dir()
    print(f"[entrypoint] artifact_dir={art if art else '(unset, uses run dir)'}  （产物目录）")
    print("[entrypoint] ================================================", flush=True)


def as_command(argv: list) -> list:
    """判断平台是不是把命令当参数传进来了（docker run <image> python -m harness.predict …）。

    这是必须的：镜像的 ENTRYPOINT 是 `python -m harness.entrypoint`，任何追加在后面的
    命令都会被 argparse 当成"未知参数"忽略掉，于是我们要跑闭环、平台要的是预测，
    双方都拿不到自己要的东西——表现正是"镜像可用但评测运行失败"。
    规则：第一个参数不是 - 开头，且像可执行文件名/模块名，就当命令原样执行。
    """
    if not argv:
        return []
    first = argv[0]
    if first.startswith("-"):
        return []
    if first in ("python", "python3", "sh", "bash", "/bin/sh", "-c"):
        return list(argv)
    if "/" in first or "\\" in first or first.endswith((".py", ".sh")):
        return list(argv)
    if first in ("predict", "run", "audit", "entrypoint", "describe"):
        return [sys.executable, "-m", f"harness.{first}"] + list(argv[1:])
    return []


def main(argv=None) -> int:
    incoming = list(sys.argv[1:] if argv is None else argv)
    # 入口声明：平台可能用 `--describe` / `describe` / 环境变量来要求"原生描述"。
    # 这三个写法都在跑任何训练之前短路，保证描述检查不会因为找不到数据而失败。
    if incoming and incoming[0] in ("describe", "--describe", "-d"):
        from . import describe as describe_mod
        text = json.dumps(describe_mod.build(), ensure_ascii=False, indent=2)
        print(text)
        return 0
    if os.environ.get("CONTEST_DESCRIBE", "").lower() in ("1", "true", "yes", "on"):
        from . import describe as describe_mod
        print(json.dumps(describe_mod.build(), ensure_ascii=False, indent=2))
        return 0
    if incoming and incoming[0] in ("--help", "-h", "help"):
        # 平台/人工都可能用 `docker run <image> --help` 看这镜像到底是什么。
        # 这里给的是人能读的同一份原生描述，而不是 argparse 的英文参数表。
        from . import describe as describe_mod
        print(describe_mod.render_markdown(describe_mod.build()))
        return 0
    # 启动 banner：无论走哪条路径都先自报家门，平台日志里能直接看到版本与描述入口。
    from .describe import _label_version
    print(f"[zhixun-harness] v{_label_version()} | entrypoint=harness.entrypoint | "
          f"describe: --describe | predict: python -m harness.predict", file=sys.stderr, flush=True)
    command = as_command(incoming)
    if command:
        # 平台追加的是命令而不是开关：原样执行，别当未知参数吞掉
        print(f"[entrypoint] 检测到外部命令，直接执行：{command}", file=sys.stderr, flush=True)
        return subprocess.call(command, cwd=str(ROOT))

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profiles", default=None, help="逗号分隔，如 A,B,C；默认=发现到的全部")
    parser.add_argument("--profile", choices=list(baseline.PROFILES), default=None,
                        help="只跑一个 profile（等价于 --profiles=X）")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--out", default=None)
    parser.add_argument("--data", default=None, help="数据包父目录（也可用 ZHIXUN_DATA）")
    parser.add_argument("--parent-ckpt", default=None, help="平台父检查点路径；不给则自动发现")
    parser.add_argument("--label-provenance", default=None,
                        help="B 的标签来源说明 JSON（官方 README 约定的 labels-provenance.json）；"
                             "不给则读 ZHIXUN_LABEL_PROVENANCE，再不给就从官方包字段转述生成")
    parser.add_argument("--policy", default="rule", choices=["rule", "llm", "auto"])
    parser.add_argument("--gate", default="v1", choices=["v1", "v2"])
    parser.add_argument("--control", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--swanlab", default="offline",
                        choices=["offline", "online", "local", "disabled", "off"],
                        help="实验记录模式，转发给 harness.run（默认 offline：本地优先）")
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--json", action="store_true")
    args, unknown = parser.parse_known_args(incoming)
    if unknown:
        print(f"[entrypoint] 忽略未知参数：{unknown}", file=sys.stderr)

    if args.data:
        os.environ["ZHIXUN_DATA"] = args.data
        baseline.discover_packages(refresh=True)

    forward = ["--policy", args.policy]
    if args.gate != "v1":
        forward += ["--gate", args.gate]          # v1 是 harness.run 的默认，不重复传
    if args.control in ("auto", "on"):
        forward.append("--control")               # 控制器只省算力不改精度（b_w2_nc 已验证同结果）
    if args.seed is not None:
        forward += ["--seed", str(args.seed)]
    if args.swanlab != "offline":
        forward += ["--swanlab", args.swanlab]
    if args.parent_ckpt:
        forward += ["--parent-ckpt", args.parent_ckpt]
    if args.label_provenance:
        forward += ["--label-provenance", args.label_provenance]

    wanted = ([args.profile] if args.profile
              else [p.strip().upper() for p in args.profiles.split(",")] if args.profiles
              else list(baseline.PROFILES))
    wanted = [p for p in wanted if p in baseline.PROFILES]

    if args.selfcheck:
        report = selfcheck(wanted)
        print(json.dumps(report, ensure_ascii=False, indent=2) if args.json
              else json.dumps(report, ensure_ascii=False))
        return 0 if report["ok"] else 1

    out = Path(args.out) if args.out else default_out()
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:                                # 输出目录不可写也要先把布局打出来
        print(f"[entrypoint] 输出目录不可用 {out}: {exc}", file=sys.stderr, flush=True)
    boot_banner(wanted, out)
    discovered = baseline.discover_packages()

    def has_data(profile: str) -> bool:
        try:
            baseline.data_files(profile)
            return True
        except Exception:                                 # noqa: BLE001 诊断已在 boot_banner 里
            return False

    runnable = [p for p in wanted
                if p in discovered or baseline.PACKAGES[p].is_dir() or has_data(p)]
    skipped = [p for p in wanted if p not in runnable]
    summary = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "profiles_requested": wanted,
               "profiles_run": runnable, "profiles_skipped": skipped, "rounds": args.rounds,
               "out": str(out), "runs": []}
    for profile in runnable:
        print(f"[entrypoint] === {profile}：续训闭环 {args.rounds} 轮 ===", flush=True)
        record = run_profile(profile, out, args.rounds, forward, args.max_seconds)
        summary["runs"].append(record)
        print(f"[entrypoint] {profile}: {record['status']} "
              f"best={record.get('best')} accepted={record.get('accepted_rounds')} "
              f"worst_stage_gain={record.get('worst_stage_gain')} {record['seconds']}s", flush=True)
    summary["ok"] = bool(runnable) and all(run["status"] == "ok" and "best" in run
                                           for run in summary["runs"])
    write_json(out / "entrypoint_report.json", summary)
    print(("[entrypoint] 全部 profile 完成" if summary["ok"]
           else "[entrypoint] 有 profile 未完成，见 entrypoint_report.json"), flush=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
