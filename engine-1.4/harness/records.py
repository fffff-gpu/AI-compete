"""实验记录：SwanLab（本地优先）+ 与 journal 同源的本地 CSV/JSONL 兜底。

赛规任务 2 与《实验记录规范》要求「全部训练、微调、调整与评估实验」都完整记录，
提交资料里要「完整的云端实验记录链接或查询依据」，主观分还要「与训练平台日志一致」。
所以这里的顺序是刻意的：

1. **先写本地**（`<run>/swanlab_export/rounds.csv|jsonl`）——记录层是权威副本，
   swanlab 装没装、有没有网、key 对不对，都不影响记录完整性；
2. **再喂 SwanLab**，默认 `mode="offline"`：本地落盘到 `<run>/_swanlog/`，
   等账号/额度到位后用 `python -m swanlab sync <_swanlog/run-...>` 一次上传；
3. SwanLab 的任何异常只记一条 `swanlab_error`，**绝不中断训练**（封闭评测里
   记录层挂了不该把整场评测带崩，那是 0 分风险）。

评测环境是统一离线的（规则第 79 行），所以默认 offline；要直连云端用 `--swanlab online`。
"""
import csv
import json
import os
from pathlib import Path

PROJECT = "university_AI_innovate"
SWANLAB_MODES = ("online", "offline", "local", "disabled")
FIELDS = ("round", "candidate", "source", "state", "epochs", "lr", "wd", "ls", "jitter", "clip",
          "score_before", "score_after", "gain", "accepted", "worst_stage_gain",
          "stage1_gain", "stage2_gain", "stage3_gain", "train_seconds",
          "labels_read", "labels_unique", "tokens_total", "cost_units",
          "steps_initial", "steps_final",
          # v2 的判定依据。以前只有 SwanLab 和 rounds.jsonl 有这四列，CSV 里没有
          # （DictWriter 的 fieldnames 不含它们就被 extrasaction="ignore" 丢了）——
          # 于是"本地看 CSV"和"云端看 SwanLab"结论不一致：CSV 里能看到 gain=+0.01
          # 却看不到 gain_lcb=-0.004，队友照着 CSV 会以为这轮涨了，其实门禁判的是噪声。
          "gain_lcb", "worst_stage_lcb", "zone", "gate",
          "parent_checkpoint_sha256", "checkpoint_sha256", "resample_sha256", "hypothesis")


# 不发成曲线、但**照样写进本地 CSV/JSONL** 的列。
# 只有一列符合"纯冗余"的定义：`round` 本身就是 step（_log 用它当横轴），
# 再画一遍就是一条 45° 直线，占一张图位、不带任何信息。
# 注意：**常数列不算冗余**，别顺手删。`clip` 恒 5、`ls` 恒 0 是"这个旋钮我们没动过"的
# 可视证据；A 的 `worst_stage_gain` 全程 0、C 的 `steps_initial` 全程 6630 更是结论本身
# （三段打不动 / 9 轮被拒所以父亲一步没走）。要精简曲线，应该按"这张图能不能证伪什么"来筛，
# 不是按"它是不是平的"来筛。
NOT_A_SERIES = ("round",)


class ExperimentRecorder:
    """一个 run 一个实例；每轮一条记录，字段与 journal 完全对齐。"""

    def __init__(self, run_dir, *, profile=None, mode="offline", project=None,
                 config=None, enabled=True, name=None):
        self.run_dir = Path(run_dir)
        self.profile = profile
        raw = (mode or "offline").lower()
        if raw == "off":
            self.mode = "off"                                    # 只要本地副本，不碰 swanlab
        elif raw in SWANLAB_MODES:
            self.mode = raw
        else:
            self.mode = "offline"
            self.mode_error = f"未知 swanlab 模式 {raw!r}，按 offline 处理"
        self.project = project or os.environ.get("SWANLAB_PROJECT") or PROJECT
        self.enabled = enabled
        self.export = self.run_dir / "swanlab_export"
        self.export.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.export / "rounds.jsonl"
        self.csv_path = self.export / "rounds.csv"
        self.rows = []
        self.error = getattr(self, "mode_error", None)
        self.run = None
        self._swanlab = None
        self.meta = {"profile": profile, "mode": self.mode, "project": self.project,
                     "run_name": name or self.run_dir.name, "config": config or {}}
        if self.enabled and self.mode != "off":
            self._init_swanlab()

    # ── 本地权威副本 ──────────────────────────────────────────────────
    def _append_local(self, row: dict) -> None:
        self.rows.append(row)
        with self.jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with self.csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            for item in self.rows:
                writer.writerow({key: item.get(key) for key in FIELDS})

    # ── SwanLab（任何失败只记录，不抛）───────────────────────────────
    def _init_swanlab(self):
        if self.mode == "disabled":
            return
        try:
            import swanlab

            self._swanlab = swanlab
            self.run = swanlab.init(project=self.project, name=self.meta["run_name"],
                                    mode=self.mode, log_dir=str(self.run_dir / "_swanlog"),
                                    config={**self.meta["config"], "profile": self.profile,
                                            "run_dir": str(self.run_dir)},
                                    tags=[f"profile-{self.profile}"] if self.profile else None)
            self.meta["swanlab_logdir"] = str(getattr(self.run, "logdir", self.run_dir / "_swanlog"))
            # offline 模式下 .url 这个 property 会抛 ValueError（没有云端 run 路径），
            # 这是预期行为而不是故障，所以单独 try，不污染 swanlab_error。
            try:
                self.meta["swanlab_url"] = getattr(self.run, "url", None)
            except Exception:                                     # noqa: BLE001
                self.meta["swanlab_url"] = None
        except Exception as exc:                                  # noqa: BLE001 记录层不得崩
            self.error = f"{type(exc).__name__}: {exc}"
            self._swanlab = None

    def _log(self, payload: dict, step=None):
        if self._swanlab is None:
            return
        scalars = {}
        for key, value in payload.items():
            if key in NOT_A_SERIES:
                continue                                   # 见上面的定义：把轮次号再画一遍是纯冗余
            if isinstance(value, bool):
                scalars[key] = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                scalars[key] = float(value)
        try:
            self._swanlab.log(scalars, step=step)
        except Exception as exc:                            # noqa: BLE001
            self.error = f"log {type(exc).__name__}: {exc}"

    def _set_summary(self, payload: dict):
        if self._swanlab is None:
            return
        try:
            self._swanlab.log({f"summary/{key}": value for key, value in payload.items()
                               if isinstance(value, (int, float, bool)) or value is None})
        except Exception as exc:                                  # noqa: BLE001
            self.error = f"summary {type(exc).__name__}: {exc}"

    # ── 对外接口 ──────────────────────────────────────────────────────
    def start(self, **payload):
        self.meta.update({key: value for key, value in payload.items()
                          if isinstance(value, (int, float, str, bool)) or value is None})
        if self._swanlab is not None:
            try:
                self._swanlab.config.update({"parent_checkpoint": payload.get("parent_checkpoint"),
                                             "parent_source": payload.get("parent_source"),
                                             "metric": payload.get("metric"),
                                             "rounds": payload.get("rounds")},
                                            allow_val_change=True)
            except Exception as exc:                              # noqa: BLE001
                self.error = f"config {type(exc).__name__}: {exc}"

    def record(self, **row):
        """一轮一条：决策 + 训练 + 门禁 + 记账，全部落在同一行，便于「分析驱动决策」核对。"""
        clean = {}
        for key in FIELDS:
            if key in row:
                clean[key] = row[key]
        for extra in ("round_no", "stage", "ks", "gate", "gain_lcb", "worst_stage_lcb", "zone"):
            if extra in row:
                clean[extra] = row[extra]
        self._append_local(clean)
        self._log(clean, step=clean.get("round"))
        return clean

    def finish(self, results: dict, ledger: dict = None):
        self._set_summary({"best": (results.get("best") or {}).get(results.get("metric")),
                           "baseline": (results.get("baseline") or {}).get(results.get("metric")),
                           "accepted_rounds": results.get("accepted_rounds"),
                           "worst_stage_gain": results.get("worst_stage_gain"),
                           "train_seconds_total": results.get("train_seconds_total")})
        if ledger:
            self._set_summary({f"ledger_{key}": value for key, value in ledger.items()
                               if isinstance(value, (int, float))})
        summary = {"run": str(self.run_dir), "rows": len(self.rows), "mode": self.mode,
                   "project": self.project, "swanlab_error": self.error,
                   "local_csv": str(self.csv_path), "local_jsonl": str(self.jsonl),
                   "swanlab_logdir": self.meta.get("swanlab_logdir"),
                   "swanlab_url": self.meta.get("swanlab_url")}
        (self.export / "record_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if self._swanlab is not None:
            try:
                self._swanlab.finish()
            except Exception as exc:                              # noqa: BLE001
                self.error = f"finish {type(exc).__name__}: {exc}"
        return summary


def sync_hint(run_dir) -> str:
    """离线记录怎么变成「云端链接」：一行命令，写进提交材料索引里给评审看。

    注意路径层级：`swanlab sync` 要的是**里层那个 run 目录**（里面有 `run-*.swanlab`），
    不是外层 `_swanlog`。传 `_swanlog` 会报 `No run-*.swanlab file found`——实测踩过，
    这个提示是我们自己写给评审和队友看的，写错等于把人往坑里带。
    """
    inner = sorted((Path(run_dir) / "_swanlog").glob("run-*"))
    target = str(inner[-1]) if inner else str(Path(run_dir) / "_swanlog" / "run-<时间戳>")
    return (f"python -m swanlab sync {target}   "
            "# 需要已 swanlab login；返回的 run 链接填进 00-提交材料索引.md")
