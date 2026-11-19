"""Governed decision layer for the 智训引擎 harness (v2).

v2 不改模型、不改评测口径，只把「谁能改、改前怎么证明、改后怎么记账、坏了怎么回滚」
做成可执行、可核验的代码。设计借了 multi-agentic-framework-main 的六个模式：

  AgentMessage / AgentDecision -> Observation / Proposal / Verdict
  EscalationRouter             -> Supervisor（strict / fast / hold / rollback）
  LineageBus                   -> LineageLog（SHA-256 前向链，篡改可发现）
  SafetyGuardrail              -> PolicyGuard（default deny，失败即拒绝）
  Prometheus metrics           -> Ledger（JSON 三本账：标注 / token / 算力）
  UNS Context Broker           -> 单一执行入口（所有训练/评测都要过 guard）

只用标准库，不引入 pydantic / OPA / Ed25519 / AutoGen：
镜像仍是 numpy-only，离线可跑，单元测试不需要官方三个包。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 基础：规范化 JSON 与摘要
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """稳定序列化：同一份内容永远得到同一个字符串。"""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _same_path(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


def _sd(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


class PolicyError(RuntimeError):
    """动作被治理层拒绝；调用方应记录并进入 hold，而不是重试绕过。"""


# ---------------------------------------------------------------------------
# 数据契约
# ---------------------------------------------------------------------------

STATES = {"normal", "warn", "alarm"}
DRIFT_TYPES = {"none", "covariate", "concept", "mixed", "unknown"}
DIRECTIONS = {"improve", "non_degrade"}
REVERSIBILITY = {"reversible", "soft", "irreversible"}


@dataclass
class Observation:
    """一轮开始时的只读观测。不要把标签、隐藏集或模型内部状态塞进来。"""

    round: int
    stage_id: str
    ks: float
    state: str
    metric: float
    score: float  # 已统一成「越大越好」
    drift_type: str = "none"
    onset_stage: int | None = None
    confidence: float = 1.0
    labels_used: int = 0
    data_digest: str = ""

    def validate(self) -> None:
        if self.round < 1:
            raise PolicyError("round 必须 >= 1")
        if not self.stage_id:
            raise PolicyError("stage_id 不能为空")
        if self.state not in STATES:
            raise PolicyError(f"state 必须是 {sorted(STATES)}，收到 {self.state!r}")
        if self.drift_type not in DRIFT_TYPES:
            raise PolicyError(f"drift_type 必须是 {sorted(DRIFT_TYPES)}，收到 {self.drift_type!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise PolicyError("confidence 必须在 [0, 1]")
        if self.labels_used < 0:
            raise PolicyError("labels_used 不能为负")

    def to_dict(self) -> dict:
        return {
            "round": self.round, "stage_id": self.stage_id, "ks": self.ks,
            "state": self.state, "metric": self.metric, "score": self.score,
            "drift_type": self.drift_type, "onset_stage": self.onset_stage,
            "confidence": self.confidence, "labels_used": self.labels_used,
            "data_digest": self.data_digest,
        }


@dataclass
class Proposal:
    """策略层给出的一个候选动作。它只是「申请」，不是「命令」。"""

    round: int
    action: str
    epochs: int
    params: dict[str, Any] = field(default_factory=dict)
    hypothesis: str = ""
    expected_direction: str = "improve"
    confidence: float = 0.5
    reversibility: str = "reversible"
    source: str = "rule"  # rule | llm | rule_fallback
    inputs_hash: str = ""

    def validate(self) -> None:
        if self.round < 1:
            raise PolicyError("round 必须 >= 1")
        if not self.action:
            raise PolicyError("action 不能为空")
        if not 0 <= int(self.epochs) <= 12:
            raise PolicyError("epochs 必须在 [0, 12]")
        if self.expected_direction not in DIRECTIONS:
            raise PolicyError(f"expected_direction 必须是 {sorted(DIRECTIONS)}")
        if self.reversibility not in REVERSIBILITY:
            raise PolicyError(f"reversibility 必须是 {sorted(REVERSIBILITY)}")
        if not 0.0 <= self.confidence <= 1.0:
            raise PolicyError("confidence 必须在 [0, 1]")
        if len(self.hypothesis.strip()) < 6:
            raise PolicyError("hypothesis 太短：必须写出可被下一轮证伪的假设")

    def to_dict(self) -> dict:
        return {
            "round": self.round, "action": self.action, "epochs": int(self.epochs),
            "params": dict(self.params), "hypothesis": self.hypothesis,
            "expected_direction": self.expected_direction,
            "confidence": self.confidence, "reversibility": self.reversibility,
            "source": self.source, "inputs_hash": self.inputs_hash,
        }


@dataclass
class Verdict:
    """门禁裁决。accepted 是唯一能改变 best_checkpoint 的开关。"""

    accepted: bool
    reason: str
    gain: float
    gain_lcb: float
    worst_stage_gain: float
    worst_stage_lcb: float
    retention_delta: float | None
    gate_mode: str
    zone: str
    cost: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted, "reason": self.reason, "gain": self.gain,
            "gain_lcb": self.gain_lcb, "worst_stage_gain": self.worst_stage_gain,
            "worst_stage_lcb": self.worst_stage_lcb, "retention_delta": self.retention_delta,
            "gate_mode": self.gate_mode, "zone": self.zone,
            "cost": dict(self.cost), "evidence": dict(self.evidence),
        }


# ---------------------------------------------------------------------------
# 预算与账本
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Budget:
    """平台/队伍给出的硬额度；None 表示不限制。"""

    labels: int | None = None
    tokens: int | None = None
    epochs: int | None = None
    seconds: float | None = None

    def exhausted(self, used: dict[str, float]) -> list[str]:
        out = []
        for name in ("labels", "tokens", "epochs", "seconds"):
            limit = getattr(self, name)
            if limit is not None and float(used.get(name, 0.0)) >= float(limit):
                out.append(name)
        return out

    def remaining(self, used: dict[str, float]) -> dict[str, float | None]:
        return {name: (None if getattr(self, name) is None else max(0.0, float(getattr(self, name)) - float(used.get(name, 0.0))))
                for name in ("labels", "tokens", "epochs", "seconds")}

    def to_dict(self) -> dict:
        return {"labels": self.labels, "tokens": self.tokens, "epochs": self.epochs, "seconds": self.seconds}


class Ledger:
    """三本账：标注 / token / 算力。unit_gain 只在同一 profile 内比较。"""

    FIELDS = ("labels", "tokens_in", "tokens_out", "epochs", "seconds", "train_calls", "eval_calls")

    def __init__(self) -> None:
        self.used = {name: 0.0 for name in self.FIELDS}
        self.events: list[dict] = []

    def charge(self, **kwargs: float) -> None:
        unknown = set(kwargs) - set(self.FIELDS)
        if unknown:
            raise PolicyError(f"未知记账字段: {sorted(unknown)}")
        for key, value in kwargs.items():
            if float(value) < 0:
                raise PolicyError(f"{key} 不能为负")
            self.used[key] += float(value)
        self.events.append({key: float(value) for key, value in kwargs.items()})

    @property
    def tokens(self) -> float:
        return self.used["tokens_in"] + self.used["tokens_out"]

    def for_budget(self) -> dict[str, float]:
        return {"labels": self.used["labels"], "tokens": self.tokens,
                "epochs": self.used["epochs"], "seconds": self.used["seconds"]}

    def cost_units(self) -> float:
        """相对成本：只用于同 profile 的「单位增益」排序，不是真实货币。"""
        return (self.used["labels"] / 1000.0 + self.tokens / 1000.0
                + self.used["epochs"] / 10.0 + self.used["seconds"] / 60.0)

    def unit_gain(self, gain: float) -> float:
        return float(gain) / max(self.cost_units(), 1e-9)

    def to_dict(self) -> dict:
        return {"used": dict(self.used), "tokens": self.tokens,
                "cost_units": round(self.cost_units(), 6),
                "events": list(self.events)}


# ---------------------------------------------------------------------------
# 治理：default deny
# ---------------------------------------------------------------------------


@dataclass
class GuardResult:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"allowed": self.allowed, "reasons": list(self.reasons), "warnings": list(self.warnings)}


class PolicyGuard:
    """所有训练/评测调用的唯一闸门。默认拒绝，任何不确定都 fail closed。

    六条不变量（与 `设计说明.md` 的 I1–I6 对齐）：
      I1 只从「当前 best checkpoint」续训
      I2 只跑白名单动作，且只动该动作允许的参数
      I3 候选必须先过门禁才可能被采纳（由 run.py 的顺序保证）
      I4 每轮都要记账（Ledger 非空）
      I5 每个决策都要有可证伪假设
      I6 门禁阈值冻结，策略层不能自我放松
    """

    def __init__(self, allowed_actions: dict[str, set[str]], *,
                 max_epochs: int = 12,
                 forbidden_train_tokens: tuple[str, ...] = ("validation", "hidden", "test", "holdout"),
                 frozen_thresholds: dict | None = None) -> None:
        self.allowed_actions = {name: set(keys) for name, keys in allowed_actions.items()}
        self.max_epochs = int(max_epochs)
        self.forbidden_train_tokens = tuple(token.lower() for token in forbidden_train_tokens)
        self.frozen_thresholds = dict(frozen_thresholds) if frozen_thresholds is not None else None
        self.thresholds_hash = digest(self.frozen_thresholds) if self.frozen_thresholds is not None else None
        self.denials: list[dict] = []

    def freeze(self, thresholds: dict) -> None:
        self.frozen_thresholds = dict(thresholds)
        self.thresholds_hash = digest(self.frozen_thresholds)

    def check(self, proposal: Proposal, context: dict[str, Any] | None = None) -> GuardResult:
        context = context or {}
        reasons: list[str] = []
        warnings: list[str] = []

        try:
            proposal.validate()
        except PolicyError as exc:
            reasons.append(f"schema: {exc}")

        # 1) 动作白名单：未知动作一律拒绝。
        if proposal.action not in self.allowed_actions:
            reasons.append(f"action: {proposal.action!r} 不在白名单")
        else:
            allowed_keys = self.allowed_actions[proposal.action]
            extra = set(proposal.params) - allowed_keys
            if extra:
                reasons.append(f"params: {proposal.action!r} 不允许 {sorted(extra)}")

        # 2) 轮数上限。
        if not 0 <= int(proposal.epochs) <= self.max_epochs:
            reasons.append(f"epochs: {proposal.epochs} 超出 [0, {self.max_epochs}]")

        # 3) I1：只能从 best checkpoint 续训。
        if int(proposal.epochs) > 0:
            resume, best = context.get("resume"), context.get("best_checkpoint")
            if resume is not None and best is not None and not _same_path(resume, best):
                reasons.append("I1: resume 不是当前 best_checkpoint，拒绝从被拒轮次滚雪球")

            # 4) 训练数据只能来自 public/train 的派生文件。
            train_file = context.get("train_file")
            if train_file:
                lowered = str(train_file).lower()
                hit = [token for token in self.forbidden_train_tokens if token in lowered]
                if hit:
                    reasons.append(f"data: train_file 命中禁止路径 {hit}，验证/隐藏标签不得进梯度")

        # 5) I5：可证伪假设。
        if len(proposal.hypothesis.strip()) < 6:
            reasons.append("I5: hypothesis 为空或太短")

        # 6) I6：阈值冻结。
        if self.thresholds_hash is not None:
            got = context.get("thresholds_hash")
            if got != self.thresholds_hash:
                reasons.append("I6: thresholds_hash 与冻结值不一致，策略层不得放松门禁")

        # 7) 预算。
        budget: Budget | None = context.get("budget")
        used = context.get("used")
        if budget is not None and used is not None:
            exhausted = budget.exhausted(used)
            if exhausted:
                reasons.append(f"budget: 已耗尽 {exhausted}")

        # 8) 特征泄漏 canary（可选；工业包是数值特征，通常没有名字）。
        names = context.get("feature_names") or []
        leak_tokens = ("bearing", "snapshot", "rul", "life", "id", "target")
        leaked = [str(n) for n in names if any(token in str(n).lower() for token in leak_tokens)]
        if leaked:
            reasons.append(f"leakage: 特征名含可疑字段 {leaked[:5]}")

        # 9) LLM 低置信度不直接拒绝，但要显式告警，让 supervisor 决定是否降级。
        if proposal.source == "llm" and proposal.confidence < 0.5:
            warnings.append("llm_confidence_low")

        result = GuardResult(allowed=not reasons, reasons=reasons, warnings=warnings)
        if not result.allowed:
            self.denials.append({"round": proposal.round, "action": proposal.action, **result.to_dict()})
        return result

    def assert_allowed(self, proposal: Proposal, context: dict[str, Any] | None = None) -> None:
        result = self.check(proposal, context)
        if not result.allowed:
            raise PolicyError("; ".join(result.reasons))


# ---------------------------------------------------------------------------
# 证据链：前向哈希链
# ---------------------------------------------------------------------------


class LineageLog:
    """append-only JSONL；每条事件带 prev_hash，verify() 能定位第一个断点。

    这不是 Ed25519 签名，也不依赖 OpenLineage：离线评测里只需要「同一份日志
    改一个字就能被发现」，SHA-256 链足够，且零依赖。
    """

    GENESIS = "0" * 64

    def __init__(self, path: str | Path, run_id: str | None = None, reset: bool = True) -> None:
        self.path = Path(path)
        self.run_id = run_id or time.strftime("%Y%m%dT%H%M%S")
        self._seq = 0
        self._last_hash = self.GENESIS
        if reset and self.path.exists():
            self.path.unlink()
        if not reset and self.path.exists():
            self._load_tail()

    def _load_tail(self) -> None:
        lines = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return
        last = json.loads(lines[-1])
        self._seq = int(last.get("seq", len(lines)))
        self._last_hash = str(last.get("hash", self.GENESIS))

    def append(self, kind: str, payload: dict[str, Any] | None = None, *, trace_id: str | None = None) -> dict:
        event = {
            "seq": self._seq + 1,
            "ts": round(time.time(), 3),
            "run_id": self.run_id,
            "kind": kind,
            "trace_id": trace_id,
            "payload": payload or {},
            "prev_hash": self._last_hash,
        }
        event["hash"] = digest(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._seq += 1
        self._last_hash = event["hash"]
        return event

    def events(self, kind: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [row for row in rows if kind is None or row.get("kind") == kind]

    def verify(self) -> tuple[bool, int, str | None]:
        """返回 (ok, 检查到的事件数, 第一个错误)。"""
        prev = self.GENESIS
        rows = self.events()
        for index, row in enumerate(rows, start=1):
            got = row.pop("hash", None)
            if got is None:
                return False, index, "missing hash"
            if digest(row) != got:
                return False, index, "hash mismatch"
            if row.get("prev_hash") != prev:
                return False, index, "prev_hash mismatch"
            prev = got
        return True, len(rows), None

    def summary(self) -> dict:
        kinds: dict[str, int] = {}
        for row in self.events():
            kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
        ok, count, error = self.verify()
        return {"path": str(self.path), "events": count, "ok": ok, "error": error, "kinds": kinds}


# ---------------------------------------------------------------------------
# 监督：strict / fast / hold / rollback
# ---------------------------------------------------------------------------


class Supervisor:
    """把 EscalationRouter 的「何时暂停」改成「何时换挡」。

    没有人工可等（规则要求全自动），所以升级目标不是 HITL，而是：
      strict   正常证据标准
      fast     确认漂移后的应急通道：预注册动作、证据门槛降低、有轮次上限
      hold     预算耗尽 / 连续被拒 / 治理拒绝：不训练，省算力
      rollback 出现灾难性回退：回滚到 best 并停止本轮
    """

    def __init__(self, confidence_threshold: float = 0.85,
                 max_consecutive_failures: int = 2, fast_rounds: int = 2) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.max_consecutive_failures = int(max_consecutive_failures)
        self.fast_rounds = int(fast_rounds)
        self.consecutive_failures = 0
        self.fast_used = 0
        self.rollback_required = False
        self.log: list[dict] = []

    def decide(self, observation: Observation, proposal: Proposal,
               guard: GuardResult, used: dict[str, float] | None = None,
               budget: Budget | None = None) -> dict:
        used = used or {}
        if budget is not None:
            exhausted = budget.exhausted(used)
            if exhausted:
                return self._record({"mode": "hold", "level": "alert", "reason": "budget_exhausted", "detail": exhausted})
        if not guard.allowed:
            return self._record({"mode": "hold", "level": "warning", "reason": "guard_denied", "detail": guard.reasons})
        if self.rollback_required:
            return self._record({"mode": "rollback", "level": "critical", "reason": "catastrophic_regression"})
        if self.consecutive_failures >= self.max_consecutive_failures:
            return self._record({"mode": "hold", "level": "warning", "reason": "consecutive_failures"})
        if (observation.state == "alarm" and observation.drift_type in {"covariate", "concept", "mixed"}
                and self.fast_used < self.fast_rounds):
            return self._record({"mode": "fast", "level": "alert", "reason": "confirmed_drift"})
        if proposal.confidence < self.confidence_threshold and proposal.source == "llm":
            return self._record({"mode": "strict", "level": "info", "reason": "llm_low_confidence_downgrade"})
        return self._record({"mode": "strict", "level": "info", "reason": "normal"})

    def mark_fast(self) -> None:
        self.fast_used += 1

    def record(self, accepted: bool, *, catastrophic: bool = False) -> None:
        if catastrophic:
            self.rollback_required = True
        if accepted:
            self.consecutive_failures = 0
            self.fast_used = 0
        else:
            self.consecutive_failures += 1

    def _record(self, verdict: dict) -> dict:
        self.log.append(verdict)
        return verdict

    def state(self) -> dict:
        return {"consecutive_failures": self.consecutive_failures,
                "fast_used": self.fast_used,
                "rollback_required": self.rollback_required,
                "log": list(self.log)}


# ---------------------------------------------------------------------------
# 恢复速度
# ---------------------------------------------------------------------------


class RecoveryTracker:
    """记录「漂移后第几个阶段恢复到漂移前的 90%」。

    传入的 score 必须已经统一成「越大越好」，否则 B/C 的 RMSE 会算反。
    """

    def __init__(self, target: float = 0.9) -> None:
        self.target = float(target)
        self.drift_stage: int | None = None
        self.rows: list[dict] = []

    def mark_drift(self, stage_index: int) -> None:
        self.drift_stage = int(stage_index)

    def observe(self, stage_index: int, score: float, baseline_score: float) -> float:
        ratio = float(score) / float(baseline_score) if baseline_score else 0.0
        self.rows.append({"stage": int(stage_index), "score": float(score),
                          "baseline": float(baseline_score), "ratio": ratio})
        return ratio

    def recovered(self) -> bool:
        return bool(self.rows) and self.rows[-1]["ratio"] >= self.target

    def stages_to_recover(self) -> int | None:
        if self.drift_stage is None or not self.recovered():
            return None
        return int(self.rows[-1]["stage"]) - self.drift_stage

    def to_dict(self) -> dict:
        return {"target": self.target, "drift_stage": self.drift_stage,
                "recovered": self.recovered(), "stages_to_recover": self.stages_to_recover(),
                "rows": list(self.rows)}


# ---------------------------------------------------------------------------
# 门禁 v2：全阶段 × 多 seed × 三区 + 可选对照臂
# ---------------------------------------------------------------------------

Z_ALPHA = 1.6448536269514722   # one-sided 95%
Z_POWER = 0.8416212335729143   # 80% power


@dataclass
class GateConfig:
    min_gain: float
    sigma: float                 # 训练种子噪声标准差（同 profile 的标定值）
    k: float = 1.0               # LCB 的保守系数；1.0 = 一个标准误
    catastrophic_ratio: float = 0.05
    retention_floor: float = -0.05
    clear_sigma: float = 2.0
    fast_min_gain: float = 0.0


class Gate:
    """三区门禁：

      |gain| >= clear_sigma * sigma  -> 明确赢/输，不跑对照臂
      否则                           -> ambiguous，建议同 seed 跑一次对照臂
    采纳条件（strict）：
      LCB(gain) >= min_gain 且 LCB(worst_stage_gain) >= -catastrophic_ratio*|static|
      且 retention_delta >= retention_floor
    fast 模式只把 min_gain 降到 fast_min_gain，灾难性下限和抗遗忘下限不变。
    """

    def __init__(self, config: GateConfig) -> None:
        self.config = config

    def classify(self, gain: float) -> str:
        threshold = self.config.clear_sigma * max(self.config.sigma, 1e-12)
        if gain >= threshold:
            return "clear_win"
        if gain <= -threshold:
            return "clear_loss"
        return "ambiguous"

    def evaluate(self, seed_scores: list[float], static_score: float, *,
                 static_stages: list[float] | None = None,
                 stage_scores: list[list[float]] | None = None,
                 retention_delta: float | None = None,
                 mode: str = "strict") -> Verdict:
        if not seed_scores:
            raise PolicyError("seed_scores 不能为空")
        gains = [float(score) - float(static_score) for score in seed_scores]
        n = len(gains)
        mean_gain = sum(gains) / n
        se = (_sd(gains) / math.sqrt(n)) if n > 1 else float(self.config.sigma)
        lcb = mean_gain - self.config.k * se
        zone = self.classify(mean_gain)

        worst_mean = mean_gain
        worst_lcb = lcb
        if stage_scores and static_stages:
            per_stage_means, per_stage_lcbs = [], []
            for index, static_value in enumerate(static_stages):
                stage_gains = [float(row[index]) - float(static_value) for row in stage_scores]
                stage_mean = sum(stage_gains) / len(stage_gains)
                stage_se = (_sd(stage_gains) / math.sqrt(len(stage_gains))) if len(stage_gains) > 1 else float(self.config.sigma)
                per_stage_means.append(stage_mean)
                per_stage_lcbs.append(stage_mean - self.config.k * stage_se)
            worst_mean = min(per_stage_means)
            worst_lcb = min(per_stage_lcbs)

        floor = -self.config.catastrophic_ratio * abs(float(static_score))
        retention_ok = retention_delta is None or float(retention_delta) >= self.config.retention_floor
        need = self.config.fast_min_gain if mode == "fast" else self.config.min_gain
        accepted = bool(lcb >= need and worst_lcb >= floor and retention_ok)

        reasons = []
        if lcb < need:
            reasons.append(f"lcb {lcb:.6f} < {need:.6f}")
        if worst_lcb < floor:
            reasons.append(f"worst_lcb {worst_lcb:.6f} < floor {floor:.6f}")
        if not retention_ok:
            reasons.append(f"retention {retention_delta:.4f} < {self.config.retention_floor:.4f}")
        verdict = Verdict(
            accepted=accepted,
            reason="; ".join(reasons) if reasons else "pass",
            gain=mean_gain, gain_lcb=lcb, worst_stage_gain=worst_mean,
            worst_stage_lcb=worst_lcb, retention_delta=retention_delta,
            gate_mode=mode, zone=zone,
        )
        return verdict

    @staticmethod
    def mde(n: int, sigma: float, alpha: float = 0.05, power: float = 0.8) -> float:
        """最小可检测效应：n 个 seed、噪声 sigma 时，能以给定 power 检出的增益。"""
        if n < 1:
            raise PolicyError("n 必须 >= 1")
        z_alpha = Z_ALPHA if abs(alpha - 0.05) < 1e-9 else 1.2815515655446004
        z_power = Z_POWER if abs(power - 0.8) < 1e-9 else 0.0
        return (z_alpha + z_power) * float(sigma) / math.sqrt(n)

    @staticmethod
    def power(n: int, effect: float, sigma: float) -> float:
        if sigma <= 0 or n < 1:
            return 1.0
        z = float(effect) * math.sqrt(n) / float(sigma) - Z_ALPHA
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# 运行清单：把「这次跑了什么」变成可复现的哈希
# ---------------------------------------------------------------------------


def code_digest(root: str | Path, subdir: str = "harness") -> str:
    base = Path(root) / subdir
    files = sorted(path for path in base.glob("*.py") if "__pycache__" not in path.parts)
    return digest({path.name: file_digest(path) for path in files})


def build_manifest(root: str | Path, *, extra: dict[str, Any] | None = None,
                   data_files: list[str | Path] | None = None) -> dict:
    base = Path(root)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "code_digest": code_digest(base),
        "data_digest": digest({str(Path(path).name): file_digest(Path(path)) for path in (data_files or [])}),
        "env": {key: os.environ.get(key) for key in ("PYTRIO_LLM_MODEL", "ZHIXUN_WORKSPACE") if os.environ.get(key)},
    }
    if extra:
        payload["extra"] = extra
    return payload


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")