"""Governance layer tests: chain, guard, gate, supervisor, ledger.

Run from `official/`:
    python -m pytest tests/test_governance.py -q
"""
from __future__ import annotations

import json

import pytest

from harness.governance import (
    Budget, Gate, GateConfig, GuardResult, Ledger, LineageLog, Observation,
    PolicyGuard, PolicyError, Proposal, RecoveryTracker, Supervisor,
    build_manifest, canonical_json, digest, write_json,
)
from harness.audit import collect, main as audit_main, render


def proposal(**overrides):
    base = dict(round=1, action="calibrate", epochs=5,
                params={"lr": 0.003, "wd": 0.0},
                hypothesis="如果校准配方仍有收益，则全阶段增益应为正",
                expected_direction="improve", confidence=0.7, source="rule")
    base.update(overrides)
    return Proposal(**base)


def observation(**overrides):
    base = dict(round=1, stage_id="stage-1", ks=0.2, state="normal",
                metric=0.71, score=0.71, drift_type="none")
    base.update(overrides)
    return Observation(**base)


def guard(**kwargs):
    return PolicyGuard({"calibrate": {"lr", "wd", "ls", "jitter", "clip", "weights"}}, **kwargs)


# ---------------------------------------------------------------- 基础


def test_canonical_json_is_order_insensitive():
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1})
    assert digest({"a": 1}) == digest({"a": 1})


# ---------------------------------------------------------------- 证据链


def test_lineage_chain_verifies(tmp_path):
    log = LineageLog(tmp_path / "lineage.jsonl")
    log.append("observation", {"ks": 0.1})
    log.append("proposal", {"action": "calibrate"})
    log.append("verdict", {"accepted": True})
    ok, count, error = log.verify()
    assert ok and count == 3 and error is None


def test_lineage_detects_tamper(tmp_path):
    path = tmp_path / "lineage.jsonl"
    log = LineageLog(path)
    log.append("observation", {"ks": 0.1})
    log.append("verdict", {"accepted": True})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[1]["payload"]["accepted"] = False
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    ok, count, error = log.verify()
    assert not ok and count == 2 and error == "hash mismatch"


# ---------------------------------------------------------------- 治理闸门


def test_guard_denies_unknown_action():
    result = guard().check(proposal(action="wd_strong"))
    assert not result.allowed
    assert any("不在白名单" in reason for reason in result.reasons)


def test_guard_denies_extra_param():
    result = guard().check(proposal(params={"hidden": 64}))
    assert not result.allowed
    assert any("不允许" in reason for reason in result.reasons)


def test_guard_denies_resume_mismatch():
    result = guard().check(proposal(), {"resume": "runs/round-01/checkpoint.npz",
                                        "best_checkpoint": "runs/best/checkpoint.npz"})
    assert not result.allowed
    assert any(reason.startswith("I1") for reason in result.reasons)


def test_guard_denies_validation_as_train_file():
    result = guard().check(proposal(), {"train_file": "/data/validation.jsonl",
                                         "resume": "/runs/best.npz", "best_checkpoint": "/runs/best.npz"})
    assert not result.allowed
    assert any(reason.startswith("data") for reason in result.reasons)


def test_guard_freezes_thresholds():
    frozen = guard()
    frozen.freeze({"warn": 0.1, "alarm": 0.2})
    bad = frozen.check(proposal(), {"thresholds_hash": "deadbeef"})
    good = frozen.check(proposal(), {"thresholds_hash": frozen.thresholds_hash})
    assert not bad.allowed and any(reason.startswith("I6") for reason in bad.reasons)
    assert good.allowed


def test_guard_checks_budget():
    result = guard().check(proposal(), {"budget": Budget(labels=10), "used": {"labels": 10}})
    assert not result.allowed
    assert any(reason.startswith("budget") for reason in result.reasons)


def test_guard_allows_valid_proposal():
    result = guard().check(proposal(), {"resume": "/runs/best.npz", "best_checkpoint": "/runs/best.npz",
                                        "train_file": "/data/public/train.jsonl"})
    assert result.allowed and result.reasons == []


# ---------------------------------------------------------------- 门禁


def test_gate_clear_win_is_accepted():
    gate = Gate(GateConfig(min_gain=0.002, sigma=0.0268))
    verdict = gate.evaluate([0.78, 0.77, 0.76], 0.714)
    assert verdict.accepted and verdict.zone == "clear_win"
    assert verdict.gain_lcb > 0.002


def test_gate_noise_is_rejected():
    gate = Gate(GateConfig(min_gain=0.002, sigma=0.0268))
    verdict = gate.evaluate([0.715, 0.714, 0.716], 0.714)
    assert not verdict.accepted and verdict.zone == "ambiguous"


def test_gate_fast_mode_lowers_bar_but_keeps_floor():
    gate = Gate(GateConfig(min_gain=0.05, sigma=0.0268, fast_min_gain=0.0))
    verdict = gate.evaluate([0.725, 0.725, 0.725], 0.714, mode="fast")
    assert verdict.accepted and verdict.gate_mode == "fast"


def test_gate_blocks_catastrophic_stage():
    gate = Gate(GateConfig(min_gain=5.0, sigma=1.0, k=1.0, catastrophic_ratio=0.05))
    verdict = gate.evaluate([110.0], 100.0, static_stages=[100.0, 100.0, 100.0],
                            stage_scores=[[110.0, 50.0, 110.0]])
    assert not verdict.accepted
    assert verdict.worst_stage_lcb < -5.0


def test_gate_requires_retention():
    gate = Gate(GateConfig(min_gain=0.002, sigma=0.0268))
    verdict = gate.evaluate([0.75, 0.76, 0.74], 0.714, retention_delta=-0.2)
    assert not verdict.accepted
    assert any("retention" in reason for reason in verdict.reason.split("; "))


def test_mde_and_power():
    mde = Gate.mde(3, 0.0268)
    assert 0.03 < mde < 0.05          # A 的 +0.0118 远低于 MDE，说明 v1 门禁在追噪声
    assert Gate.power(3, 0.05, 0.0268) > 0.8


# ---------------------------------------------------------------- 监督与恢复


def test_supervisor_fast_then_hold_then_rollback():
    sup = Supervisor(max_consecutive_failures=2, fast_rounds=2)
    alarm = observation(state="alarm", drift_type="mixed")
    allowed = GuardResult(allowed=True)
    assert sup.decide(alarm, proposal(), allowed)["mode"] == "fast"
    sup.mark_fast()
    sup.record(accepted=False)
    sup.record(accepted=False)
    assert sup.decide(observation(), proposal(), allowed)["mode"] == "hold"
    sup.record(accepted=False, catastrophic=True)
    assert sup.decide(observation(), proposal(), allowed)["mode"] == "rollback"


def test_supervisor_holds_on_guard_denial():
    sup = Supervisor()
    verdict = sup.decide(observation(), proposal(), GuardResult(allowed=False, reasons=["action"]))
    assert verdict["mode"] == "hold" and verdict["reason"] == "guard_denied"


def test_recovery_tracker_counts_stages():
    tracker = RecoveryTracker(target=0.9)
    tracker.mark_drift(1)
    tracker.observe(1, 0.5, 1.0)
    assert not tracker.recovered()
    tracker.observe(2, 0.95, 1.0)
    assert tracker.recovered() and tracker.stages_to_recover() == 1


# ---------------------------------------------------------------- 账本与清单


def test_ledger_unit_gain():
    ledger = Ledger()
    ledger.charge(labels=1000, tokens_in=600, tokens_out=400, epochs=10, seconds=60)
    assert ledger.tokens == 1000
    assert ledger.cost_units() == pytest.approx(4.0)
    assert ledger.unit_gain(2.0) == pytest.approx(0.5)
    assert ledger.for_budget()["tokens"] == 1000


def test_ledger_rejects_unknown_field():
    with pytest.raises(PolicyError):
        Ledger().charge(gpu_hours=1)


def test_manifest_is_deterministic(tmp_path):
    (tmp_path / "harness").mkdir()
    (tmp_path / "harness" / "a.py").write_text("x = 1\n", encoding="utf-8")
    first = build_manifest(tmp_path)
    second = build_manifest(tmp_path)
    assert first["code_digest"] == second["code_digest"]
    assert len(first["code_digest"]) == 64


# ---------------------------------------------------------------- 组合：一轮完整闭环 + 审计


def _synthetic_run(root):
    log = LineageLog(root / "lineage.jsonl")
    ledger = Ledger()
    prop = proposal()
    result = guard().check(prop, {"resume": "/x/best.npz", "best_checkpoint": "/x/best.npz",
                                  "train_file": "/x/public/train.jsonl"})
    mode = Supervisor().decide(observation(), prop, result)
    verdict = Gate(GateConfig(min_gain=0.002, sigma=0.0268)).evaluate([0.78, 0.77, 0.76], 0.714)
    log.append("observation", observation().to_dict(), trace_id="t1")
    log.append("proposal", prop.to_dict(), trace_id="t1")
    log.append("supervisor", mode, trace_id="t1")
    log.append("verdict", verdict.to_dict(), trace_id="t1")
    ledger.charge(labels=100, epochs=5, seconds=2.0)
    write_json(root / "ledger.json", ledger.to_dict())
    write_json(root / "results.json", {"profile": "A", "rounds": 1, "accepted_rounds": 1,
                                       "baseline": {"macro_f1": 0.714}, "best": {"macro_f1": 0.78},
                                       "worst_stage_gain": 0.0})
    return verdict


def test_synthetic_run_audits_clean(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _synthetic_run(run)
    payload = collect(run)
    assert payload["lineage"]["ok"] and payload["lineage"]["events"] == 4
    assert payload["kinds"] == {"observation": 1, "proposal": 1, "supervisor": 1, "verdict": 1}
    assert "chain ok" in render(payload)
    assert audit_main([str(run)]) == 0


def test_audit_fails_on_tampered_chain(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    _synthetic_run(run)
    path = run / "lineage.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[2]["payload"]["mode"] = "rollback"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    assert audit_main([str(run)]) == 1
    assert "BROKEN" in capsys.readouterr().out


def test_audit_reports_malformed_lineage(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "lineage.jsonl").write_text("{not json}\n", encoding="utf-8")
    payload = collect(run)
    assert payload["lineage"]["ok"] is False
    assert "bad json" in payload["lineage"]["error"].lower() or "expecting" in payload["lineage"]["error"].lower()
    assert audit_main([str(run)]) == 1