"""B 的标签来源说明：阶梯、只读性、以及"严格版训练器下真的能跑"的功能验证。

背景（2026-11-15 平台预检 + 用 `官方基线工业模型源码（不许改）` 核对）：
官方 A/C 包 `baselines/README.md` 要求 B 训练必须传 `--label-provenance`，
但 B 包自己那份 `train.py`（6257f0b1…）把检查收窄到 `task == "classification"`；
A/C 那份（77043a0a…）对 `profile == "B"` 一律要求。平台用后者，B 就在开训前 ValueError。

这里最关键的是 `test_strict_trainer_*` 两条：它们**真的执行严格版训练器**，
所以"我们补的声明能被官方判据接受"不是推断出来的。

Run from `official/`:
    python -m pytest tests/test_provenance.py -q
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from harness import baseline, provenance

ROOT = Path(__file__).resolve().parents[1]


def info(**overrides) -> dict:
    """官方 baseline-info.json 的最小可用形态（字段名与 B 包原文一致）。"""
    base = {"dataset_key": "B", "id": "cpu-b-rul-20261110-v1", "task": "regression",
            "label_status": "approved",
            "target_origin": "remaining_time_from_lifecycle",
            "target_definition": "(last_snapshot - current_snapshot) * 1 minute",
            "target_unit": "minutes"}
    base.update(overrides)
    return base


def make_pkg(tmp: Path, **overrides) -> Path:
    pkg = tmp / "cpu-b-rul-test"
    pkg.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in info(**overrides).items() if v is not None}
    (pkg / "baseline-info.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return pkg


# ── 阶梯与硬规矩 ────────────────────────────────────────────────────────────

def test_only_b_requires_a_declaration(tmp_path):
    pkg = make_pkg(tmp_path)
    assert provenance.resolve("A", pkg, tmp_path / "a") is None
    assert provenance.resolve("C", pkg, tmp_path / "c") is None
    assert provenance.preview("A", pkg)["required"] is False


def test_synthesized_declaration_is_valid_for_the_official_validator(tmp_path):
    """官方 train.py 第 61 行的判据：status ∈ {draft, approved} 且 method 非空字符串。"""
    rec = provenance.resolve("B", make_pkg(tmp_path), tmp_path / "run")
    doc = json.loads(Path(rec["_path"]).read_text(encoding="utf-8"))
    assert doc["status"] in ("draft", "approved")
    assert isinstance(doc["method"], str) and doc["method"].strip()


def test_we_never_declare_our_own_approval(tmp_path):
    """官方 label_status=approved 只能被**转述**；我们 authored 的声明永远是 draft。

    官方 README 原话：这里仅记录来源文件的摘要和声明状态，不能证明委员会已批准。
    把自己写的声明标成 approved 就是伪造审批，比不交更糟。
    """
    rec = provenance.resolve("B", make_pkg(tmp_path, label_status="approved"), tmp_path / "run")
    doc = json.loads(Path(rec["_path"]).read_text(encoding="utf-8"))
    assert rec["status"] == "draft"
    assert doc["official_label_status"] == "approved"          # 转述保留原值
    assert doc["source"]["sha256"].startswith("sha256:")
    assert (Path(doc["source"]["file"])).name == "baseline-info.json"


def test_method_only_paraphrases_official_fields(tmp_path):
    rec = provenance.resolve("B", make_pkg(tmp_path), tmp_path / "run")
    doc = json.loads(Path(rec["_path"]).read_text(encoding="utf-8"))
    for phrase in ("remaining_time_from_lifecycle", "(last_snapshot - current_snapshot) * 1 minute",
                   "minutes", "label_status=approved"):
        assert phrase in doc["method"]
    assert "no health/status label was derived from RUL" in doc["method"]      # 明确声明没做红线禁止的派生


def test_declaration_that_flows_into_official_training_json_is_pure_ascii(tmp_path):
    """官方 train.py 写 training.json 不指定 encoding（Windows 上按 GBK 落盘），而 report 会带上
    method 原文。声明里只要有一个非 ASCII 字符，官方产物就变成非 UTF-8，
    我们所有按 UTF-8 读的复核工具当场解不开——本机实测踩过，所以钉成测试。
    """
    rec = provenance.resolve("B", make_pkg(tmp_path), tmp_path / "run")
    raw = Path(rec["_path"]).read_bytes()
    assert raw.isascii()
    doc = json.loads(raw.decode("ascii"))
    assert doc["method"].isascii() and doc["note"].isascii()


def test_missing_official_fields_is_an_error_not_a_silent_skip(tmp_path):
    """没有任何官方字段时报错，绝不现编一段 method。"""
    pkg = tmp_path / "bare"
    pkg.mkdir()
    with pytest.raises(ValueError, match="没有任何可转述的官方字段"):
        provenance.resolve("B", pkg, tmp_path / "run")


def test_projection_package_declares_what_it_cannot_see(tmp_path):
    """平台投影包只有 baselines/configs/B.json（target_unit 级信息），没有 target_origin/definition。

    这时**既不能崩**（每轮投影都这样，评测直接全灭），**也不能补全**（那才是编造标签口径）。
    正确做法：引用能引用的，把缺的三个字段原样列进 missing_fields，并标 complete=False，
    让评审看得见这轮声明的边界在哪。
    """
    pkg = tmp_path / "current_round_projection"
    (pkg / "baselines" / "configs").mkdir(parents=True)
    (pkg / "baselines" / "configs" / "B.json").write_text(json.dumps(
        {"profile": "B", "task": "regression", "target_unit": "minutes",
         "classes": [], "hidden": 24, "seed": 42}), encoding="utf-8")

    rec = provenance.resolve("B", pkg, tmp_path / "run")
    doc = json.loads(Path(rec["_path"]).read_text(encoding="utf-8"))
    assert doc["complete"] is False
    assert set(doc["missing_fields"]) == {"target_origin", "target_definition", "label_status"}
    assert doc["target_unit"] == "minutes" and doc["official_label_status"] is None
    assert "does not carry target_origin/target_definition" in doc["method"]
    assert "label_status=minutes" not in doc["method"]
    # 查过哪些官方文件、哈希是什么，必须能被别人复算
    assert doc["source"]["consulted"] and all(c["sha256"].startswith("sha256:")
                                              for c in doc["source"]["consulted"])
    assert Path(rec["_path"]).read_bytes().isascii()


def test_full_package_is_marked_complete(tmp_path):
    rec = provenance.resolve("B", make_pkg(tmp_path), tmp_path / "run")
    doc = json.loads(Path(rec["_path"]).read_text(encoding="utf-8"))
    assert doc["complete"] is True and doc["missing_fields"] == []
    assert provenance.preview("B", make_pkg(tmp_path))["complete"] is True


def test_falls_back_to_training_json_when_info_absent(tmp_path):
    pkg = tmp_path / "cpu-b-only-training"
    pkg.mkdir()
    (pkg / "training.json").write_text(json.dumps(
        {"target_origin": "remaining_time_from_lifecycle", "target_unit": "minutes",
         "target_definition": "RUL in minutes"}), encoding="utf-8")
    doc = json.loads(Path(provenance.resolve("B", pkg, tmp_path / "run")["_path"]).read_text(encoding="utf-8"))
    assert doc["source"]["file"].endswith("training.json")


def test_official_package_declaration_is_used_verbatim(tmp_path):
    """包里/显式给了官方约定的 labels-provenance.json 时：原样引用，一个字节都不改。"""
    pkg = make_pkg(tmp_path)
    official = pkg / "public" / "B"
    official.mkdir(parents=True)
    doc = {"status": "approved", "method": "组委会发布的标签来源说明"}
    target = official / "labels-provenance.json"
    target.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    before = target.read_bytes()

    rec = provenance.resolve("B", pkg, tmp_path / "run")
    assert rec["_kind"] == "package" and rec["status"] == "approved"
    assert target.read_bytes() == before
    assert rec["sha256"] == provenance.sha256_file(target)


def test_env_var_and_cli_take_priority_over_synthesis(tmp_path, monkeypatch):
    pkg = make_pkg(tmp_path)
    mine = pkg / "mine.json"
    mine.write_text(json.dumps({"status": "draft", "method": "显式指定的声明"}), encoding="utf-8")
    monkeypatch.setenv(provenance.ENV_PATH, str(mine))
    assert provenance.resolve("B", pkg, tmp_path / "r1")["_kind"] == "explicit"
    assert provenance.resolve("B", pkg, tmp_path / "r2", explicit=str(mine))["_kind"] == "explicit"


def test_invalid_official_file_is_reported_then_paraphrased(tmp_path, capsys):
    """组委会给的文件不合规时：不改写它，记录原因后退回我们转述官方字段的那份。"""
    pkg = make_pkg(tmp_path)
    (pkg / "labels-provenance.json").write_text(json.dumps({"status": "final", "method": "x"}),
                                                encoding="utf-8")
    rec = provenance.resolve("B", pkg, tmp_path / "run")
    assert rec["_kind"] == "synthesized"
    assert "不合规" in capsys.readouterr().out


def test_never_writes_into_the_read_only_package(tmp_path):
    """官方包目录（含投影包）全程只读；声明只落在可写输出目录。"""
    pkg = make_pkg(tmp_path)
    before = sorted(p.name for p in pkg.rglob("*"))
    provenance.resolve("B", pkg, tmp_path / "run")
    assert sorted(p.name for p in pkg.rglob("*")) == before


def test_explicit_path_missing_or_directory_is_a_clear_error(tmp_path):
    pkg = make_pkg(tmp_path)
    with pytest.raises(FileNotFoundError):
        provenance.resolve("B", pkg, tmp_path / "r", explicit=str(tmp_path / "nope.json"))
    with pytest.raises(ValueError, match="目录"):
        provenance.resolve("B", pkg, tmp_path / "r", explicit=str(pkg))


# ── 功能验证：真的用严格版（A/C 包那份）训练器跑 B ───────────────────────────

@pytest.fixture
def strict_vendor(tmp_path, monkeypatch):
    """把 vendor/B 拷到临时目录，用 A 包里那份严格版 train.py 覆盖它。

    这是平台预检描述的场景：对 `profile == "B"` 一律要求标签来源声明。
    monkeypatch `baseline.VENDOR` 让 source_dir() 指到临时树；
    `baselines.*` 已经在 sys.modules 里（别的测试可能导过），必须先清掉才会重新加载。
    """
    tree = tmp_path / "vendor_strict"
    shutil.copytree(baseline.VENDOR / "B", tree / "B",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(ROOT / "vendor" / "A" / "baselines" / "train.py",
                    tree / "B" / "baselines" / "train.py")
    for key in [k for k in list(sys.modules) if k == "baselines" or k.startswith("baselines.")]:
        sys.modules.pop(key)
    monkeypatch.setattr(baseline, "VENDOR", tree)
    yield tree
    for key in [k for k in list(sys.modules) if k == "baselines" or k.startswith("baselines.")]:
        sys.modules.pop(key)
    while str(tree) in sys.path:
        sys.path.remove(str(tree))


def test_strict_trainer_rejects_round_without_declaration(strict_vendor):
    parent, _src = baseline.parent_checkpoint("B")
    with pytest.raises(ValueError, match="requires explicit label provenance"):
        baseline.continue_round("B", strict_vendor.parent / "round-noprov", parent,
                                epochs=1, stage_id="t-noprov", label_provenance=None)


def test_strict_trainer_accepts_our_declaration(strict_vendor, tmp_path):
    """核心一条：我们生成的声明在**官方严格版判据**下通过，并被记进 training.json。"""
    rec = provenance.resolve("B", baseline.package("B"), tmp_path / "run")
    parent, _src = baseline.parent_checkpoint("B")
    out = tmp_path / "round-prov"
    report = baseline.continue_round("B", out, parent, epochs=1, stage_id="t-prov",
                                     label_provenance=rec["_path"])
    logged = report["label_provenance"]
    assert logged is not None, "严格版应当把声明记进 report"
    assert logged["declared_status"] == "draft"
    assert logged["sha256"].split(":")[-1] == provenance.sha256_file(rec["_path"]).split(":")[-1]
    assert "no health/status label was derived from RUL" in logged["method"]
    # 落地的那份 training.json 也必须带上（评测侧读的是文件，不是我们的返回值）
    written = json.loads((out / "training.json").read_text(encoding="utf-8"))
    assert written["label_provenance"]["declared_status"] == "draft"
