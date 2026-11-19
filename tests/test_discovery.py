"""自适应发现层的独立单测：全部用合成小目录，不读真实数据包，也不训练。

这些用例的存在理由：平台把官方包放在「指定平台固定目录下」，目录名、父检查点位置、
产物目录都没公开约定（规则第 85 行只说了固定目录）。所以「认不出布局」和
「把我们自己的历史输出当成平台父亲」都必须是有测试覆盖的失败模式，而不是上线才发现。
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from harness import baseline


def write_ckpt(path, profile="A", steps=80, epochs=20, hidden=32, fmt=None,
               task="classification"):
    """造一个 metadata 结构与官方一致的假检查点（发现层只看 metadata）。"""
    meta = {"format": fmt or baseline.CHECKPOINT_FORMAT, "profile": profile, "seed": 42,
            "hidden": hidden, "steps": steps, "epochs": epochs, "task": task,
            "classes": ["a", "b"] if task == "classification" else [],
            "paths": ["inputs.features[0]"], "parameter_count": 10,
            "learning_rate": 0.003, "batch_size": 128, "device": "cpu",
            "target_mean": 0.0, "target_scale": 1.0}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, w1=np.zeros((2, 2)), b1=np.zeros(2), metadata=np.array(json.dumps(meta)))
    return path


def make_package(root, profile="A", name=None, info=True, data=True, ckpt=True):
    directory = Path(root) / (name or f"cpu-{profile.lower()}-guide-20261110-v1")
    directory.mkdir(parents=True, exist_ok=True)
    if info:
        (directory / "baseline-info.json").write_text(
            json.dumps({"dataset_key": profile, "task": "classification"}), encoding="utf-8")
    if data:
        public = directory / "public"
        public.mkdir(exist_ok=True)
        (public / "train.jsonl").write_text('{"id":"t1","inputs":{},"answer":"a"}\n', encoding="utf-8")
        (public / "validation.jsonl").write_text('{"id":"v1","inputs":{},"answer":"a"}\n', encoding="utf-8")
    configs = directory / "baselines" / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    for letter in baseline.PROFILES:
        (configs / f"{letter}.json").write_text(
            json.dumps({"profile": letter, "task": "classification", "hidden": 32, "seed": 42}),
            encoding="utf-8")
    if ckpt:
        write_ckpt(directory / baseline.ARTIFACT_NAME, profile=profile)
    return directory


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """把 WORKSPACE/PACKAGES 指向临时目录，清空相关环境变量，并重置发现缓存。"""
    root = tmp_path / "platform"
    root.mkdir()
    monkeypatch.setattr(baseline, "WORKSPACE", root)
    monkeypatch.setattr(baseline, "PACKAGES", {p: root / f"cpu-{p.lower()}-guide-20261110-v1"
                                               for p in baseline.PROFILES})
    for key in (*baseline.DATA_ENVS, *baseline.CKPT_ENVS, *baseline.ARTIFACT_ENVS):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ZHIXUN_DATA", str(root))
    monkeypatch.setenv("ZHIXUN_DATA_STRICT", "1")     # 不回退扫真实工作区/文件系统根
    baseline.discover_packages._cache = None
    yield root
    baseline.discover_packages._cache = None


# ── 包识别 ─────────────────────────────────────────────────────────────
def test_dataset_key_is_authoritative_even_if_dir_name_is_weird(sandbox):
    directory = make_package(sandbox, "A", name="dataset_7")
    assert baseline._profile_of_dir(directory) == "A"


def test_profile_from_checkpoint_metadata_when_info_file_absent(sandbox):
    directory = make_package(sandbox, "C", name="turbofan", info=False)
    assert baseline._profile_of_dir(directory) == "C"


def test_profile_from_dir_name_when_neither_present(sandbox):
    directory = make_package(sandbox, "B", info=False, ckpt=False)
    assert baseline._profile_of_dir(directory) == "B"


def test_artifact_output_is_never_mistaken_for_a_package(sandbox):
    """只有 checkpoint.npz 的产物目录不是包（实测踩过：入口拷进 _artifacts/A/ 的
    检查点被认成 A 的数据包，package(A) 于是指向一个没有 public/ 的目录）。"""
    package = make_package(sandbox, "A")
    artifact = sandbox / "_artifacts" / "A"
    write_ckpt(artifact / baseline.ARTIFACT_NAME, "A", steps=999)
    assert baseline._profile_of_dir(artifact) is None
    assert baseline.discover_packages()["A"] == package


def test_data_without_package_markers_is_not_a_package(sandbox):
    stray = sandbox / "just_data"
    stray.mkdir()
    (stray / "checkpoint.npz").write_text("not an npz", encoding="utf-8")
    assert baseline._profile_of_dir(stray) is None


def test_discover_walks_two_levels_and_caches(sandbox):
    inner = sandbox / "datasets"
    make_package(inner, "A", name="alpha")
    make_package(sandbox, "B", name="beta")
    found = baseline.discover_packages()
    assert found["A"] == inner / "alpha"
    assert found["B"] == sandbox / "beta"
    assert baseline.discover_packages() is found          # 缓存生效（每进程只扫一次）


def test_package_explicit_root_beats_the_default_convention(sandbox, monkeypatch):
    """显式给了 ZHIXUN_DATA 就以它为准：默认约定名可能与平台根里的包不是同一份数据。"""
    make_package(sandbox, "A")                             # 默认约定名也存在
    alpha = make_package(sandbox, "A", name="alpha")
    assert baseline.package("A") == alpha                  # 显式根优先
    monkeypatch.delenv("ZHIXUN_DATA")
    baseline.discover_packages._cache = None
    assert baseline.package("A") == baseline.PACKAGES["A"]  # 没给显式根才用约定名


def test_package_falls_back_to_discovery_when_default_broken(sandbox, monkeypatch):
    make_package(sandbox, "A", name="alpha")
    monkeypatch.delenv("ZHIXUN_DATA")                      # 走「约定名不存在」这条路
    monkeypatch.delenv("ZHIXUN_DATA_STRICT")               # 允许回退到 WORKSPACE
    baseline.discover_packages._cache = None
    assert baseline.package("A") == sandbox / "alpha"


def test_missing_package_error_names_roots_and_search(sandbox):
    with pytest.raises(FileNotFoundError) as exc:
        baseline.package("C")
    message = str(exc.value)
    assert "ZHIXUN_DATA" in message
    assert str(sandbox) in message or str(sandbox.resolve()) in message


# ── 父检查点发现 ───────────────────────────────────────────────────────
def test_parent_defaults_to_package_checkpoint(sandbox):
    make_package(sandbox, "A")
    path, source = baseline.parent_checkpoint("A")
    assert source == "package" and path.parent == baseline.PACKAGES["A"]


def test_external_platform_checkpoint_beats_package(sandbox):
    package = make_package(sandbox, "A")
    platform = sandbox / "parent" / "A" / baseline.ARTIFACT_NAME
    write_ckpt(platform, "A", steps=120)
    path, source = baseline.parent_checkpoint("A")
    assert source == "scan" and path == platform


def test_env_var_outranks_scan_and_cli_outranks_everything(sandbox, monkeypatch):
    make_package(sandbox, "A")
    write_ckpt(sandbox / "parent" / "A" / baseline.ARTIFACT_NAME, "A", steps=120)
    env_ckpt = write_ckpt(sandbox / "from_env.npz", "A", steps=200)
    cli_ckpt = write_ckpt(sandbox / "from_cli.npz", "A", steps=300)
    monkeypatch.setenv("CHECKPOINT_PATH", str(env_ckpt))
    path, source = baseline.parent_checkpoint("A")
    assert (path, source) == (env_ckpt, "env:CHECKPOINT_PATH")
    path, source = baseline.parent_checkpoint("A", explicit=cli_ckpt)
    assert (path, source) == (cli_ckpt, "cli")


def test_env_checkpoint_directory_is_accepted(sandbox, monkeypatch):
    make_package(sandbox, "A")
    directory = sandbox / "model_out"
    write_ckpt(directory / baseline.ARTIFACT_NAME, "A", steps=150)
    monkeypatch.setenv("ZHIXUN_CKPT", str(directory))
    path, source = baseline.parent_checkpoint("A")
    assert path == directory / baseline.ARTIFACT_NAME and source.startswith("env:")


def test_missing_explicit_checkpoint_raises(sandbox):
    make_package(sandbox, "A")
    with pytest.raises(FileNotFoundError):
        baseline.parent_checkpoint("A", explicit=sandbox / "nope.npz")


def test_wrong_profile_checkpoint_is_not_used_as_parent(sandbox):
    make_package(sandbox, "A")
    write_ckpt(sandbox / "parent" / "B" / baseline.ARTIFACT_NAME, "B", steps=999)
    path, source = baseline.parent_checkpoint("A")
    assert source == "package"


def test_corrupt_ckpt_is_skipped_not_fatal(sandbox):
    make_package(sandbox, "A")
    bad = sandbox / "parent" / "A"
    bad.mkdir(parents=True)
    (bad / baseline.ARTIFACT_NAME).write_text("garbage", encoding="utf-8")
    path, source = baseline.parent_checkpoint("A")
    assert source == "package"


def test_our_own_run_outputs_never_become_the_parent(sandbox):
    """红线：把 runs/*/best/checkpoint.npz 当平台父亲，等于自报来路不明的检查点。"""
    make_package(sandbox, "A")
    for relative in ("runs/a_10r/best/checkpoint.npz", "runs/a_10r/round-03/checkpoint.npz",
                     "submission/03-模型权重/A/checkpoint.npz"):
        write_ckpt(sandbox / relative, "A", steps=500)
    path, source = baseline.parent_checkpoint("A")
    assert source == "package"


def test_is_our_output_matches_scratch_and_staging(sandbox):
    assert baseline._is_our_output(Path("/data/runs/a/best/checkpoint.npz"))
    assert baseline._is_our_output(Path("/data/official/_scratch/x.npz"))
    assert baseline._is_our_output(Path("/data/.baseline-abc123/result/checkpoint.npz"))
    assert not baseline._is_our_output(Path("/data/official/parent/A/checkpoint.npz"))


# ── 环境变量与根目录 ───────────────────────────────────────────────────
def test_env_paths_keeps_windows_drive_letter(sandbox, monkeypatch):
    monkeypatch.setenv("ZHIXUN_DATA", r"D:\data\platform")
    paths = baseline._env_paths(("ZHIXUN_DATA",))
    assert paths == [Path(r"D:\data\platform")]


def test_data_roots_dedupes_and_drops_missing(sandbox):
    roots = baseline.data_roots()
    target = sandbox.resolve()
    assert roots.count(target) == 1
    assert all(root.is_dir() for root in roots)


def test_strict_mode_does_not_fall_back_to_the_local_workspace(sandbox, monkeypatch):
    """镜像里 ROOT.parent 是 /，本机是真实工作区；严格模式只认显式给的根。"""
    monkeypatch.setenv("ZHIXUN_DATA_STRICT", "1")
    strict = baseline.data_roots()
    monkeypatch.setenv("ZHIXUN_DATA_STRICT", "0")
    relaxed = baseline.data_roots()
    assert strict == [sandbox.resolve()]
    assert len(relaxed) > len(strict)                  # 非严格时多出隐式回退根


def test_filesystem_root_is_never_scanned(sandbox, monkeypatch):
    import os

    root = "/" if os.name == "posix" else str(sandbox.resolve().drive) + "\\"
    monkeypatch.setenv("ZHIXUN_DATA", root)
    assert baseline.data_roots() == []


def test_artifact_dir_uses_env_and_creates_it(sandbox, monkeypatch):
    target = sandbox / "artifacts"
    monkeypatch.setenv("ZHIXUN_ARTIFACT_DIR", str(target))
    assert baseline.artifact_dir() == target and target.is_dir()


def test_checkpoint_meta_reads_real_official_shape(tmp_path):
    path = write_ckpt(tmp_path / "x.npz", "C", steps=6630, epochs=15, task="regression")
    meta = baseline.checkpoint_meta(path)
    assert meta["profile"] == "C" and meta["steps"] == 6630 and meta["task"] == "regression"


# ── 扁平数据布局（平台可能不套包目录）──────────────────────────────────
def _write_flat(pair_dir, features=128, task="classification"):
    import json as _json

    pair_dir.mkdir(parents=True, exist_ok=True)
    answer = "Ethanol" if task == "classification" else 100.0
    row = {"id": "x1", "inputs": {"features": [0.0] * features}, "answer": answer}
    (pair_dir / "train.jsonl").write_text(_json.dumps(row) + "\n", encoding="utf-8")
    (pair_dir / "validation.jsonl").write_text(_json.dumps(row) + "\n", encoding="utf-8")
    return pair_dir / "train.jsonl", pair_dir / "validation.jsonl"


def _package_without_data(sandbox, profile="A", paths=None):
    """建一个「有代码与检查点、但 public/ 被拿掉」的包，逼 data_files 走扁平兜底。"""
    package = make_package(sandbox, profile)
    shutil.rmtree(package / "public")
    if paths:
        _patch_meta(package / baseline.ARTIFACT_NAME, {"paths": paths})
    return package


def test_flat_pair_is_found_and_matched_by_feature_count(sandbox):
    _package_without_data(sandbox, "A",
                          paths=[f"inputs.features[{i}]" for i in range(128)])
    _write_flat(sandbox / "flat" / "eval", features=128)
    train, val = baseline.data_files("A")
    assert train.parent.name == "eval" and val.parent == train.parent


def test_flat_pair_with_wrong_feature_count_is_rejected(sandbox):
    _package_without_data(sandbox, "A",
                          paths=[f"inputs.features[{i}]" for i in range(128)])
    _write_flat(sandbox / "flat", features=34)
    with pytest.raises(FileNotFoundError) as exc:
        baseline.data_files("A")
    message = str(exc.value)
    assert "期望特征数=128" in message and "特征数 34" in message


def test_flat_pair_with_wrong_answer_kind_is_rejected(sandbox):
    """回归目标（num）绝不能被绑给分类 profile（str）——静默用错数据是最坏失败。"""
    _package_without_data(sandbox, "A",
                          paths=[f"inputs.features[{i}]" for i in range(128)])
    _write_flat(sandbox / "flat", features=128, task="regression")
    with pytest.raises(FileNotFoundError):
        baseline.data_files("A")


def test_env_train_val_override_wins(sandbox, monkeypatch):
    train, val = _write_flat(sandbox / "explicit")
    monkeypatch.setenv("ZHIXUN_TRAIN_FILE", str(train))
    monkeypatch.setenv("ZHIXUN_VALIDATION_FILE", str(val))
    assert baseline.data_files("C") == (train, val)


def test_data_error_names_the_searched_roots(sandbox):
    with pytest.raises(FileNotFoundError) as exc:
        baseline.data_files("B")
    assert str(sandbox) in str(exc.value) or str(sandbox.resolve()) in str(exc.value)


def _patch_meta(ckpt, updates) -> None:
    """就地改写假检查点的 metadata（保留全部张量，只换 metadata）。"""
    with np.load(ckpt, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    meta = json.loads(str(payload["metadata"].item()))
    meta.update(updates)
    payload["metadata"] = np.array(json.dumps(meta))
    ckpt.unlink()
    np.savez(ckpt, **payload)


# ── 命令透传：平台追加命令时不能被当成未知参数吞掉 ────────────────────
def test_as_command_passes_through_interpreter_invocations():
    from harness.entrypoint import as_command

    assert as_command(["python", "-m", "harness.predict", "--selfcheck"])
    assert as_command(["/usr/local/bin/python", "predict.py"])
    assert as_command(["bash", "-c", "echo hi"])
    # 裸子命令映射到对应模块，参数原样跟在后面
    assert as_command(["predict"]) == [sys.executable, "-m", "harness.predict"]
    assert as_command(["predict", "--data", "x.jsonl"]) == [
        sys.executable, "-m", "harness.predict", "--data", "x.jsonl"]
    assert as_command(["audit", "runs/a_v2x"]) == [
        sys.executable, "-m", "harness.audit", "runs/a_v2x"]


def test_as_command_leaves_flags_to_argparse():
    from harness.entrypoint import as_command

    assert as_command(["--selfcheck"]) == []
    assert as_command(["--profiles", "A,B"]) == []
    assert as_command([]) == []


def test_entrypoint_executes_command_instead_of_swallowing_it(tmp_path, monkeypatch):
    """回归：镜像 ENTRYPOINT 是 entrypoint，平台若追加命令，过去会被 parse_known_args 忽略，
    于是"我们要跑闭环、平台要的是预测"，表现就是镜像可用但评测运行失败。"""
    from harness import entrypoint

    seen = tmp_path / "ran.txt"
    argv = [sys.executable, "-c", f"open(r'{seen}','w').write('ok')"]
    monkeypatch.setattr(sys, "argv", ["harness.entrypoint"] + argv)
    rc = entrypoint.main(argv)
    assert rc == 0 and seen.read_text(encoding="utf-8") == "ok"


# ── 产物目录候选（平台常把产物目录固定叫 /output 或 /runs）──────────────
def test_artifact_dir_prefers_env_then_common_container_paths(sandbox, monkeypatch):
    from harness.baseline import artifact_dir

    explicit = sandbox / "artifacts_env"
    monkeypatch.setenv("ZHIXUN_ARTIFACT_DIR", str(explicit))
    assert artifact_dir() == explicit and explicit.is_dir()
    monkeypatch.delenv("ZHIXUN_ARTIFACT_DIR")
    fallback = sandbox / "given_default"
    assert artifact_dir(default=fallback) == fallback     # 没有 env、没有 /output 时用调用方给的
