"""证明"给 B 补标签来源声明"没有改变训练本身：同参数跑两轮，比 checkpoint 字节与全部指标。

为什么要单独跑：官方 report 里会写 `label_provenance`，看代码它只进 report 字典
（train.py 第 60/122/129 行），不进模型、不进 RNG、不进 checkpoint 的 15 个字段——
但"看代码觉得没事"和"实测逐位相同"是两回事，这条链上我们已经吃过一次
"说了改了其实没改"的亏（镜像导出那次），所以固化成证据文件。

同时跑第二种场景：把 vendor/B 的训练器换成 A/C 包里那份**严格版**，
证明带声明能跑通、不带声明必抛 ValueError。这是平台预检描述的那条路。

用法（cwd = official/）：
    PYTHONUTF8=0 python scripts/check_b_provenance_parity.py
产出：docs/evidence/b_provenance_parity.json
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import baseline, provenance                       # noqa: E402

PROFILE = "B"
EVIDENCE = ROOT / "docs" / "evidence" / "b_provenance_parity.json"
OUT = ROOT / "_bprov_parity"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_round(land: Path, vendor_root: Path | None, give_declaration: bool) -> dict:
    """跑一轮 B，返回 {指标, checkpoint 字节哈希, training.json 里的 label_provenance}。

    `vendor_root` 不为 None 时把 baseline.VENDOR 指过去（严格版训练器场景）；
    `baselines` 模块在进程里是缓存的，换树前必须清掉，否则 import 到的还是旧的一份。
    """
    for key in [k for k in list(sys.modules) if k == "baselines" or k.startswith("baselines.")]:
        sys.modules.pop(key)
    original_vendor = baseline.VENDOR
    if vendor_root is not None:
        baseline.VENDOR = vendor_root
    try:
        pkg = baseline.package(PROFILE)
        parent, source = baseline.parent_checkpoint(PROFILE)
        prov = provenance.resolve(PROFILE, pkg, land) if give_declaration else None
        out_dir = land / "round"
        report = baseline.continue_round(PROFILE, out_dir, parent, epochs=2, stage_id="parity",
                                         label_provenance=(prov["_path"] if prov else None),
                                         wd=0.1)               # 非平凡配方：真的会动权重
        val = baseline.dataset(PROFILE, "validation")
        metrics = baseline.metrics(PROFILE, out_dir / baseline.ARTIFACT_NAME, val)
        trained = json.loads((out_dir / "training.json").read_text(encoding="utf-8-sig"))
        return {"metrics": metrics, "checkpoint_sha256": sha(out_dir / baseline.ARTIFACT_NAME),
                "training_label_provenance": trained.get("label_provenance"),
                "parent_source": source, "steps": report.get("final_steps"),
                "declaration_kind": (prov or {}).get("_kind") if prov else None}
    finally:
        baseline.VENDOR = original_vendor


def strict_tree(dest: Path) -> Path:
    """vendor/B 的副本，但 baselines/train.py 换成 A 包里那份严格版（profile B 一律要声明）。"""
    tree = dest / "vendor_strict"
    if tree.exists():
        shutil.rmtree(tree)
    shutil.copytree(baseline.VENDOR / PROFILE, tree / PROFILE,
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(ROOT / "vendor" / "A" / "baselines" / "train.py",
                    tree / PROFILE / "baselines" / "train.py")
    return tree


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    result = {"profile": PROFILE, "vendor_train_sha256": {},
              "code_sha": subprocess.run(["git", "-C", str(ROOT.parent), "rev-parse", "--short", "HEAD"],
                                         capture_output=True, text=True).stdout.strip() or "nogit"}
    for letter in baseline.PROFILES:
        f = baseline.VENDOR / letter / "baselines" / "train.py"
        result["vendor_train_sha256"][letter] = sha(f)[:16] if f.is_file() else None

    # 1) 我们实际带的（宽松版）训练器：有声明 vs 无声明
    relaxed_with = run_round(OUT / "relaxed_with", None, True)
    relaxed_without = run_round(OUT / "relaxed_without", None, False)
    same_metrics = relaxed_with["metrics"] == relaxed_without["metrics"]
    same_bytes = relaxed_with["checkpoint_sha256"] == relaxed_without["checkpoint_sha256"]
    result["relaxed"] = {"with_declaration": relaxed_with, "without_declaration": relaxed_without,
                         "metrics_identical": same_metrics, "checkpoint_bytes_identical": same_bytes,
                         "note": "宽松版（=官方 B 包原样那份）会忽略声明：report 里两行都该是 null"}

    # 2) 严格版训练器（平台预检描述的那条路）：有声明必须跑通，无声明必须抛
    tree = strict_tree(OUT)
    strict_with = run_round(OUT / "strict_with", tree, True)
    raised = ""
    try:
        run_round(OUT / "strict_without", tree, False)
    except Exception as exc:                                 # noqa: BLE001 这就是要验的失败模式
        raised = f"{type(exc).__name__}: {exc}"
    result["strict"] = {
        "with_declaration_ok": strict_with["training_label_provenance"] is not None,
        "declared_status": (strict_with["training_label_provenance"] or {}).get("declared_status"),
        "checkpoint_sha256": strict_with["checkpoint_sha256"],
        "metrics": strict_with["metrics"],
        "without_declaration_raises": raised or None}

    # 3) 两个版本的指标必须一致——声明只该动 metadata，不该动模型
    result["cross_trainer_metrics_identical"] = strict_with["metrics"] == relaxed_with["metrics"]
    result["pass"] = bool(same_metrics and same_bytes
                          and result["strict"]["with_declaration_ok"]
                          and raised and result["cross_trainer_metrics_identical"])

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("→", EVIDENCE)
    if not result["pass"]:
        print("✗ 未通过：补声明必须满足「指标与 checkpoint 逐位不变 + 严格版有声明能跑、无声明必抛」",
              file=sys.stderr)
        return 1
    print("✓ B 的标签来源声明不影响训练本身；严格版训练器下这轮能跑")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
