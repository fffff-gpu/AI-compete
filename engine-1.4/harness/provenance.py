"""B 的标签来源说明（label provenance）：只转述官方字段，绝不编造审批。

为什么需要这个模块（2026-11-15 平台预检反馈 + 用 `官方基线工业模型源码（不许改）` 逐位核对）：

* 官方 A/C 包的 `baselines/README.md` 第 13 行明确要求：**"B 训练必须传入标签来源说明，
  `status` 可为 `draft` 或 `approved`，且必须说明 `method`。这里仅记录来源文件的摘要和
  声明状态，不能证明委员会已批准；正式准入仍由平台负责。"** 同文件第 35 行给出约定路径
  `--label-provenance /public/B/labels-provenance.json`。
* 但三个包里发的 `train.py` 不是同一份：A/C 那份（sha256 `77043a0a…`）对
  `profile == "B"` **一律**要求声明；B 自己那份（sha256 `6257f0b1…`）把条件收窄成
  `profile == "B" and task == "classification"`，而 B 是回归——所以我们的十轮 B 从没触发，
  平台侧一上严格版就在开训前 `ValueError: B requires explicit label provenance`。
* 唯一同时满足两边的做法：**训练器代码保持与原版逐位相同**（不动 `vendor/B`，这样
  `training.json` 的 `source_sha256` 仍等于官方，谁也说不出我们改了训练器），
  由 harness 每轮**显式传入**一份转述官方字段的声明。严格版把它记进 `training.json`
  的 `label_provenance`，宽松版忽略它；两种情况下指标都不受影响——`provenance` 只进
  report 字典，不进模型、不进 RNG、不进 checkpoint 的 15 个字段（`train.py` 里可读）。

三条硬规矩，`tests/test_provenance.py` 逐条钉住：

1. **不自封 approved**：我们生成的声明一律 `status="draft"`。官方包里 `label_status` 的
   原值（B 是 `approved`）作为**被引用的事实**写进 `official_label_status`，并带源文件
   sha256；判准入是平台的权力，我们自己给自己盖 "approved" 才是造假。
2. **官方给了就用官方的，一字不改**：显式 `--label-provenance` / `ZHIXUN_LABEL_PROVENANCE`
   优先，其次是包里自带的 `public/B/labels-provenance.json`（README 的约定路径）。
   这类文件只读、只引用，绝不改写；若它本身不合格式，记录原因后回退到我们自己转述的那份，
   而不是偷偷替组委会"修好"它。
3. **无话可说就报错，不静默跳步**：官方字段一个都取不到时抛异常，并列出查过哪些文件及其
   sha256。凭空编一段 `method` 正是赛规要防的"标签来源不实"。
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

VALID_STATUS = ("draft", "approved")

# 官方 README 里的约定名；投影包可能把它们放在包根或 public/ 下，三种都试。
CONVENTION_NAMES = ("public/B/labels-provenance.json",
                    "public/labels-provenance.json",
                    "labels-provenance.json")

ENV_PATH = "ZHIXUN_LABEL_PROVENANCE"

# 官方 baseline-info.json 里可作为"标签来源"证据的字段，以及它们在声明里的中文说明。
_CITABLE = ("label_status", "target_origin", "target_definition", "target_unit")


def sha256_file(path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_obj(path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _valid_declaration(obj: dict) -> tuple[bool, str]:
    """按官方训练器第 61 行的判据检查一份声明（我们不改它，只能提前自查）。"""
    if obj.get("status") not in VALID_STATUS:
        return False, "status 必须是 draft 或 approved，实际为 {!r}".format(obj.get("status"))
    method = obj.get("method")
    if not isinstance(method, str) or not method.strip():
        return False, "method 必须是非空字符串"
    return True, ""


def package_declaration(pkg: Path) -> Path | None:
    """包里是否自带官方约定的标签来源文件（有就原样引用，一个字节都不改）。"""
    for name in CONVENTION_NAMES:
        cand = Path(pkg) / name
        if cand.is_file():
            return cand
    return None


def citable_metadata(pkg: Path, profile: str = "B") -> tuple[dict, str]:
    """取官方包自己写下的标签口径。返回 (字段, 主要来源文件, 是否完整)。

    优先级与"完整度"（这条直接决定声明能写多少，写不出来就必须标 missing）：

    1. `baseline-info.json`：有 target_origin/target_definition → **完整**（本地 B 包就是这样）；
    2. `training.json`：同样带 target_* → 完整；
    3. `baselines/configs/<P>.json` + `checkpoint.npz` 的 metadata：平台每轮发的投影包
       （`public/{train,validation}.jsonl` + `checkpoint.npz` + `baselines/`）里只有这些，
       能引用 task/profile/target_unit，**引用不到 target_origin/target_definition** → 不完整，
       缺什么必须在声明里点名，不能拿 target_unit 冒充标签派生定义。
    """
    pkg = Path(pkg)
    info = pkg / "baseline-info.json"
    if info.is_file():
        obj = _json_obj(info) or {}
        fields = {key: obj[key] for key in _CITABLE if key in obj}
        if fields.get("target_definition") or fields.get("target_origin"):
            fields.setdefault("dataset_key", obj.get("dataset_key"))
            fields.setdefault("package_id", obj.get("id"))
            return fields, str(info), True
    report = pkg / "training.json"
    if report.is_file():
        obj = _json_obj(report) or {}
        fields = {key: obj[key] for key in _CITABLE if key in obj}
        if fields.get("target_definition") or fields.get("target_origin"):
            return fields, str(report), True
    # 投影包兜底：官方 config 的 profile 字段是训练器自己校验过的（train.py:48），
    # checkpoint metadata 的 target_unit 是训练器写进去的（train.py:85）——都是官方声明。
    fields: dict = {}
    primary = ""
    cfg_dir = pkg / "baselines" / "configs"
    if cfg_dir.is_dir():
        for cand in sorted(cfg_dir.glob("*.json")):
            obj = _json_obj(cand) or {}
            if str(obj.get("profile", "")).upper() == (profile or "").upper():
                fields.update({"profile": obj.get("profile"), "task": obj.get("task"),
                               "target_unit": obj.get("target_unit"),
                               "classes": obj.get("classes")})
                fields["dataset_key"] = str(obj.get("profile")).upper()
                primary = str(cand)
                break
    own = pkg / "checkpoint.npz"
    if own.is_file():
        try:
            from .baseline import checkpoint_meta
            meta = checkpoint_meta(own) or {}
            for key in ("target_unit", "task", "profile"):
                fields.setdefault(key, meta.get(key))
            primary = primary or str(own)
        except Exception:                                # 读不动就别用它，不要拿半截结论当依据
            pass
    fields = {k: v for k, v in fields.items() if v is not None}
    return fields, primary, False


def consulted_sources(pkg: Path, profile: str = "B") -> list[dict]:
    """把所有查过的官方文件连哈希一起交出去——"我们引用了什么"必须能被别人复算。"""
    out = []
    for name in ("baseline-info.json", "training.json", "labels-provenance.json", "checkpoint.npz"):
        cand = Path(pkg) / name
        if cand.is_file():
            out.append({"file": name, "sha256": sha256_file(cand)})
    cfg_dir = Path(pkg) / "baselines" / "configs"
    if cfg_dir.is_dir():
        for cand in sorted(cfg_dir.glob("*.json")):
            out.append({"file": str(cand.relative_to(pkg)).replace("\\", "/"),
                        "sha256": sha256_file(cand)})
    return out


def _method_text(profile: str, meta: dict, complete: bool) -> str:
    """method 只允许转述官方原文 + 一句"本队没做什么"的陈述，不引入任何新口径。

    **必须是纯 ASCII**：官方 train.py 写 `training.json` 时没有指定 encoding
    （`(staging / "training.json").write_text(...)`，本机实测按 GBK 落盘），
    而 report 里会带上这段 method。中文声明会让官方产物在 Windows 上变成非 UTF-8，
    我们的复核工具（audit / pack / compare，全按 UTF-8 读）当场解不开。
    中文解释放 `docs/label-provenance-B.md`，进产物的字段留英文。
    """
    src = Path(meta["_source"]).name if meta.get("_source") else "package metadata"
    quoted = "; ".join("{}={}".format(key, meta[key]) for key in
                       ("target_origin", "target_definition", "target_unit", "label_status")
                       if meta.get(key) is not None)
    head = "Official package {} declares in {}: {}. ".format(
        meta.get("package_id") or "this round's package", src, quoted)
    if not complete:
        missing = [k for k in ("target_origin", "target_definition", "label_status") if not meta.get(k)]
        head += ("This round's package does not carry {} (absent from the projection package), "
                 "so those are recorded as missing rather than inferred. ".format("/".join(missing)))
    return (head +
            "This team paraphrases the official fields above verbatim and trained on the answer field "
            "already present in the official public/train.jsonl; no health/status label was derived "
            "from RUL, and no hidden label was read, inferred or constructed. label_status, when "
            "present, is quoted, not an approval claimed by the team (official README: the recorded "
            "status cannot prove committee approval; admission is decided by the platform).")


def synthesize(profile: str, pkg: Path, land_dir: Path, now: str | None = None) -> dict:
    """把官方字段转述成一份声明，落在可写目录（绝不写进官方包）。"""
    meta, source, complete = citable_metadata(Path(pkg), profile)
    if not meta:
        seen = [{"file": name, "present": (Path(pkg) / name).is_file()}
                for name in ("baseline-info.json", "training.json")]
        raise ValueError(
            "profile {} 需要标签来源说明，但官方包 {} 里没有任何可转述的官方字段（查过：{}）。"
            "请勿让我们编造 method：应由组委会提供 labels-provenance.json，"
            "或用 --label-provenance / {} 指向它。".format(
                profile, pkg, seen, ENV_PATH))
    meta = dict(meta)
    meta["_source"] = source
    doc = {
        # 我们 authored 的声明永远是 draft：判 approved 是平台/组委会的事。
        "status": "draft",
        "method": _method_text(profile, meta, complete),
        "dataset_key": meta.get("dataset_key") or profile,
        "official_label_status": meta.get("label_status"),
        "target_origin": meta.get("target_origin"),
        "target_definition": meta.get("target_definition"),
        "target_unit": meta.get("target_unit"),
        # 投影包只带 configs/checkpoint 时，完整性要如实标出来：缺的是标签派生定义，
        # 不是我们能在本地补齐的东西——它决定评审能不能把这轮当"已声明口径"采信。
        "complete": bool(complete),
        "missing_fields": ([] if complete else
                           [k for k in ("target_origin", "target_definition", "label_status")
                            if not meta.get(k)]),
        "source": {"package": Path(pkg).name, "file": Path(source).name if source else None,
                   "sha256": sha256_file(source) if source else None,
                   "fields": [k for k in _CITABLE if k in meta],
                   "consulted": consulted_sources(Path(pkg), profile)},
        "declared_by": "participating team (paraphrase of official fields; not a committee approval)",
        "generated_at": now or datetime.now(timezone.utc).isoformat(),
        "note": "Generated by harness.provenance from fields already present in the official package. "
                "Official README: the recorded status cannot prove committee approval; admission is the "
                "platform's decision. Chinese write-up: docs/label-provenance-B.md",
    }
    ok, why = _valid_declaration(doc)
    if not ok:                                   # 转述不出合规声明就是缺材料，不是我们的自由发挥空间
        raise ValueError("生成的标签来源说明不合规：{}（源：{}）".format(why, source))
    blob = json.dumps(doc, ensure_ascii=False)
    if not blob.isascii():
        # 见 _method_text 的注释：官方 train.py 写 training.json 不指定 encoding，
        # 而 report 会带上 method 原文。整份声明保持 ASCII，产物编码就与平台/本机无关。
        bad = sorted({c for c in blob if ord(c) > 127})
        raise ValueError("标签来源说明必须全是 ASCII，出现非 ASCII 字符：{}".format(" ".join(bad)))
    land = Path(land_dir)
    land.mkdir(parents=True, exist_ok=True)
    path = land / "{}-labels-provenance.json".format(profile)
    path.write_text(json.dumps(doc, ensure_ascii=True, indent=2) + "\n",
                    encoding="ascii")
    doc["_path"], doc["_kind"] = str(path), "synthesized"
    return doc


def resolve(profile: str, pkg: Path, land_dir: Path | None = None,
            explicit: str | os.PathLike | None = None, now: str | None = None) -> dict | None:
    """返回本轮该用的标签来源声明；非 B（或无需声明）返回 None。

    阶梯：显式指定 > 包内官方约定文件 > 我们转述官方字段生成 > 抛错。
    `explicit` 不给时读环境变量 `ZHIXUN_LABEL_PROVENANCE`——投影包那侧如果按 README 的
    路径放了一份，平台只要在环境里说一个路径，我们就原样引用它，不改镜像代码。
    前两档**只读不改**；第三档落在可写目录；第四档是故意的——静默跳过等于
    在严格训练器下让 B 整轮失败，而失败原因会被压成一句 ValueError。
    """
    if explicit is None:
        explicit = os.environ.get(ENV_PATH) or None
    if (profile or "").upper() != "B":
        return None                     # A/C 在两种训练器下都不要求；不传就别传

    for kind, cand in (("explicit", explicit), ("package", package_declaration(Path(pkg)))):
        if not cand:
            continue
        cand = Path(cand)
        if cand.is_dir():
            raise ValueError("{} 指向的是目录，标签来源说明必须是单个 JSON 文件：{}"
                             .format("--label-provenance" if kind == "explicit" else "包内约定路径", cand))
        if not cand.is_file():
            if kind == "explicit":
                raise FileNotFoundError("指定的标签来源说明不存在：{}".format(cand))
            continue
        obj = _json_obj(cand)
        if obj is None:
            print("[provenance] {} 不是合法 JSON，忽略并改用转述生成的声明".format(cand), flush=True)
            continue
        ok, why = _valid_declaration(obj)
        if not ok:
            print("[provenance] {} 不合规（{}）：官方原文件不改写，改用转述生成的声明；"
                  "请组委会修正该文件或直接采用它。".format(cand, why), flush=True)
            continue
        return {"_path": str(cand), "_kind": kind, "status": obj["status"],
                "method": obj["method"], "sha256": sha256_file(cand)}

    if land_dir is None:
        raise ValueError("profile B 需要标签来源说明，但没有可写目录来放置生成的声明"
                         "（run.py 传 out/，或设置 --label-provenance / {}）".format(ENV_PATH))
    return synthesize(profile, Path(pkg), Path(land_dir), now=now)


def preview(profile: str, pkg: Path, explicit: str | os.PathLike | None = None) -> dict:
    """**不落盘**地回答"这一轮会交哪份声明"。selfcheck 与 boot_banner 用它。

    为什么单独写一个：诊断通道的唯一用途是失败时把现场贴回来，它绝不能反过来产生副作用
    （在只读挂载上尝试写文件、或让 selfcheck 跑一半把 B 的声明生成出来）。
    """
    if explicit is None:
        explicit = os.environ.get(ENV_PATH) or None
    out = {"required": (profile or "").upper() == "B", "kind": None, "status": None,
           "path": None, "official_label_status": None, "citable": False, "complete": False}
    meta, _src, complete = citable_metadata(Path(pkg), profile)
    out["citable"] = bool(meta)
    out["complete"] = bool(complete)
    out["official_label_status"] = meta.get("label_status")
    if not out["required"]:
        return out
    for kind, cand in (("explicit", explicit), ("package", package_declaration(Path(pkg)))):
        if cand and Path(cand).is_file():
            obj = _json_obj(Path(cand))
            ok, _why = _valid_declaration(obj) if obj else (False, "不是合法 JSON")
            out.update(kind=kind, path=str(cand), status=obj.get("status") if obj else None,
                       valid=bool(ok))
            return out
    out.update(kind="synthesized" if out["citable"] else "MISSING",
               status="draft" if out["citable"] else None, path=None)
    return out


def journal_fields(rec: dict | None) -> dict:
    """进 journal / lineage 的最小字段（哈希而非全文，全文已在声明文件里）。"""
    if not rec:
        return {"label_provenance": None}
    return {"label_provenance": rec["_path"],
            "label_provenance_kind": rec["_kind"],
            "label_provenance_status": rec.get("status"),
            "label_provenance_sha256": rec.get("sha256") or sha256_file(rec["_path"])}
