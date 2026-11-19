"""Thin adapter around the official CPU baseline package.

The harness never re-implements the model: it calls `baselines.train.train_run`
so every checkpoint keeps the official provenance (parent digest, Adam state,
normalisation fitted on the initial training set).

Training-recipe knobs (weight decay / label smoothing / input jitter / clip) are
read from the environment by the vendored copy in `official/vendor/<profile>`;
the checkpoint format is unchanged.
"""
import contextlib
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from . import ROOT

WORKSPACE = Path(os.environ.get("ZHIXUN_WORKSPACE") or ROOT.parent)
PACKAGES = {
    "A": WORKSPACE / "cpu-a-guide-20261110-v1",
    "B": WORKSPACE / "cpu-b-rul-20261110-v1",
    "C": WORKSPACE / "cpu-c-guide-20261110-v1",
}
VENDOR = ROOT / "vendor"
RECIPE_ENV = ("ZHIXUN_WD", "ZHIXUN_LS", "ZHIXUN_JITTER", "ZHIXUN_CLIP",
              "ZHIXUN_LOSS", "ZHIXUN_CENTER", "ZHIXUN_SEV", "ZHIXUN_SEV_END", "ZHIXUN_SWA")

# 平台侧的路径约定我们并不知道（规则只说「上传至指定平台固定目录下」），
# 所以入口路径一律按「环境变量 → 常见挂载点 → 扫描识别」的顺序自适应，不认死目录名。
DATA_ENVS = ("ZHIXUN_DATA", "ZHIXUN_WORKSPACE", "DATA_DIR", "DATA_ROOT", "INPUT_DIR", "DATASET_DIR")
CKPT_ENVS = ("ZHIXUN_CKPT", "ZHIXUN_PARENT_CHECKPOINT", "PARENT_CHECKPOINT", "PARENT_CKPT",
             "CHECKPOINT_PATH", "CHECKPOINT", "MODEL_PATH", "MODEL_DIR")
ARTIFACT_ENVS = ("ZHIXUN_ARTIFACT_DIR", "OUTPUT_DIR", "ARTIFACT_DIR", "RESULT_DIR", "SUBMISSION_DIR")
ARTIFACT_NAME = "checkpoint.npz"
CHECKPOINT_FORMAT = "contest.cpu-mlp.v1"
PROFILES = ("A", "B", "C")

# package() 每次选定一个包时把「凭什么选它」记在这儿，诊断通道原样打印。
# 投影包识别靠的是兜底规则，兜底规则必须可见，否则一次误判就能伪装成一次正常评测。
_PACKAGE_REASON: dict = {}


def _env_paths(keys) -> list:
    """把若干环境变量里的路径取出来。

    只按 os.pathsep 切分（Linux 是 `:`、Windows 是 `;`）——不能按冒号切，
    否则 Windows 上的 `D:\\data` 会被切成 `D` 和 `\\data`。
    """
    found = []
    for key in keys:
        raw = os.environ.get(key)
        if not raw:
            continue
        for chunk in raw.split(os.pathsep):
            chunk = chunk.strip().strip('"').strip("'")
            if chunk:
                found.append(Path(chunk))
    return found


def data_roots(extra=None, strict=None) -> list:
    """候选数据根：显式传入 > 环境变量 > /data > 本机工程根。只返回真实存在的目录。

    `ZHIXUN_DATA_STRICT=1` 时只用显式与环境变量给的根，不做隐式回退：
    镜像里 `ROOT.parent` 是 `/`，不加这个开关就会去扫文件系统根；本机跑单测时
    它也会把真实工作区的检查点误当成「平台父亲」。平台侧建议显式设 ZHIXUN_DATA。
    """
    strict = (os.environ.get("ZHIXUN_DATA_STRICT", "") or "0").lower() if strict is None else strict
    cands = list(extra or []) + _env_paths(DATA_ENVS)
    if str(strict).lower() not in ("1", "true", "yes", "on"):
        cands.append(Path("/data"))
        cands += [WORKSPACE, ROOT.parent, ROOT]
    out, seen = [], set()
    for path in cands:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved == resolved.parent:               # 文件系统根（/ 或 D:\\）：不扫
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def checkpoint_meta(checkpoint):
    """不导入 baselines 就读出 checkpoint 的 metadata（zip 里那个 0-d unicode 数组）。"""
    import numpy as np

    with np.load(Path(checkpoint), allow_pickle=False) as data:
        if "metadata" not in data:
            return None
        return json.loads(str(data["metadata"].item()))


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile_of_dir(directory: Path):
    """判断一个目录是不是官方数据包、是哪个 profile。

    先要「像个包」：有 public/train.jsonl、或有 baselines/ 官方代码、或有 baseline-info.json。
    只有一个 checkpoint.npz 的目录是**产物**，不是包（实测踩过：入口拷进 _artifacts/A/ 的
    检查点被当成 A 的数据包，于是 package(A) 指向一个没有数据的目录）。
    识别顺序：baseline-info.json 的 dataset_key（平台目录名可以随便叫）
    → 包内 checkpoint 的 meta.profile → 目录名 cpu-a/cpu-b/cpu-c。
    """
    looks_like_package = ((directory / "public" / "train.jsonl").is_file()
                          or (directory / "baselines").is_dir()
                          or (directory / "baseline-info.json").is_file())
    if not looks_like_package:
        return None
    info = directory / "baseline-info.json"
    if info.exists():
        try:
            key = str(json.loads(info.read_text(encoding="utf-8")).get("dataset_key", "")).upper()
        except (OSError, ValueError):
            key = ""
        if key in PROFILES:
            return key
    own = directory / ARTIFACT_NAME
    if own.is_file():
        try:
            meta = checkpoint_meta(own) or {}
        except Exception:                                    # 坏文件不该让整次扫描失败
            meta = {}
        if meta.get("format") == CHECKPOINT_FORMAT and str(meta.get("profile", "")).upper() in PROFILES:
            return str(meta["profile"]).upper()
    match = re.search(r"cpu-([abc])", directory.name.lower())
    if match:
        return match.group(1).upper()
    # 第四档：投影包（平台每轮给的 current_round_projection）没有 baseline-info.json，目录名也
    # 不含 cpu-x，checkpoint 的 meta 万一读不出来时就只剩 baselines/configs/*.json 里的 profile。
    # 官方 config 的 profile 字段是训练器自己校验过的（train.py:48 `config["profile"]`），
    # 所以用它认包不是我们猜测，是读官方声明。
    cfg_dir = directory / "baselines" / "configs"
    if cfg_dir.is_dir():
        keys = sorted(p.stem.upper() for p in cfg_dir.glob("*.json"))
        profiles = [k for k in keys if k in PROFILES]
        if len(profiles) == 1:
            return profiles[0]
    if (directory / "public" / "train.jsonl").is_file() and (directory / "baselines" / "configs").is_dir():
        return None                                          # 是包但认不出 profile：交出去当未知
    return None


def _scan(roots):
    """扫一批根，返回 (认出 profile 的包, 像包但没标 profile 的候选)。深度 ≤2，不碰隐藏目录。"""
    found, unknown = {}, []
    for root in roots:
        queue = [(root, 0)]
        while queue:
            current, depth = queue.pop(0)
            if depth > 2:
                continue
            try:
                children = sorted(p for p in current.iterdir() if p.is_dir() and not p.name.startswith("."))
            except OSError:
                children = []
            for child in children:
                if _is_our_output(child):
                    continue                                 # 我们自己的产物/输出目录不参与识别
                profile = _profile_of_dir(child)
                if profile and profile not in found:
                    found[profile] = child
                elif (child / "public" / "train.jsonl").is_file():
                    unknown.append(child)
                if depth < 1:
                    queue.append((child, depth + 1))
    return found, [u for i, u in enumerate(unknown) if u not in unknown[:i]]


def discover_packages(roots=None, refresh=False) -> dict:
    """扫描候选数据根，找出 A/B/C 三个官方包（结果带缓存）。"""
    if getattr(discover_packages, "_cache", None) is not None and not refresh:
        return discover_packages._cache
    found, unknown = _scan(data_roots(roots))
    discover_packages._cache = found
    # 「像包但认不出 profile」的目录一起留着：平台的 current_round_projection 就是这种
    # ——没有 baseline-info.json、目录名不含 cpu-x、这一轮还可能不带 checkpoint.npz
    # （父亲走 ZHIXUN_CKPT 给），三条识别路全断。丢掉它们就等于把可用投影判成"没有包"。
    discover_packages._unknown = unknown
    return found


def discover_scoped(profile: str):
    """**只在显式给的数据根里**扫一遍，返回 (包或 None, 未识别候选)。

    为什么必须有：全盘扫描会把工作区约定名包（cpu-b-rul-20261110-v1）一起认进来，
    于是本地拿投影目录做自测时，`found[profile]` 命中的是那个过期本地包——测试看着绿，
    测的却是错的包（本机实测：声明 complete 从 False 变 True）。显式根优先这条规则
    要靠"先只看显式根"才成立，而不是靠运气排在前面。
    """
    roots = _env_paths(DATA_ENVS)
    if not roots:
        return None, []
    found, unknown = _scan(roots)
    return found.get(profile), unknown


def package(profile: str) -> Path:
    """官方数据包目录，以及**凭什么选它**（`package_reason(profile)`）。

    优先级按「谁说了算」排：显式给了 ZHIXUN_DATA/DATA_DIR 或开了严格模式时，以显式根里
    发现的包为准；显式根里只认出一个"像包但没标 profile"的目录时（投影包就长这样），
    把它当本轮的包，但把判定理由记下来、在诊断里印出来——不静默。

    这里修过一个会静默骗过本地自测的缺陷：以前 strict 模式下只要工作区里存在约定名包
    （`cpu-b-rul-20261110-v1`），就会绕过显式根直接回退到它，于是"投影包测试"实际测的是
    本地完整包（本机实测：complete 从 False 变成 True）。严格模式必须只认显式根，
    认不出就报错，而不是拿手边最像的东西顶上。
    """
    if profile not in PACKAGES:
        raise KeyError(f"unknown profile {profile!r}")
    default = PACKAGES[profile]
    strict = (os.environ.get("ZHIXUN_DATA_STRICT", "") or "").lower() in ("1", "true", "yes", "on")
    env_roots = _env_paths(DATA_ENVS)
    if env_roots:
        scoped, unknown = discover_scoped(profile)
        if scoped is not None:
            _PACKAGE_REASON[profile] = "discovered-in-explicit-root"
            return scoped
        # 单一候选兜底：只有当**调用方明确说了这轮是哪个 profile**（ZHIXUN_PROFILE）时才用。
        # 早先想用「包里有 baselines/configs/<P>.json」当证据，是错的：官方每个包都同时带
        # A.json/B.json/C.json 三份 config（真实包实测），那条判据会把 A 的包认成 C。
        # 投影包不带身份文件时，唯一诚实的信息来源是调用方的意图，不是目录内容的巧合。
        hint = (os.environ.get("ZHIXUN_PROFILE", "") or "").strip().upper()
        if hint == profile and len(unknown) == 1:
            _PACKAGE_REASON[profile] = "single-unknown-plus-zhixun-profile"
            return unknown[0]
        if strict:
            # 显式给了根、又要求严格：根里没有就是没有。这时回退到约定名或全盘扫描，
            # 会让"我测的是投影包"这句话变成假的（本机就差点被这样骗过去）。
            raise FileNotFoundError(
                f"ZHIXUN_DATA_STRICT=1 只认显式数据根，但根里没认出 profile {profile} 的包。"
                f"显式根里认出：{scoped if scoped is None else str(scoped)}；"
                f"像包但没标 profile 的候选："
                f"{', '.join(str(u) for u in unknown) or '(无)'}。"
                "投影包若不带 baseline-info.json/checkpoint.npz，请设 ZHIXUN_PROFILE="
                f"{profile}（且候选目录只有一个）；搜过的根："
                + ", ".join(str(r) for r in data_roots()))
    found = discover_packages()
    if env_roots:
        # 走到这里 = 显式根里没认出这个 profile，但下面两种回退都可能拿到**根外**的包
        # （约定名目录、或全盘扫描扫到工作区里的完整包）。这在平台侧不会发生
        # （容器里只有 /data），却是本地自测最容易中招的坑：投影测试读到了真包还自以为是的。
        # 所以回退允许，但必须出声，并把判定理由一起报出来。
        print("[baseline] 注意：显式数据根 {} 里没认出 profile {} 的包，将回退到根外的包"
              "（本地自测可能测的不是投影包）。只认显式根请设 ZHIXUN_DATA_STRICT=1；"
              "裸投影包请同时设 ZHIXUN_PROFILE={}。".format(
                  [str(r) for r in env_roots], profile, profile), file=sys.stderr, flush=True)
    if (default / "public" / "train.jsonl").is_file():
        _PACKAGE_REASON[profile] = "default-convention"
        return default
    discovered = discover_packages().get(profile)
    if discovered is not None:
        _PACKAGE_REASON[profile] = "discovered"
        return discovered
    raise FileNotFoundError(
        f"profile {profile} 的官方数据包没找到。搜过的数据根：" + ", ".join(str(r) for r in data_roots())
        + "；认出的包：" + (", ".join(f"{k}={v}" for k, v in sorted(discover_packages().items())) or "(无)")
        + "。请设 ZHIXUN_DATA=<含 cpu-*- 包的父目录> 或 ZHIXUN_WORKSPACE=<该父目录>；"
          "若只想用显式给的根，可设 ZHIXUN_DATA_STRICT=1。")


def package_reason(profile: str) -> str:
    """上一次 package() 凭什么选定这个包（诊断用：discovered/single-unknown-package-dir/…）。"""
    return _PACKAGE_REASON.get(profile, "unresolved")


def artifact_dir(default=None) -> Path:
    """产物（checkpoint.npz）写回的位置。

    顺序：环境变量（ZHIXUN_ARTIFACT_DIR / OUTPUT_DIR / …）> 容器里常见的固定约定
    （/output、/runs）> 调用方给的默认值。多认一个候选不花什么，但漏掉一个就是
    "平台找不到我们的 checkpoint"——那种失败在远程只表现为一句「评测运行失败」。
    """
    for path in _env_paths(ARTIFACT_ENVS):
        if path.is_dir() or str(path).startswith(("/", "C:", "D:")):
            path.mkdir(parents=True, exist_ok=True)
            return path
    # 只在 POSIX（容器）里认这两个隐式约定：Windows 上 `/output` 会被解析成当前盘根，
    # 本机开发时会把产物莫名其妙写到 D:\output（default_out 里已经踩过同一个坑）
    for candidate in ((Path("/output"), Path("/runs")) if os.name == "posix" else ()):
        try:                                    # 只认"自己已存在"或"父目录可建"，不去创建半截路径
            if candidate.is_dir():
                return candidate
            if candidate.parent.is_dir() and os.access(candidate.parent, os.W_OK):
                candidate.mkdir()
                return candidate
        except OSError:
            continue
    return Path(default) if default else None


OUR_OUTPUT_PARTS = ("runs", "round", "best", "submission", "docker-runs", "swanlog",
                    "_scratch", "_artifacts", "_swanlog", "_idtest")


def _is_our_output(path: Path) -> bool:
    """排除我们自己的产物：否则扫描会把 runs/*/best/checkpoint.npz 当成「平台父亲」，
    续训起点就会被我们自己的历史输出污染（红线 C：不能自报来路不明的检查点）。"""
    parts = {part.lower() for part in path.parts}
    if parts & set(OUR_OUTPUT_PARTS):
        return True
    return any(part.lower().startswith(("round-", ".baseline-", "b_seed", "a_10r", "c_10r"))
               for part in parts)


def discover_parent_checkpoints(roots=None) -> list:
    """收集数据根/产物目录里所有 checkpoint.npz，附上 profile 与是否等于包内初始基线。"""
    hits, seen = [], set()
    bases = {}
    package_dirs = dict(discover_packages())
    for profile, default in PACKAGES.items():
        package_dirs.setdefault(profile, default)
    for profile, directory in package_dirs.items():
        own = directory / ARTIFACT_NAME          # 「包内初始基线」的哈希集合：默认名与扫描到的都算
        if own.is_file():
            bases[sha256_file(own)] = profile
            seen.add(own.resolve())
    candidates = list(_env_paths(CKPT_ENVS))
    if (art := artifact_dir()) is not None:
        candidates += sorted(art.rglob(ARTIFACT_NAME))
    for root in data_roots(roots):
        for depth_glob in (ARTIFACT_NAME, f"*/{ARTIFACT_NAME}", f"*/*/checkpoint.npz"):
            try:
                candidates += sorted(root.glob(depth_glob))
            except OSError:
                continue
    for path in candidates:
        try:
            resolved = path.resolve()
            if resolved in seen or not path.is_file() or _is_our_output(resolved):
                continue
            seen.add(resolved)
            meta = checkpoint_meta(path) or {}
            digest = sha256_file(path)
        except Exception:
            continue
        if meta.get("format") != CHECKPOINT_FORMAT:
            continue
        hits.append({"path": path, "profile": str(meta.get("profile", "")).upper(),
                     "sha256": digest, "meta": meta, "is_package_baseline": digest in bases})
    return hits


def parent_checkpoint(profile: str, explicit=None) -> tuple:
    """本轮续训的起点：CLI > 环境变量 > 平台产物目录/数据根扫描 > 包内初始检查点。

    返回 (路径, 来源标签)。来源标签会写进 journal，评审能看到我们真的用了平台父亲，
    而不是「附带一个未使用的父检查点」（公告明令禁止的那种）。
    """
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"--parent-ckpt 指向的文件不存在：{path}")
        return path, "cli"
    for key in CKPT_ENVS:
        raw = os.environ.get(key)
        for path in _env_paths((key,)) if raw else []:
            if path.is_dir():
                path = path / ARTIFACT_NAME
            if path.is_file():
                meta = checkpoint_meta(path) or {}
                if str(meta.get("profile", "")).upper() in (profile, ""):
                    return path, f"env:{key}"
    own = package(profile) / ARTIFACT_NAME
    external = [hit for hit in discover_parent_checkpoints()
                if hit["profile"] == profile and not hit["is_package_baseline"]]
    if external:
        best = max(external, key=lambda h: h["path"].stat().st_mtime)
        return best["path"], "scan"
    return own, "package"


def source_dir(profile: str) -> Path:
    """Vendored copy (with recipe knobs) when present, else the official package."""
    vendor = VENDOR / profile
    return vendor if (vendor / "baselines").exists() else package(profile)


def _api(profile: str):
    """Import this profile's `baselines.*` modules (one profile per process)."""
    root = source_dir(profile)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from baselines import data as data_mod
    from baselines import model as model_mod
    from baselines import train as train_mod

    return train_mod, data_mod, model_mod


def config(profile: str) -> dict:
    path = source_dir(profile) / "baselines" / "configs" / f"{profile}.json"
    return json.loads(path.read_text(encoding="utf-8"))


TRAIN_ENVS = ("ZHIXUN_TRAIN_FILE", "TRAIN_FILE", "TRAIN_PATH", "TRAIN_DATA")
VAL_ENVS = ("ZHIXUN_VALIDATION_FILE", "VALIDATION_FILE", "VAL_FILE", "EVAL_FILE", "VALID_DATA")


def _leaf_count(node) -> int:
    """数 inputs 里的标量叶子个数（用来判断扁平数据是不是这个 profile 的）。"""
    if isinstance(node, dict):
        return sum(_leaf_count(value) for value in node.values())
    if isinstance(node, (list, tuple)):
        return len(node)
    return 1


def _row_feature_count(path) -> int:
    try:
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    return _leaf_count(json.loads(line).get("inputs"))
    except (OSError, ValueError):
        return -1
    return -1


def _row_answer_kind(path) -> str:
    """首行 answer 的类型：'str'（分类标签）/ 'num'（回归目标）/ ''（读不出）。"""
    try:
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    answer = json.loads(line).get("answer")
                    if isinstance(answer, str):
                        return "str"
                    if isinstance(answer, (int, float)):
                        return "num"
    except (OSError, ValueError):
        return ""
    return ""


def flat_data_pairs(roots=None) -> list:
    """找「train.jsonl 与 validation.jsonl 同目录直放」的布局（平台可能不套包目录）。"""
    pairs = []
    for root in data_roots(roots):
        queue = [(root, 0)]
        while queue:
            current, depth = queue.pop(0)
            if depth > 2:
                continue
            train, val = current / "train.jsonl", current / "validation.jsonl"
            if train.is_file() and val.is_file() and not _is_our_output(current):
                pairs.append({"dir": current, "train": train, "validation": val})
            try:
                children = sorted(p for p in current.iterdir()
                                  if p.is_dir() and not p.name.startswith("."))
            except OSError:
                children = []
            if depth < 2:
                queue += [(child, depth + 1) for child in children]
    return pairs


def data_files(profile: str) -> tuple:
    """本轮训练/评测用的两个 JSONL：环境变量 > 包内 public/ > 扁平同名对（按特征数匹配）。

    全失败时把「搜过哪些根、期望多少特征、看到哪些扁平对各有多少特征」打进异常——
    封闭评测里这段文字是唯一能远程看清平台布局的东西，不能只说"文件不存在"。
    """
    env_train = next((path for path in _env_paths(TRAIN_ENVS) if path.is_file()), None)
    env_val = next((path for path in _env_paths(VAL_ENVS) if path.is_file()), None)
    if env_train and env_val:
        return env_train, env_val
    for candidate in (PACKAGES[profile], discover_packages().get(profile)):
        if candidate is None:
            continue
        train, val = candidate / "public" / "train.jsonl", candidate / "public" / "validation.jsonl"
        if train.is_file() and val.is_file():
            return train, val
    want = -1
    try:
        ckpt, _ = parent_checkpoint(profile)
        meta = checkpoint_meta(ckpt) or {}
        want = len(meta["paths"]) if meta.get("paths") else -1
    except Exception:                                       # noqa: BLE001 找不到父亲时退回弱匹配
        want = -1
    pairs = flat_data_pairs()
    want_kind = "str" if config(profile).get("task") == "classification" else "num"
    scored = []
    for item in pairs:
        count = _row_feature_count(item["train"])
        kind = _row_answer_kind(item["train"])
        if kind and kind != want_kind:
            continue                                       # 答案类型都不对，绝不绑给这个 profile
        if want > 0:
            if count != want:
                continue
            scored.append((0, item))                       # 特征数精确匹配：最可信
        elif kind == want_kind:
            scored.append((1, item))                       # 只能靠类型匹配：次优，但会打警告
    if scored:
        scored.sort(key=lambda pair: pair[0])
        chosen = scored[0][1]
        if scored[0][0] == 1:
            print(f"[baseline] 警告：profile {profile} 的扁平数据按答案类型匹配到 "
                  f"{chosen['dir']}（无法校验特征数，因为找不到任何检查点）", file=sys.stderr)
        return chosen["train"], chosen["validation"]
    seen = "; ".join("{}(特征数 {})".format(item["dir"], _row_feature_count(item["train"]))
                     for item in pairs[:6]) or "无"
    raise FileNotFoundError(
        f"profile {profile} 找不到训练/验证数据。搜过的数据根："
        + ", ".join(str(root) for root in data_roots())
        + f"；期望特征数={want if want > 0 else '(包内无初始检查点，未知)'}；"
        + f"看到的扁平数据对：{seen}。可用 ZHIXUN_TRAIN_FILE / ZHIXUN_VALIDATION_FILE 显式指定。")


def dataset(profile: str, which: str):
    """Load public train/validation through the official reader (adds ids/x/y)."""
    cfg = config(profile)
    _, data_mod, _ = _api(profile)
    train, val = data_files(profile)
    path = {"train": train, "validation": val}[which]
    return data_mod.read_dataset(path, cfg["task"], cfg.get("classes") or [])


def load_model(profile: str, checkpoint):
    _, _, model_mod = _api(profile)
    return model_mod.Model.load(Path(checkpoint))


def metrics_model(profile: str, model, ds: dict) -> dict:
    """用已加载的模型算指标（bootstrap 要在同一模型上重复几百次，不能再走 load）。"""
    train_mod, _, _ = _api(profile)
    return train_mod.metrics(model, {"x": ds["x"], "y": ds["y"]})


def metrics(profile: str, checkpoint, ds: dict) -> dict:
    """Evaluate a checkpoint on any {x, y} slice using the official metric code."""
    return metrics_model(profile, load_model(profile, checkpoint), ds)


@contextlib.contextmanager
def recipe(wd: float = 0.0, ls: float = 0.0, jitter: float = 0.0, clip: float = 5.0,
           loss=None, center=None, sev=None, sev_end=None, swa=None):
    """Temporarily set the vendored recipe knobs for one training round.

    The first four knobs are the official ones.  The extra five are ours and are
    only written when a caller asks for them, so an official round keeps the exact
    environment it had before this file learned about them.
    """
    values = {"ZHIXUN_WD": wd, "ZHIXUN_LS": ls, "ZHIXUN_JITTER": jitter, "ZHIXUN_CLIP": clip}
    extended = {"ZHIXUN_LOSS": loss, "ZHIXUN_CENTER": center, "ZHIXUN_SEV": sev,
                "ZHIXUN_SEV_END": sev_end, "ZHIXUN_SWA": swa}
    previous = {key: os.environ.get(key) for key in RECIPE_ENV}
    os.environ.update({key: str(value) for key, value in values.items()})
    for key, value in extended.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(int(value)) if isinstance(value, bool) else str(value)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def resampled_train(profile: str, out_dir, scheme: str, seed: int = 0, persist: bool = False):
    """Write a resampled copy of the public train split (regression profiles).

    The weights lean towards late-life (small RUL) snapshots, because plain L2
    fixes the long-horizon stages while quietly degrading the last one.  Rows are
    resampled with replacement and their IDs get a suffix, so the official reader
    still sees unique IDs.

    A copy costs ~10 MB per round and the delivery contract caps total artifacts at
    64 MiB, so by default the file lands in a scratch dir outside the run tree and
    we keep only a digest (scheme + seed + row count + sha256) that reproduces it
    exactly.  `persist=True` puts it back next to the round dir, for debugging.
    """
    import numpy as np

    src, val_file = data_files(profile)
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    # 官方 train.py:65 的闸门是「训练/验证 ID 有交集就报错」。下面给重采样行的 ID 加了
    # #wNNNNN 后缀，那道闸门从此永远不会命中——盲区是**我们自己**造出来的，不是官方漏了。
    # 所以改名之前先按官方判据在原始 ID 上查一遍：查到交集就直接失败，
    # 而不是让后缀把它变成不可见。查过这件事本身也进摘要，评审看得见。
    original_ids = {row.get("id") for row in rows}
    validation_ids = set()
    for line in Path(val_file).read_text(encoding="utf-8").splitlines():
        if line.strip():
            validation_ids.add(json.loads(line).get("id"))
    overlap = sorted(original_ids & validation_ids)
    if overlap:
        raise ValueError(
            "profile {} 的训练/验证 ID 有 {} 个交集（例如 {}）：官方闸门本应报错，"
            "重采样的后缀会把它藏起来，所以这里直接失败，不改数据。".format(
                profile, len(overlap), ", ".join(str(x) for x in overlap[:3])))
    answers = [row["answer"] for row in rows]
    if scheme in ("sqrt_inv", "inv"):
        # 数值目标（回归）才做末端加权；分类标签是字符串，float() 会当场炸
        y = np.array([float(value) for value in answers], dtype=float)
        w = 1.0 / np.sqrt(y + 50.0) if scheme == "sqrt_inv" else 1.0 / (y + 50.0)
    elif scheme == "class_balanced":
        labels = np.asarray([str(value) for value in answers])
        counts = {value: max(1, int(np.sum(labels == value))) for value in np.unique(labels)}
        w = np.array([1.0 / counts[str(value)] for value in labels], dtype=float)
    else:
        raise KeyError(f"unknown weight scheme {scheme!r}")
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(rows), size=len(rows), replace=True, p=w / w.sum())
    if persist:
        dst = Path(out_dir).parent / f"{Path(out_dir).name}.train.jsonl"
    else:
        scratch = Path(os.environ.get("ZHIXUN_SCRATCH") or (Path(out_dir).parent / "_scratch"))
        scratch.mkdir(parents=True, exist_ok=True)
        dst = scratch / f"{Path(out_dir).name}.train.jsonl"
    with dst.open("w", encoding="utf-8") as fh:
        for k, i in enumerate(idx):
            row = dict(rows[int(i)])
            row["id"] = f"{row['id']}#w{k:05d}"
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = sha256_file(dst)
    counts = {}
    for i in idx:
        key = str(rows[int(i)]["answer"])
        counts[key] = counts.get(key, 0) + 1
    info = {"scheme": str(scheme), "seed": int(seed), "rows": int(len(idx)),
            "id_overlap_checked": True, "id_overlap": 0,          # 官方闸门的等价检查，改名之前做的
            "source_sha256": sha256_file(src), "sha256": digest, "label_counts": counts,
            "path": str(dst), "persisted": bool(persist)}
    return dst, info


def continue_round(profile: str, out_dir, resume, *, epochs: int, learning_rate=None,
                   batch_size=None, stage_id: str, label_provenance=None,
                   wd: float = 0.0, ls: float = 0.0, jitter: float = 0.0,
                   clip: float = 5.0, weights=None, seed=None, persist_resample=False,
                   loss=None, center=None, sev=None, sev_end=None, swa=None) -> dict:
    """Run one official continued-training round; returns the training report.

    The report is the official `training.json` payload, untouched, plus two
    harness-only keys: `parent_checkpoint_source` (cli/env/scan/package — proof we
    really continued from the platform checkpoint) and `resample` (digest of the
    weighted train file when a sampling scheme was used).
    """
    train_mod, _, _ = _api(profile)
    train_file, validation_file = data_files(profile)
    resample = None
    if weights:
        train_file, resample = resampled_train(profile, out_dir, str(weights), int(seed or 0),
                                               persist=persist_resample)
    prov = label_provenance
    if prov is not None and not Path(prov).exists():
        raise FileNotFoundError(f"profile {profile} needs a label provenance file: {prov}")
    try:
        with recipe(wd=wd, ls=ls, jitter=jitter, clip=clip, loss=loss, center=center,
                    sev=sev, sev_end=sev_end, swa=swa):
            report = train_mod.train_run(
                profile,
                train_file,
                validation_file,
                Path(out_dir),
                resume=Path(resume),
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                stage_id=stage_id,
                label_provenance=prov,
            )
    finally:
        if resample is not None and not resample["persisted"]:
            try:                                    # 只留摘要：交付体积是硬约束
                Path(resample["path"]).unlink()
            except OSError:
                pass
    report["resample"] = resample
    return report


def predict_api(profile: str):
    """This profile's official prediction module (schema check stays official)."""
    root = source_dir(profile)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from baselines import predict as predict_mod

    return predict_mod


def baseline_checkpoint(profile: str) -> Path:
    return package(profile) / "checkpoint.npz"