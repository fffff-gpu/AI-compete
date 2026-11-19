"""诊断 docker-save 归档：解释平台为什么报「镜像处理未完成，请重试」。

    python _verify_tar.py submission/01-镜像/zhixun-harness-1.0.tar

公告 11-13 §一 只承认一种上传文件：`docker save -o harness.tar 镜像名:标签` 的产物；
源码 zip、目录压缩包、单独 Dockerfile、`docker export` 的容器文件系统包都不算。
本脚本用标准库读归档结构，把「像不像 docker save 产物」变成可核对的判据，
而不是反复重传同一个无效文件。退出码 0=结构合格，1=有问题（附具体修法）。

实测过的两类「结构合格但仍处理失败」：
1. **OCI 布局**：Docker Desktop 开着 containerd 镜像存储时，docker save 出 OCI 格式
   （index.json + oci-layout + blobs/sha256/…），只认 docker v1/v2 schema 的 importer 读不了；
2. **attestation 条目**：BuildKit 默认往 index 里塞一条 platform=unknown/unknown、
   config=application/vnd.oci.empty.v1+json、层为 application/vnd.in-toto+json 的
   provenance/SBOM 证明。真镜像在里面是对的，但不少 importer 会挑到这条空条目或直接
   拒绝多条目 index。关掉：`docker build --provenance=false --sbom=false`。
"""
import json
import sys
import tarfile
from pathlib import Path

TOP_DIRS_OF_EXPORT = ("bin", "usr", "etc", "lib", "lib64", "opt", "var", "root", "sbin")
EXPECTED_ENTRYPOINT = "harness.entrypoint"


def human(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return "{:.2f} {}".format(size, unit)
        size /= 1024
    return "{:.2f} GiB".format(size)


def oci_platforms(tar) -> dict:
    """遍历 OCI index：列出每个条目的 os/architecture，并单独数出 attestation 空条目。"""
    names = {m.name for m in tar.getmembers()}
    out = {"entries": [], "attestations": 0, "images": [], "error": None}

    def read_json(digest):
        key = "blobs/sha256/" + digest.split(":")[1]
        if key not in names:
            raise KeyError(key)
        return json.loads(tar.extractfile(key).read())

    try:
        top = json.loads(tar.extractfile("index.json").read())
    except (KeyError, OSError, ValueError) as exc:
        out["error"] = "index.json 不可读：{}".format(exc)
        return out

    def walk(digest, depth=0):
        try:
            obj = read_json(digest)
        except KeyError as exc:
            out["error"] = "缺失 blob：{}".format(exc)
            return
        media = str(obj.get("mediaType", ""))
        if media.endswith("index.v1+json"):
            for item in obj.get("manifests", []):
                platform = item.get("platform") or {}
                out["entries"].append({"depth": depth, "os": platform.get("os"),
                                       "arch": platform.get("architecture"),
                                       "digest": item["digest"]})
                if platform.get("os") == "unknown":
                    out["attestations"] += 1
                elif depth < 2:
                    walk(item["digest"], depth + 1)
        else:
            config = obj.get("config") or {}
            image = {"mediaType": media, "configMediaType": config.get("mediaType"),
                     "layers": len(obj.get("layers", []))}
            try:
                data = read_json(config["digest"])
                cfg = data.get("config") or {}
                image.update(os=data.get("os"), architecture=data.get("architecture"),
                             variant=data.get("variant"), entrypoint=cfg.get("Entrypoint"),
                             env=[e for e in cfg.get("Env", [])
                                  if e.startswith(("ZHIXUN_", "PYTHONPATH", "BASELINE_"))])
            except (KeyError, TypeError, ValueError):
                image["error"] = "config blob 读不出（可能是空的 attestation 条目）"
            out["images"].append(image)

    for item in top.get("manifests", []):
        # 顶层 index 的条目也要记录：单平台 OCI 的平台信息就写在这一层，
        # 漏记会把「干净的单平台包」误判成「多条目包」
        platform = item.get("platform") or {}
        out["entries"].append({"depth": 0, "os": platform.get("os"),
                               "arch": platform.get("architecture"), "digest": item["digest"]})
        if platform.get("os") == "unknown":
            out["attestations"] += 1
        else:
            walk(item["digest"], 1)
    return out


def classify(path: Path) -> dict:
    """机器可读的分级：archive_valid=是不是有效 docker-save 归档；format=经典还是 OCI。

    分级的意义：平台可能只认经典格式，也可能两种都吃。把"归档本身是否有效"和
    "格式是否会被平台接受"分开，提交包索引才不会把一份干净的可上传归档写成"待构建"。
    """
    result = {"archive_valid": False, "format": "unknown", "tags": [], "layers": 0,
              "attestations": None, "platform": None, "entrypoint": None,
              "size_bytes": 0, "problems": [], "notes": [], "oci": False}
    data = inspect(path, collect=result, quiet=True)
    result["exit_code"] = data
    result["archive_valid"] = not any(p.startswith(("不是可解析", "没有 manifest", "顶层是 rootfs",
                                                 "缺少", "归档里没有可用的", "归档里出现了不该有"))
                                      for p in result["problems"])
    return result


def inspect(path: Path, collect: dict | None = None, quiet: bool = False) -> int:
    def say(*args_, **kw_):
        if not quiet:
            print(*args_, **kw_)
    problems = [] if collect is None else collect["problems"]
    notes = [] if collect is None else collect.setdefault("notes", []) if collect is None else collect.setdefault("notes", [])
    size = path.stat().st_size
    if collect is not None:
        collect["size_bytes"] = size
    say("文件：{}".format(path))
    say("大小：{}（{:,} 字节）".format(human(size), size))

    try:
        tar = tarfile.open(path)
    except tarfile.ReadError as exc:
        say("✗ 不是可解析的 tar：{}".format(exc))
        say("  修法：用 `docker save -o <file> <镜像名:标签>` 重新导出；"
              "把 zip 或目录改名为 .tar 不会变成镜像归档。")
        return 1
    with tar:
        members = tar.getmembers()
        if collect is not None:
            collect["layers"] = len([m for m in members if m.name.endswith("layer.tar")])
        names = [m.name for m in members]
        tops = {n.split("/")[0] for n in names if n}
        say("成员：{} 条，顶层条目 {} 个".format(len(members), len(tops)))

        if "manifest.json" not in names:
            if set(TOP_DIRS_OF_EXPORT) & tops:
                problems.append(
                    "顶层是 rootfs（bin/、usr/…）而不是镜像归档：这是 `docker export`"
                    "（导出容器文件系统）的产物，平台不接受。改用 "
                    "`docker save -o x.tar 镜像名:标签`。")
            else:
                problems.append("没有 manifest.json：不像 docker save 的产物。"
                                "重跑 `docker save -o harness.tar zhixun-harness:1.0`。")
        else:
            try:
                manifest = json.loads(tar.extractfile("manifest.json").read())
            except Exception as exc:                            # noqa: BLE001
                problems.append("manifest.json 读不出来：{}".format(exc))
                manifest = []
            entry = manifest[0] if manifest else {}
            tags = entry.get("RepoTags")
            layers = entry.get("Layers") or []
            config = entry.get("Config") or entry.get("ConfigFile")
            say("RepoTags：{}　层数：{}　Config：{}".format(tags, len(layers), config))
            if collect is not None:
                collect["tags"] = list(tags or [])
                collect["layers"] = len(layers)
                collect["format"] = "docker"
            if not tags or any(":" not in t or t.endswith(":<none>") for t in (tags or [])):
                problems.append("归档没有可用的 镜像名:标签（RepoTags={}）。"
                                "先 `docker tag <id> zhixun-harness:1.0` 再 save；"
                                "按 image ID 直接 save 会丢标签。".format(tags))
            if config and config not in names:
                problems.append("manifest 指向的 Config 文件 {!r} 不在归档里：归档不完整。".format(config))
            missing = [layer for layer in layers
                       if layer.rstrip("/") not in names
                       and not any(n.startswith(layer.rstrip("/")) for n in names)]
            if missing:
                problems.append("缺少 {} 个层的目录，例如 {}：归档不完整。".format(len(missing), missing[:2]))

        if {"oci-layout", "index.json"} & tops:
            say("\n检出 OCI 布局（{}）".format(sorted({"oci-layout", "index.json"} & tops)))
            if "index.json" in names:
                oci = oci_platforms(tar)
                if collect is not None:
                    collect["oci"] = True
                    collect["format"] = "oci"
                    collect["attestations"] = oci["attestations"]
                say("index 条目：{}（其中 attestation 空条目 {} 条）".format(
                    len(oci["entries"]), oci["attestations"]))
                for image in oci["images"]:
                    say("  真实镜像：{}/{}{}　层={}　config={}".format(
                        image.get("os"), image.get("architecture"),
                        "/" + str(image.get("variant")) if image.get("variant") else "",
                        image.get("layers"), image.get("configMediaType")))
                    if collect is not None:
                        collect["platform"] = "{}/{}".format(image.get("os"),
                                                             image.get("architecture"))
                        collect["entrypoint"] = image.get("entrypoint")
                    say("    ENTRYPOINT={}".format(image.get("entrypoint")))
                    say("    ENV={}".format(image.get("env")))
                if oci["error"]:
                    problems.append("OCI index 遍历失败：{}".format(oci["error"]))
                if oci["attestations"]:
                    problems.append(
                        "index 里有 {} 条 **attestation 证明条目**（platform=unknown/unknown、"
                        "config=application/vnd.oci.empty.v1+json、层为 "
                        "application/vnd.in-toto+json）。这是 BuildKit 默认附赠的 "
                        "provenance/SBOM：真镜像在里面是完好的，但不少 importer 读 index 时"
                        "会挑到这条空条目、或直接拒绝多条目 index，症状同样是「镜像处理未完成」。"
                        "关掉它：docker build --provenance=false --sbom=false。".format(oci["attestations"]))
                for image in oci["images"]:
                    os_name = image.get("os")
                    if os_name and os_name != "linux":
                        problems.append(
                            "镜像 os={}，不是 linux。评测设备是 Linux：请确认 Docker Desktop "
                            "处于 Linux 容器模式（右下角托盘 → Switch to Linux containers）。".format(os_name))
                    elif image.get("architecture") and image.get("architecture") != "amd64":
                        problems.append(
                            "镜像 architecture={}，不是 amd64。请加 --platform=linux/amd64 重新构建。"
                            .format(image.get("architecture")))
                    ep = " ".join(image.get("entrypoint") or [])
                    # 两种合法入口：SDK 1.0.4（protocol v2）规定 `python3 /app/harness.py`；
                    # 1.0/1.1 旧包用 `python -m harness.entrypoint`。其余入口视为未知包。
                    if ep and "harness.py" not in ep and EXPECTED_ENTRYPOINT not in ep:
                        problems.append(
                            "镜像入口是 [{}]，既不是 SDK v2 的 `python3 /app/harness.py`，"
                            "也不是旧包的 `python -m {}`：请确认镜像内容，不要直接上传。"
                            .format(ep.strip(), EXPECTED_ENTRYPOINT))
            entries = oci.get('entries', [])
            # 平台信息可能只写在 image config 里（顶层 index 不带 platform 字段），两处都兜住
            linux_count = (len([e for e in entries if e.get('os') == 'linux'])
                           or len([im for im in oci['images'] if im.get('os') == 'linux']))
            clean_single = (linux_count == 1 and not oci['attestations']
                            and len(oci['images']) == 1)
            # 判据跟着事实走：平台已把"单平台、无 attestation"的 OCI 判为镜像可用，
            # 再把它记成问题就是体检器和现实吵架，还会让导出脚本拒绝一个可用产物。
            (notes if clean_single else problems).append(
                '归档是 OCI 布局（index.json + oci-layout）。'
                + ('当前是单平台、无 attestation 的干净 OCI，很多实现能吃，可以先上传试一次；'
                   if clean_single else '多条目或含 attestation，先按上面的办法清掉；')
                + '若平台 importer 只认 docker v1/v2 schema，两条确定路径：'
                  '① Docker Desktop -> Settings -> General -> 取消勾选 [Use containerd for '
                  'pulling and storing images] -> Restart -> 重新 pull 基础镜像、build、save；'
                  '② 不切存储，改用 docker buildx build --output type=docker,dest=x.tar'
                  '（本机 Docker 29.9.2 实测：docker save --format docker 不存在，别用这条）。')

        dataish = [name for name in names
                   if "public/train.jsonl" in name or name.endswith(".train.jsonl")
                   or name.endswith(".env") or "/runs/" in name]
        if dataish:
            problems.append("归档里出现了不该有的东西（{} 项，例如 {}）：镜像只含代码，"
                            "数据包运行时挂载；含 .env 更是红线（凭据入镜像=取消资格）。"
                            .format(len(dataish), dataish[0]))

    say()
    for note in notes:
        say("· 提示：" + note)
    if problems:
        say("✗ 发现 {} 个问题：".format(len(problems)))
        for index, problem in enumerate(problems, 1):
            say("  {}. {}".format(index, problem))
        return 1
    say("✓ 结构上是有效的 docker save 归档。若平台仍报「镜像处理未完成」，"
          "按公告 §四 联系管理员并提供：队名、上传时间、文件名与大小、状态截图、导出命令。")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--json-out":
        # --json-out <file>：PowerShell 5.1 会按控制台代码页（本机 GBK）解码子进程输出，
        # 带中文的 JSON 经管道就被改坏、ConvertFrom-Json 抛 ArgumentException（实测踩过）。
        # 落盘再按 UTF-8 读，是唯一稳的通路。
        result = classify(Path(sys.argv[3]))
        Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False, indent=2) + chr(10),
                                     encoding="utf-8")
        print(json.dumps({"written": sys.argv[2], "archive_valid": result["archive_valid"],
                          "format": result["format"], "attestations": result["attestations"]},
                         ensure_ascii=False))
        raise SystemExit(0)
    if len(sys.argv) == 3 and sys.argv[1] == "--json":
        print(json.dumps(classify(Path(sys.argv[2])), ensure_ascii=False, indent=2))
        raise SystemExit(0)
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    target = Path(sys.argv[1])
    if not target.is_file():
        # 实测踩过：上一步 docker save 因为 unknown flag 直接失败，这里却报 FileNotFoundError
        # 堆栈，容易让人以为是体检脚本坏了。文件不存在就是不存在，说清楚。
        print("✗ 找不到归档：{}".format(target))
        print("  多半是上一步 `docker save` 没成功（例如 Docker 29.9.2 不认 --format docker，"
              "报 unknown flag 后 tar 根本没写出来）。")
        print("  重新导出：docker save -o \"{}\" zhixun-harness:1.0".format(target))
        raise SystemExit(1)
    raise SystemExit(inspect(target))
