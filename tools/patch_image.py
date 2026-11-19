"""Rewrite /app code inside the exported image tar without a Docker daemon.

`docker build` is unavailable here, so the image is repaired the way a rebuild
would: the layers that carry the changed files are re-packed in place, the
config diff_ids, manifest, index and docker-save manifest.json are re-chained,
and the archive is written back deterministically.

Usage:
    python _patch_image.py <src.tar> <dst.tar>
"""
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

SDK = Path(__file__).resolve().parent / "sdk"
FILES = {
    "app/contest_submission.py": SDK / "contest_submission.py",
    "app/rulkit/model.py": SDK / "rulkit" / "model.py",
}
LABEL_VERSION = "1.5.1"


def blob_path(digest):
    return "blobs/sha256/" + digest.split(":", 1)[1]


def digest_of(payload):
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def canonical(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def repack_layer(raw, replacements):
    """Return (plain_tar_bytes, gz_bytes) with selected members replaced."""
    source = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
    out = io.BytesIO()
    replaced = set()
    with tarfile.open(fileobj=out, mode="w") as target:
        for member in source.getmembers():
            payload = None
            if member.isfile():
                payload = source.extractfile(member).read()
                if member.name in replacements:
                    payload = replacements[member.name].read_bytes()
                    replaced.add(member.name)
            info = tarfile.TarInfo(member.name)
            info.mode, info.uid, info.gid = member.mode, member.uid, member.gid
            info.mtime, info.type = member.mtime, member.type
            info.uname, info.gname = member.uname, member.gname
            if payload is not None:
                info.size = len(payload)
                target.addfile(info, io.BytesIO(payload))
            else:
                target.addfile(info)
    plain = out.getvalue()
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0, compresslevel=6) as stream:
        stream.write(plain)
    return plain, compressed.getvalue()


def main(src, dst):
    with tarfile.open(src) as tar:
        members, blobs = [], {}
        for member in tar.getmembers():
            members.append(member)
            if member.isfile():
                blobs[member.name] = tar.extractfile(member).read()

    index = json.loads(blobs["index.json"])
    # index.json -> inner OCI index -> image manifest
    inner_entry = index["manifests"][0]
    inner = json.loads(blobs[blob_path(inner_entry["digest"])])
    manifest_entry = inner["manifests"][0]
    manifest = json.loads(blobs[blob_path(manifest_entry["digest"])])
    config_entry = manifest["config"]
    config = json.loads(blobs[blob_path(config_entry["digest"])])

    diff_ids = list(config["rootfs"]["diff_ids"])
    patched = []
    mapping = {}
    for position, layer in enumerate(manifest["layers"]):
        path = blob_path(layer["digest"])
        raw = blobs[path]
        probe = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
        names = {m.name for m in probe.getmembers() if m.isfile()}
        todo = {name: FILES[name] for name in FILES if name in names}
        if not todo:
            continue
        plain, compressed = repack_layer(raw, todo)
        new_digest = digest_of(compressed)
        blobs[blob_path(new_digest)] = compressed
        mapping[path] = blob_path(new_digest)
        layer["digest"] = new_digest
        layer["size"] = len(compressed)
        diff_ids[position] = digest_of(plain)
        patched.append((position, sorted(todo)))

    if not patched:
        raise SystemExit("no layer carried the patched files")

    config["rootfs"]["diff_ids"] = diff_ids
    labels = config["config"].get("Labels") or {}
    if "org.opencontainers.image.version" in labels:
        labels["org.opencontainers.image.version"] = LABEL_VERSION
    if "org.opencontainers.image.description" in labels:
        labels["org.opencontainers.image.description"] = (
            "工业模型自进化 Harness：workspace v2 SDK 契约 + 受控续训配方")
    config["config"]["Labels"] = labels

    config_payload = canonical(config)
    config_digest = digest_of(config_payload)
    blobs[blob_path(config_digest)] = config_payload
    mapping[blob_path(config_entry["digest"])] = blob_path(config_digest)
    manifest["config"] = {"mediaType": config_entry["mediaType"],
                          "digest": config_digest, "size": len(config_payload)}

    manifest_payload = canonical(manifest)
    manifest_digest = digest_of(manifest_payload)
    blobs[blob_path(manifest_digest)] = manifest_payload
    manifest_entry.update({"digest": manifest_digest, "size": len(manifest_payload)})

    inner_payload = canonical(inner)
    inner_digest = digest_of(inner_payload)
    blobs[blob_path(inner_digest)] = inner_payload
    inner_entry.update({"digest": inner_digest, "size": len(inner_payload)})

    blobs["index.json"] = canonical(index)

    # docker-save manifest.json mirrors Config/Layers digests
    if "manifest.json" in blobs:
        docker_manifest = json.loads(blobs["manifest.json"])
        for item in docker_manifest:
            item["Config"] = blob_path(config_digest)
            item["Layers"] = [mapping.get(name, name) for name in item["Layers"]]
        blobs["manifest.json"] = canonical(docker_manifest)

    # drop blobs nothing references any more
    referenced = set()
    for entry in index["manifests"]:
        referenced.add(blob_path(entry["digest"]))
    for entry in inner["manifests"]:
        referenced.add(blob_path(entry["digest"]))
    referenced.add(blob_path(manifest["config"]["digest"]))
    referenced.update(blob_path(layer["digest"]) for layer in manifest["layers"])
    if "manifest.json" in blobs:
        docker_manifest = json.loads(blobs["manifest.json"])
        for item in docker_manifest:
            referenced.add(item["Config"])
            referenced.update(item["Layers"])
    for name in list(blobs):
        if name.startswith("blobs/sha256/") and name not in referenced:
            del blobs[name]

    with tarfile.open(dst, "w") as out:
        written = set()
        for member in members:
            if member.name not in blobs:
                continue
            info = tarfile.TarInfo(member.name)
            info.mode, info.uid, info.gid = member.mode, member.uid, member.gid
            info.mtime, info.type = member.mtime, member.type
            payload = blobs[member.name]
            info.size = len(payload)
            out.addfile(info, io.BytesIO(payload))
            written.add(member.name)
        # blobs created during the rewrite (new config/layers/index) are not in
        # the original member list, so append them in insertion order.
        for name, payload in blobs.items():
            if name in written:
                continue
            info = tarfile.TarInfo(name)
            info.mode, info.uid, info.gid = 0o644, 0, 0
            info.mtime, info.type = 0, tarfile.REGTYPE
            info.size = len(payload)
            out.addfile(info, io.BytesIO(payload))

    print("patched layers:", patched)
    print("config:", config_digest)
    print("manifest:", manifest_digest)
    print("blobs kept:", sum(1 for n in blobs if n.startswith("blobs/")))
    print("written:", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])