"""Strip BuildKit attestation entries from a docker-save OCI archive.

Why: `docker build` without `--provenance=false --sbom=false` appends a
platform=unknown/unknown attestation manifest into the OCI index. Some
importers pick that empty entry or reject multi-entry indexes ("image
processing not finished"). We cannot rebuild right now (no Docker daemon),
so rewrite the archive in place: drop attestation manifests, drop blobs no
longer referenced, keep the real image, config, layers and manifest.json.

Usage: python _strip_attestation.py <src.tar> <dst.tar>
"""
import hashlib
import json
import sys
import tarfile

INDEX = "index.json"
MANIFEST = "manifest.json"
LAYOUT = "oci-layout"


def blob_path(digest):
    return "blobs/sha256/" + digest.split(":", 1)[1]


def canonical(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def is_attestation(entry):
    if entry.get("annotations", {}).get("vnd.docker.reference.type") == "attestation-manifest":
        return True
    platform = entry.get("platform") or {}
    return platform.get("os") == "unknown" or platform.get("architecture") == "unknown"


def digest_of(payload):
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def main(src, dst):
    with tarfile.open(src) as tar:
        infos = {}
        blobs = {}
        for member in tar.getmembers():
            infos[member.name] = member
            if member.isfile():
                blobs[member.name] = tar.extractfile(member).read()

    index = json.loads(blobs[INDEX])
    dropped = []

    # 1) resolve inner index(es) and drop attestation manifests inside them
    for entry in index.get("manifests", []):
        if entry.get("mediaType") != "application/vnd.oci.image.index.v1+json":
            continue
        path = blob_path(entry["digest"])
        inner = json.loads(blobs[path])
        kept = [m for m in inner.get("manifests", []) if not is_attestation(m)]
        dropped.extend(m["digest"] for m in inner.get("manifests", []) if is_attestation(m))
        if len(kept) != len(inner.get("manifests", [])):
            inner["manifests"] = kept
            payload = canonical(inner)
            new_digest = digest_of(payload)
            blobs[blob_path(new_digest)] = payload
            infos[blob_path(new_digest)] = infos[path]
            entry["digest"] = new_digest
            entry["size"] = len(payload)

    # 2) drop top-level attestation entries too
    kept = [m for m in index.get("manifests", []) if not is_attestation(m)]
    dropped.extend(m["digest"] for m in index.get("manifests", []) if is_attestation(m))
    index["manifests"] = kept
    blobs[INDEX] = canonical(index)

    # 3) keep only blobs reachable from index.json / manifest.json
    referenced = set()

    def walk(entry):
        digest = entry.get("digest")
        if not digest:
            return
        referenced.add(blob_path(digest))
        media = entry.get("mediaType", "")
        if media.endswith("image.index.v1+json"):
            for child in json.loads(blobs[blob_path(digest)]).get("manifests", []):
                walk(child)
            return
        if media.endswith("image.manifest.v1+json"):
            refs = [entry.get("config") or {}] + list(entry.get("layers") or [])
            for ref in refs:
                if ref.get("digest"):
                    referenced.add(blob_path(ref["digest"]))

    for entry in index["manifests"]:
        walk(entry)

    docker_manifest = json.loads(blobs[MANIFEST])
    for item in docker_manifest:
        referenced.add(item["Config"])
        referenced.update(item["Layers"])

    for name in list(blobs):
        if name.startswith("blobs/sha256/") and name not in referenced:
            del blobs[name]

    # 4) rewrite the archive, preserving original member order
    with tarfile.open(dst, "w") as out:
        for name in infos:
            if name not in blobs:
                continue
            info = tarfile.TarInfo(name)
            src_info = infos[name]
            info.mode = src_info.mode
            info.uid = src_info.uid
            info.gid = src_info.gid
            info.mtime = src_info.mtime
            info.type = src_info.type
            payload = blobs.get(name)
            if payload is None:
                out.addfile(info)
            else:
                info.size = len(payload)
                out.addfile(info, fileobj=__import__("io").BytesIO(payload))

    print("dropped attestation manifests:", dropped)
    print("kept blobs:", sum(1 for n in blobs if n.startswith("blobs/")))
    print("written:", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])