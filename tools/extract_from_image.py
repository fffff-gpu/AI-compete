"""Extract one file from the exported image tar (OCI/docker-save layout).

The archive is a tar-of-tars: outer index.json -> inner index blob -> manifest
blob -> layer blobs. Layers are gzipped tars with absolute /app paths.
"""
import gzip
import io
import json
import sys
import tarfile


def _json_from(tf, name):
    member = tf.getmember(name)
    return json.loads(tf.extractfile(member).read().decode("utf-8"))


def layers(tar_path):
    with tarfile.open(tar_path) as tf:
        outer = _json_from(tf, "index.json")
        manifests = outer["manifests"]
        # docker-save nests a second index: manifests[0] points at a blob index
        first = manifests[0]
        if first.get("mediaType", "").endswith("image.index.v1+json"):
            inner = json.loads(tf.extractfile(tf.getmember("blobs/sha256/" + first["digest"].split(":")[1])).read())
            manifest_digest = inner["manifests"][0]["digest"]
        else:
            manifest_digest = first["digest"]
        manifest = json.loads(tf.extractfile(tf.getmember("blobs/sha256/" + manifest_digest.split(":")[1])).read())
        return [layer["digest"].split(":")[1] for layer in manifest["layers"]]


def extract(tar_path, wanted):
    """wanted: list of absolute paths like /app/contest_submission.py"""
    found = {}
    with tarfile.open(tar_path) as tf:
        for digest in layers(tar_path):
            blob = tf.extractfile(tf.getmember("blobs/sha256/" + digest)).read()
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as layer:
                for member in layer.getmembers():
                    if member.name in wanted and member.isfile():
                        found[member.name] = layer.extractfile(member).read()
    return found


if __name__ == "__main__":
    tar_path = sys.argv[1]
    wanted = sys.argv[1:]
    tar_path = sys.argv[1]
    wanted = sys.argv[2:]
    out = extract(tar_path, wanted)
    for name, data in out.items():
        dest = "_image_dump" + name
        import os
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        open(dest, "wb").write(data)
        print("extracted", name, len(data), "->", dest)
    for name in wanted:
        if name not in out:
            print("MISSING", name)