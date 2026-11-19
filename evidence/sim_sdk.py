"""In-process host stand-in for contest.harness.workspace.v2.

The Windows sandbox denies subprocess pipes, so this drives the real SDK `Round`
over two in-memory queues instead of spawning `harness.py`. It exercises the same
frames the platform sends: start -> history/model/uploads/candidate -> round_complete.
"""
import hashlib
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repo root
SDK = ROOT / "official" / "sdk"
sys.path.insert(0, str(SDK))


class Wire:
    """Line-oriented writer that a host thread consumes."""

    def __init__(self):
        self.lines = queue.Queue()

    def write(self, text):
        for line in text.splitlines():
            if line.strip():
                self.lines.put(line)

    def flush(self):
        pass


class Reader:
    def __init__(self):
        self.lines = queue.Queue()

    def readline(self):
        try:
            return self.lines.get(timeout=120)
        except queue.Empty:
            raise RuntimeError("host did not reply within 120s")

    def feed(self, value):
        self.lines.put(json.dumps(value, ensure_ascii=False) + "\n")


def frame_text(value):
    return json.dumps(value, ensure_ascii=False)


class Host(threading.Thread):
    def __init__(self, start, reply):
        super().__init__(daemon=True)
        self.frame = start
        self.reply = reply
        self.wire_in = Reader()
        self.wire_out = Wire()
        self.uploads = {}
        self.sequence = 0
        self.record = None
        self.failure = None

    def run(self):
        try:
            self.wire_in.feed(self.frame)
            while True:
                line = self.wire_out.lines.get()
                if line is None:
                    return
                frame = json.loads(line)
                kind = frame.get("type")
                if kind == "round_failed":
                    raise RuntimeError("submission failed: %s" % frame.get("error"))
                if kind == "round_complete":
                    self.record = frame["record"]
                    self.wire_in.feed({"type": "round_ack", "accepted": True})
                    return
                if frame.get("id") != self.sequence:
                    raise RuntimeError("rpc sequence mismatch")
                self.sequence += 1
                if kind == "history_request":
                    reply = {"type": "history_response", "id": frame["id"], "records": []}
                elif kind == "model_request":
                    response = dict(self.reply(frame["messages"]))
                    call_id = response.pop("request_id")
                    reply = {"type": "model_response", "id": frame["id"], "call_id": call_id, "response": response}
                elif kind == "upload_begin_request":
                    upload_id = str(len(self.uploads))
                    self.uploads[upload_id] = {"manifest": frame["files"],
                                               "buffers": {item["path"]: bytearray() for item in frame["files"]}}
                    reply = {"type": "upload_begin_response", "id": frame["id"], "upload_id": upload_id}
                elif kind == "upload_chunk_request":
                    buffer = self.uploads[frame["upload_id"]]["buffers"][frame["path"]]
                    assert frame["offset"] == len(buffer), "upload offset mismatch"
                    import base64
                    buffer.extend(base64.b64decode(frame["data_base64"], validate=True))
                    reply = {"type": "upload_chunk_response", "id": frame["id"], "offset": len(buffer)}
                elif kind == "upload_finish_request":
                    reply = {"type": "upload_finish_response", "id": frame["id"], "upload_id": frame["upload_id"]}
                elif kind == "candidate_request":
                    upload = self.uploads[frame["upload_id"]]
                    for item in upload["manifest"]:
                        buffer = upload["buffers"][item["path"]]
                        assert "sha256:" + hashlib.sha256(bytes(buffer)).hexdigest() == item["sha256"]
                        assert len(buffer) == item["size_bytes"]
                    reply = {"type": "candidate_response", "id": frame["id"],
                             "artifact_id": "artifact-1", "summary": {"metrics": {}}}
                elif kind == "state_request":
                    reply = {"type": "state_response", "id": frame["id"], "state_sha256": "sha256:" + "11" * 32}
                else:
                    raise RuntimeError("unexpected frame %s" % kind)
                self.wire_in.feed(reply)
        except Exception as error:  # noqa: BLE001
            self.failure = error


def start_frame(dataset_key, task, metric, direction, train_path, validation_path, parent_path):
    digest = "sha256:" + hashlib.sha256(Path(parent_path).read_bytes()).hexdigest()
    parent = {
        "artifact_id": "parent-0",
        "artifact_sha256": digest,
        "artifact": {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [],
                     "predict_argv": ["python3", "/app/harness.py", "predict"]},
        "files": [{"path": "checkpoint.npz", "sha256": digest, "size_bytes": Path(parent_path).stat().st_size}],
    }
    return {
        "protocol": "contest.harness.workspace.v2",
        "run_id": "run-sim",
        "dataset_key": dataset_key,
        "round_index": 1,
        "stage_id": "public-development",
        "prompt": "从官方基线与公开训练、验证数据出发，在授权预算内自主提出假设。",
        "prompt_version": "industrial-workspace-flexible-v3",
        "rounds_per_dataset": 10,
        "parent_checkpoint_sha256": digest,
        "task_contract": {
            "dataset_key": dataset_key, "task": task, "metric": metric, "metric_direction": direction,
            "public_projection": {"train_path": str(train_path), "validation_path": str(validation_path),
                                  "row_fields": ["id", "inputs", "answer"], "source_metadata": False},
        },
        "model_name": "base:test-model",
        "parent_artifact": parent,
        "artifacts": [parent],
        "state_sha256": "",
        "state_path": None,
        "max_output_tokens": 600,
        "resource_limits": {"artifact_upload_protocol": "chunked-v1",
                            "artifact_bytes": 512 * 1024 * 1024, "state_bytes": 512 * 1024 * 1024,
                            "candidate_upload_bytes_per_round": 1024 * 1024 * 1024,
                            "state_upload_bytes_per_round": 512 * 1024 * 1024},
        "wall_seconds": 600,
    }


def run_train(dataset_key, task, metric, direction, parent_ckpt, reply_text, epochs_hint=None,
              validation_override=None):
    tmp = ROOT / "official" / "_scratch" / ("sim-" + str(os.getpid()))
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    previous = tmp / "previous"
    previous.mkdir()
    shutil.copyfile(parent_ckpt, previous / "checkpoint.npz")
    work = tmp / "work"
    work.mkdir()
    validation = validation_override or (ROOT / "人工设计工业模型" / ("任务%s" % dataset_key) / "package" / "public" / "validation.jsonl")
    frame = start_frame(dataset_key, task, metric, direction,
                        ROOT / "人工设计工业模型" / ("任务%s" % dataset_key) / "package" / "public" / "train.jsonl",
                        validation,
                        previous / "checkpoint.npz")
    os.environ["CONTEST_WORK"] = str(work)
    os.environ["CONTEST_PARENT_ROOT"] = str(previous)

    def reply(messages):
        return {"request_id": "call-1",
                "choices": [{"message": {"role": "assistant", "content": reply_text}}]}

    host = Host(frame, reply)
    host.start()

    from contest_harness.round import Round
    os.environ["CONTEST_SUBMISSION"] = "contest_submission"
    import contest_submission
    rnd = Round.start(reader=host.wire_in, writer=host.wire_out)
    print("[sim] round started", flush=True)
    contest_submission.train(rnd)
    print("[sim] train() returned", flush=True)
    host.join(timeout=60)
    print("[sim] host done, failure=", host.failure, flush=True)
    if host.failure:
        raise host.failure
    return rnd, host, work


def main():
    root = Path(__file__).resolve().parents[2]
    parent = root / "official" / "artifacts" / "B" / "checkpoint.npz"
    reply = json.dumps({"recipe": "center035_sev2_end", "epochs": 2,
                        "hypothesis": "对末日段加权的条件均值做轴承内中心化，验证单快照判别力。",
                        "change_summary": "recipe=center035_sev2_end epochs=2"}, ensure_ascii=False)
    rnd, host, work = run_train("B", "regression", "rmse", "lower", parent, reply)
    print("record:", json.dumps(host.record, ensure_ascii=False)[:600])
    candidate = work / "candidate-r1" / "checkpoint.npz"
    print("candidate exists:", candidate.is_file(), candidate.stat().st_size if candidate.is_file() else 0, "bytes")
    sys.path.insert(0, str(work / "candidate-r1"))
    from rulkit.model import Model
    model = Model.load(candidate)
    print("meta:", {k: model.meta[k] for k in ("task", "steps", "epochs")}, "notes:", getattr(model, "recipe_notes", []))


if __name__ == "__main__":
    main()