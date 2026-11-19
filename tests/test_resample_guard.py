"""重采样的 ID 闸门：加后缀会让官方的「训练/验证 ID 不得重叠」检查失效，必须自己补上。

官方 `baselines/train.py:65`：`if set(train["ids"]) & set(validation["ids"]): raise`。
我们为了末端加权会重采样训练行，并把 ID 改成 `xxx#w00042`（官方 reader 要求 ID 唯一）。
后缀一加，交集就不可能命中——**这道闸门被我们改没了**。数据本身是官方分好的、
确实没交集，所以现状无害；但"无害"必须由一次显式检查保证，不能由"应该没有"保证。

Run from `official/`:
    python -m pytest tests/test_resample_guard.py -q
"""
import json

import pytest

from harness import baseline


def write_pair(root, train_rows, val_rows):
    train, val = root / "train.jsonl", root / "validation.jsonl"
    train.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train_rows),
                     encoding="utf-8")
    val.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in val_rows),
                   encoding="utf-8")
    return train, val


@pytest.fixture
def pointed(tmp_path, monkeypatch):
    """把本轮数据指到临时文件：resampled_train 只走 data_files()，不碰包内容。"""
    out = tmp_path / "round-01"
    out.mkdir()
    monkeypatch.setenv("ZHIXUN_SCRATCH", str(tmp_path / "_scratch"))
    return out, tmp_path, monkeypatch


def row(rid, answer=120.0):
    return {"id": rid, "inputs": {"features": [1.0, 2.0]}, "answer": answer}


def test_overlap_in_original_ids_is_caught_before_suffixing(pointed):
    out, tmp, monkeypatch = pointed
    train, val = write_pair(tmp, [row("A"), row("B")], [row("B"), row("C")])   # B 同时出现在两边
    monkeypatch.setenv("ZHIXUN_TRAIN_FILE", str(train))
    monkeypatch.setenv("ZHIXUN_VALIDATION_FILE", str(val))
    with pytest.raises(ValueError, match="官方闸门"):
        baseline.resampled_train("B", out, "sqrt_inv", seed=1, persist=True)


def test_disjoint_ids_pass_and_get_unique_suffixes(pointed):
    out, tmp, monkeypatch = pointed
    train, val = write_pair(tmp, [row("A"), row("B"), row("C")], [row("D"), row("E")])
    monkeypatch.setenv("ZHIXUN_TRAIN_FILE", str(train))
    monkeypatch.setenv("ZHIXUN_VALIDATION_FILE", str(val))
    path, info = baseline.resampled_train("B", out, "sqrt_inv", seed=1, persist=True)
    assert info["id_overlap_checked"] is True and info["id_overlap"] == 0
    ids = [json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(ids) == len(set(ids))                      # 官方 reader 的唯一性要求
    assert all("#w" in i for i in ids)                    # 确实加了后缀——所以那道检查不可省
    assert info["rows"] == 3
