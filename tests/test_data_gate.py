import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import checksums
import data_gate

CFG = {
    "data_version": "test", "file": "control.npz", "checksum_file": "SHA256SUMS",
    "required_arrays": ["X", "Y", "utt", "ids"], "frame_dtype": "float32", "n_features": 24,
    "min_phrases": 2, "max_phrases": 5, "min_frames": 100, "max_frames": 1000,
    "min_frames_per_phrase": 20, "value_abs_max": 100.0,
}


def make(tmp_path, **override):
    rng = np.random.default_rng(1)
    d = {
        "X": rng.normal(size=(200, 24)).astype(np.float32),
        "Y": rng.normal(size=(200, 24)).astype(np.float32),
        "utt": np.repeat(np.arange(4), 50).astype(np.int32),
        "ids": np.array(["a", "b", "c", "d"]),
    }
    d.update(override)
    p = tmp_path / "control.npz"
    np.savez(p, **d)
    (tmp_path / "SHA256SUMS").write_text(f"{checksums.sha256_file(p)}  control.npz\n")
    (tmp_path / "cfg.json").write_text(json.dumps(CFG))
    return p


def test_good_data_passes(tmp_path):
    make(tmp_path)
    assert data_gate.run(CFG, root=tmp_path) == []


def test_changed_file_fails_hash(tmp_path):
    p = make(tmp_path)
    copy = tmp_path / "copy.npz"
    shutil.copy(p, copy)
    with np.load(p) as f:
        d = {k: f[k] for k in f.files}
    d["X"][0, 0] += 1
    np.savez(copy, **d)
    assert any("SHA256" in x for x in data_gate.run(CFG, file=copy, root=tmp_path))


@pytest.mark.parametrize("override, word", [
    ({"X": np.full((200, 24), np.nan, dtype=np.float32)}, "NaN"),
    ({"X": np.full((200, 24), 1e6, dtype=np.float32)}, "больше допустимого"),
    ({"X": np.zeros((200, 23), dtype=np.float32), "Y": np.zeros((200, 23), dtype=np.float32)}, "форма"),
    ({"X": np.zeros((200, 24), dtype=np.float64)}, "тип"),
    ({"Y": np.zeros((199, 24), dtype=np.float32)}, "разной формы"),
    ({"utt": np.repeat(np.arange(4)[::-1], 50).astype(np.int32)}, "по возрастанию"),
    ({"ids": np.array(["a", "b", "c"])}, "ids"),
    ({"ids": np.array(["a", "a", "b", "c"])}, "повторы"),
])
def test_violations_detected(tmp_path, override, word):
    make(tmp_path, **override)
    assert any(word in x for x in data_gate.run(CFG, root=tmp_path))


def test_missing_array(tmp_path):
    p = make(tmp_path)
    with np.load(p) as f:
        np.savez(p, X=f["X"], Y=f["Y"])
    assert any("обязательных" in x for x in data_gate.run(CFG, root=tmp_path))


def test_pickle_file_rejected(tmp_path):
    p = make(tmp_path)
    np.savez(p, X=np.array([object()], dtype=object))
    assert any("не читается" in x for x in data_gate.run(CFG, root=tmp_path))


def test_missing_file(tmp_path):
    (tmp_path / "SHA256SUMS").write_text("")
    assert data_gate.run(CFG, root=tmp_path)


def test_cli_exit_codes(tmp_path, monkeypatch):
    make(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert data_gate.main(["--config", "cfg.json"]) == 0
    bad = Path("bad.npz")
    np.savez(bad, X=np.zeros((3, 24), dtype=np.float32))
    assert data_gate.main(["--config", "cfg.json", "--file", str(bad)]) == 1


def test_real_config_is_consistent():
    cfg = json.loads((Path(__file__).parent.parent / "configs" / "data_gate.json").read_text(encoding="utf-8"))
    assert cfg["min_frames"] < cfg["max_frames"] and cfg["min_phrases"] < cfg["max_phrases"]
