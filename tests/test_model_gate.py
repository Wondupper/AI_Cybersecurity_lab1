import json
import math

import numpy as np
import pytest

import gmm_vc
import model_gate

BASE = {
    "direction": "lower_is_better", "baseline_value": 7.6, "relative_tolerance": 0.05,
    "absolute_threshold": None, "data_version": "v1", "control_sha256": "abc",
}


def gate(candidate, **override):
    return model_gate.compare({**BASE, **override}, candidate, "v1", "abc")


def test_threshold_is_five_percent_above_baseline():
    ok, res = gate(7.6)
    assert ok and res["threshold"] == pytest.approx(7.98)


def test_boundary_and_above():
    assert gate(7.97)[0]
    assert not gate(7.99)[0]
    assert not gate(14.0)[0]


@pytest.mark.parametrize("bad", [None, "7.0", float("nan"), math.inf, True])
def test_bad_candidate_rejected(bad):
    assert not gate(bad)[0]


def test_incompatible_data_rejected():
    assert not model_gate.compare(BASE, 7.0, "other", "abc")[0]
    assert not model_gate.compare(BASE, 7.0, "v1", "other")[0]


def test_bad_baseline_rejected():
    assert not gate(7.0, baseline_value=None)[0]
    assert not gate(7.0, baseline_value=float("nan"))[0]
    assert not gate(7.0, direction="higher_is_better")[0]


def test_zero_baseline_needs_absolute_threshold():
    assert not gate(0.1, baseline_value=0.0)[0]
    assert gate(0.1, baseline_value=0.0, absolute_threshold=0.5)[0]
    assert not gate(0.6, baseline_value=0.0, absolute_threshold=0.5)[0]


def test_real_baseline_file_is_well_formed():
    from pathlib import Path
    p = Path(__file__).parent.parent / "metrics" / "ci_baseline.json"
    if p.exists():
        b = json.loads(p.read_text(encoding="utf-8"))
        assert b["relative_tolerance"] == 0.05 and b["direction"] == "lower_is_better"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Маленькая контрольная выборка и две модели: рабочая и константная."""
    import checksums
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 24))
    y = 0.5 * x + 1.0
    good, _, _ = gmm_vc.fit(x, y, 2, 42)
    weak = {**good, "A": np.zeros_like(good["A"]), "muy": np.zeros_like(good["muy"])}
    gmm_vc.save(good, tmp_path / "good.npz")
    gmm_vc.save(weak, tmp_path / "weak.npz")
    np.savez(tmp_path / "control.npz", X=x.astype(np.float32), Y=y.astype(np.float32))
    cfg = {"file": "control.npz", "data_version": "toy"}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    monkeypatch.chdir(tmp_path)
    assert checksums.sha256_file("control.npz")
    return tmp_path


def run(args):
    return model_gate.main(["--data-config", "cfg.json", "--baseline", "base.json", "--out", "out/r.json", *args])


def test_init_baseline_then_gate_passes_and_weak_model_fails(workspace):
    assert run(["--model", "good.npz", "--init-baseline"]) == 0
    assert run(["--model", "good.npz"]) == 0
    assert json.loads((workspace / "out/r.json").read_text())["exit_code"] == 0
    assert run(["--model", "weak.npz"]) == 1
    res = json.loads((workspace / "out/r.json").read_text())
    assert res["exit_code"] == 1 and res["candidate"] > res["threshold"]


def test_baseline_is_not_overwritten(workspace):
    assert run(["--model", "good.npz", "--init-baseline"]) == 0
    before = (workspace / "base.json").read_text()
    assert run(["--model", "weak.npz", "--init-baseline"]) == 1
    assert (workspace / "base.json").read_text() == before


def test_broken_model_file_rejected(workspace):
    assert run(["--model", "good.npz", "--init-baseline"]) == 0
    (workspace / "broken.npz").write_bytes(b"not a model")
    assert run(["--model", "broken.npz"]) == 1
