import numpy as np
import pytest

import gmm_vc
from mcd import mcd


@pytest.fixture(scope="module")
def toy():
    """Синтетические пары кадров: Y = 0.5 * X + 1 с небольшим шумом (фиксированный seed)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 24))
    y = 0.5 * x + 1.0 + rng.normal(scale=0.05, size=x.shape)
    model, converged, _ = gmm_vc.fit(x[:2500], y[:2500], n_components=2, seed=42)
    return model, converged, x[2500:], y[2500:]


def test_convert_shape_and_finite(toy):
    model, _, x, _ = toy
    out = gmm_vc.convert(model, x)
    assert out.shape == x.shape
    assert np.all(np.isfinite(out))


def test_model_beats_copy_on_known_mapping(toy):
    model, _, x, y = toy
    assert mcd(gmm_vc.convert(model, x), y) < 0.3 * mcd(x, y)


def test_save_load_roundtrip(toy, tmp_path):
    model, _, x, _ = toy
    gmm_vc.save(model, tmp_path / "m.npz")
    loaded = gmm_vc.load(tmp_path / "m.npz")
    assert np.allclose(gmm_vc.convert(model, x), gmm_vc.convert(loaded, x))


def test_load_rejects_pickle(tmp_path):
    p = tmp_path / "evil.npz"
    np.savez(p, a=np.array([object()], dtype=object))
    with pytest.raises(ValueError):
        gmm_vc.load(p)
