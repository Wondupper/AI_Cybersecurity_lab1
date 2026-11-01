import numpy as np
import pytest
import soundfile as sf

import predict


def wav(path, seconds=1.0, fs=16000, channels=1, subtype="PCM_16"):
    t = np.arange(int(seconds * fs)) / fs
    x = 0.3 * np.sin(2 * np.pi * 200 * t)
    if channels == 2:
        x = np.stack([x, x], axis=1)
    sf.write(str(path), x, fs, subtype=subtype)
    return path


@pytest.fixture
def no_model(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("модель не должна вызываться")
    monkeypatch.setattr(predict.gmm_vc, "convert", boom)


def test_not_audio(tmp_path, no_model):
    p = tmp_path / "a.wav"
    p.write_bytes(b"that not sound" * 100)
    with pytest.raises(predict.InputError):
        predict.predict(p, model={}, stats={})


def test_empty_file(tmp_path, no_model):
    p = tmp_path / "a.wav"
    p.write_bytes(b"")
    with pytest.raises(predict.InputError):
        predict.predict(p, model={}, stats={})


def test_wrong_rate(tmp_path, no_model):
    with pytest.raises(predict.InputError):
        predict.predict(wav(tmp_path / "a.wav", fs=8000), model={}, stats={})


def test_stereo(tmp_path, no_model):
    with pytest.raises(predict.InputError):
        predict.predict(wav(tmp_path / "a.wav", channels=2), model={}, stats={})


def test_too_long(tmp_path, no_model, monkeypatch):
    monkeypatch.setattr(predict, "MAX_SECONDS", 1.0)
    with pytest.raises(predict.InputError):
        predict.predict(wav(tmp_path / "a.wav", seconds=1.5), model={}, stats={})


def test_valid_input():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        y = predict.predict(wav(Path(d) / "a.wav", seconds=1.0))
    assert y.dtype == np.int16 and len(y) > 0
    assert abs(len(y) - 16000) < 1000


def test_file_not_found(tmp_path, no_model):
    with pytest.raises(predict.InputError):
        predict.predict(tmp_path / "missing.wav", model={}, stats={})


def test_file_too_big(tmp_path, no_model, monkeypatch):
    monkeypatch.setattr(predict, "MAX_BYTES", 1000)
    with pytest.raises(predict.InputError):
        predict.predict(wav(tmp_path / "a.wav", seconds=1.0), model={}, stats={})


def test_float_wav_rejected(tmp_path, no_model):
    with pytest.raises(predict.InputError):
        predict.predict(wav(tmp_path / "a.wav", subtype="FLOAT"), model={}, stats={})


def test_nan_in_signal_rejected(tmp_path, no_model, monkeypatch):
    p = wav(tmp_path / "a.wav")
    monkeypatch.setattr(predict.sf, "read", lambda *a, **k: (np.full(16000, np.nan), 16000))
    with pytest.raises(predict.InputError):
        predict.predict(p, model={}, stats={})


def test_convert_audio_output_properties():
    model, stats = predict.load_artifacts()
    t = np.arange(16000) / 16000
    y = predict.convert_audio(0.3 * np.sin(2 * np.pi * 200 * t), model, stats)
    assert np.all(np.isfinite(y))
    assert np.abs(y).max() <= 0.99
    assert abs(len(y) - 16000) < 1000
