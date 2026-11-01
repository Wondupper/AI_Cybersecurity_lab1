"""Единая функция предсказания: проверка входа -> предобработка -> модель -> WAV."""
import json
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf

import gmm_vc

FS = 16000
N_MC = 24
MAX_SECONDS = 10.0
MAX_BYTES = 1_000_000
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "gmm_M16.npz"
STATS_PATH = ROOT / "models" / "f0_stats.json"


class InputError(ValueError):
    """Вход отклонён; модель не вызывалась."""


def validate_wav(path):
    p = Path(path)
    if not p.is_file():
        raise InputError("файл не найден")
    size = p.stat().st_size
    if size == 0 or size > MAX_BYTES:
        raise InputError(f"размер файла должен быть от 1 байта до {MAX_BYTES} байт")
    try:
        info = sf.info(str(p))
    except Exception as exc:
        raise InputError("файл повреждён или не является аудио") from exc
    if info.format != "WAV" or info.subtype != "PCM_16":
        raise InputError("нужен WAV, PCM 16 бит")
    if info.samplerate != FS or info.channels != 1:
        raise InputError("нужна частота 16 кГц и один канал")
    if info.frames == 0 or info.frames / FS > MAX_SECONDS:
        raise InputError(f"длительность должна быть от 0 до {MAX_SECONDS} с")
    try:
        x, _ = sf.read(str(p), dtype="float64")
    except Exception as exc:
        raise InputError("не удалось декодировать файл") from exc
    if not np.all(np.isfinite(x)):
        raise InputError("в сигнале есть NaN или Inf")
    return x


def load_artifacts():
    model = gmm_vc.load(MODEL_PATH)
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    return model, stats


def convert_audio(x, model, stats):
    f0, t = pw.harvest(x, FS)
    sp = pw.cheaptrick(x, f0, t, FS)
    ap = pw.d4c(x, f0, t, FS)
    mc = pw.code_spectral_envelope(sp, FS, N_MC)
    mc2 = gmm_vc.convert(model, mc)
    sp2 = pw.decode_spectral_envelope(np.ascontiguousarray(mc2), FS, pw.get_cheaptrick_fft_size(FS))
    f0_2 = np.zeros_like(f0)
    v = f0 > 0
    s, d = stats["src"], stats["tgt"]
    f0_2[v] = np.exp((np.log(f0[v]) - s["mean"]) / s["std"] * d["std"] + d["mean"])
    y = pw.synthesize(f0_2, np.ascontiguousarray(sp2), ap, FS)
    peak = np.abs(y).max()
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y


def predict(path, model=None, stats=None):
    """Возвращает int16-массив 16 кГц. При плохом входе бросает InputError."""
    x = validate_wav(path)
    if model is None or stats is None:
        model, stats = load_artifacts()
    y = convert_audio(x, model, stats)
    return (y * 32767).astype(np.int16)


if __name__ == "__main__":
    import sys
    sf.write(sys.argv[2], predict(sys.argv[1]), FS, subtype="PCM_16")
