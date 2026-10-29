"""Небольшой эксперимент: слабый шум, изменение громкости, нетипичные входы."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import predict
from extract_features import analyze
from mcd import mcd

FS = 16000
model, stats = predict.load_artifacts()
ids = json.loads(Path("data/splits.json").read_text(encoding="utf-8"))["test"][:15]
rng = np.random.default_rng(0)


def conv_mc(x):
    y = predict.convert_audio(x, model, stats)
    return y, analyze(y, FS)[1]


res = {"noise_30dB": [], "gain_0.8": []}
for u in ids:
    x, _ = sf.read(f"data/raw/cmu_us_bdl_arctic/wav/{u}.wav", dtype="float64")
    _, m0 = conv_mc(x)
    noisy = x + rng.normal(0, np.sqrt(np.mean(x ** 2)) / 10 ** 1.5, len(x))
    quiet = x * 0.8
    for name, z in (("noise_30dB", noisy), ("gain_0.8", quiet)):
        _, m1 = conv_mc(z)
        n = min(len(m0), len(m1))
        res[name].append(mcd(m0[:n], m1[:n]))
for k, v in res.items():
    print(f"{k}: MCD между результатами, среднее {np.mean(v):.2f} дБ, макс {np.max(v):.2f} дБ (15 фраз)")

for name, z in (("белый шум 3 с", rng.uniform(-0.3, 0.3, 3 * FS)), ("тишина 2 с", np.zeros(2 * FS))):
    try:
        y, _ = conv_mc(z)
        print(f"{name}: выход {len(y) / FS:.2f} с, пик {np.abs(y).max():.3f}, конечные значения: {bool(np.all(np.isfinite(y)))}")
    except Exception as exc:
        print(f"{name}: ошибка {type(exc).__name__}: {exc}")