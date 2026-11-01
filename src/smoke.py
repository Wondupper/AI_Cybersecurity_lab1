"""Smoke-проверка: хэши модели, загрузка модели, предсказание для синтетического WAV.

Запуск из любого каталога: python src/smoke.py. Код возврата 0 - всё в порядке.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

import checksums
import predict


def main():
    problems = checksums.verify(predict.ROOT / "models" / "SHA256SUMS", predict.ROOT)
    if problems:
        print("smoke: ОТКАЗ, целостность модели:", "; ".join(problems))
        return 1
    t = np.arange(predict.FS) / predict.FS
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "smoke.wav"
        sf.write(str(wav), 0.3 * np.sin(2 * np.pi * 200 * t), predict.FS, subtype="PCM_16")
        y = predict.predict(wav)
    if y.dtype != np.int16 or abs(len(y) - predict.FS) > 1000:
        print(f"smoke: ОТКАЗ, неожиданный выход: {y.dtype}, длина {len(y)}")
        return 1
    print(f"smoke: OK, хэши совпали, выход int16, {len(y)} отсчётов")
    return 0


if __name__ == "__main__":
    sys.exit(main())
