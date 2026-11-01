"""Извлечение признаков WORLD (F0 и 24 кепстральных коэффициента) для train/val/test."""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf

RAW = Path("data/raw")
OUT = Path("data/features")
N_MC = 24
SPEAKERS = ("bdl", "slt")


def analyze(x, fs):
    """Возвращает f0 (T,) и mc (T, N_MC) для сигнала x (float64)."""
    f0, t = pw.harvest(x, fs)
    sp = pw.cheaptrick(x, f0, t, fs)
    mc = pw.code_spectral_envelope(sp, fs, N_MC)
    return f0, mc


def process(job):
    spk, utt = job
    dst = OUT / spk / f"{utt}.npz"
    if dst.exists():
        return utt, spk, "skip"
    x, fs = sf.read(RAW / f"cmu_us_{spk}_arctic" / "wav" / f"{utt}.wav")
    if fs != 16000 or x.ndim != 1 or not np.all(np.isfinite(x)):
        return utt, spk, "bad"
    f0, mc = analyze(x.astype(np.float64), fs)
    np.savez_compressed(dst, f0=f0.astype(np.float32), mc=mc.astype(np.float32))
    return utt, spk, "ok"


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    split = json.loads(Path("data/splits.json").read_text(encoding="utf-8"))
    ids = split["train"] + split["val"] + split["test"]
    if limit:
        ids = ids[:limit]
    for spk in SPEAKERS:
        (OUT / spk).mkdir(parents=True, exist_ok=True)
    jobs = [(spk, u) for u in ids for spk in SPEAKERS]
    t0 = time.time()
    stats = {"ok": 0, "skip": 0, "bad": 0}
    with Pool(4) as pool:
        for i, (_, _, status) in enumerate(pool.imap_unordered(process, jobs, chunksize=8), 1):
            stats[status] += 1
            if i % 200 == 0:
                print(f"{i}/{len(jobs)}  {time.time() - t0:.0f} с", flush=True)
    print(stats, f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
