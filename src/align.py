"""Пары выровненных кадров (bdl, slt) для выбранных выборок."""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from dtw import dtw_path

FEAT = Path("data/features")
OUT = Path("data/aligned")


def align_one(utt):
    s = np.load(FEAT / "bdl" / f"{utt}.npz")["mc"].astype(np.float64)
    t = np.load(FEAT / "slt" / f"{utt}.npz")["mc"].astype(np.float64)
    i, j, cost = dtw_path(s[:, 1:], t[:, 1:])
    return s[i].astype(np.float32), t[j].astype(np.float32), len(s), len(t), cost / len(i)


def main():
    splits = json.loads(Path("data/splits.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:]:
        ids = splits[name]
        t0 = time.time()
        with Pool(4) as pool:
            res = pool.map(align_one, ids, chunksize=8)
        X = np.concatenate([r[0] for r in res])
        Y = np.concatenate([r[1] for r in res])
        utt = np.concatenate([np.full(len(r[0]), k, dtype=np.int32) for k, r in enumerate(res)])
        np.savez_compressed(OUT / f"{name}.npz", X=X, Y=Y, utt=utt, ids=np.array(ids))
        n_src = sum(r[2] for r in res)
        print(f"{name}: фраз {len(ids)}, пар кадров {len(X)}, "
              f"кадров bdl {n_src}, средняя стоимость на пару {np.mean([r[4] for r in res]):.3f}, "
              f"{time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
