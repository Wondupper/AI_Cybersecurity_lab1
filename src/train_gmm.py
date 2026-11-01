"""Обучение JD-GMM на train, оценка MCD на val (test не трогаем)."""
import resource
import sys
import time
from pathlib import Path

import numpy as np

import gmm_vc
from mcd import mcd

SEED = 42


def main():
    n_comp = int(sys.argv[1])
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else None
    tr = np.load("data/aligned/train.npz", allow_pickle=False)
    va = np.load("data/aligned/val.npz", allow_pickle=False)
    Xtr, Ytr, Xva, Yva = tr["X"], tr["Y"], va["X"], va["Y"]
    mean_pred = np.tile(Ytr.mean(0), (len(Yva), 1))
    print(f"ориентиры на val: копия источника {mcd(Xva, Yva):.3f} дБ | "
          f"среднее цели {mcd(mean_pred, Yva):.3f} дБ")
    t0 = time.time()
    model, converged, n_iter = gmm_vc.fit(Xtr, Ytr, n_comp, SEED, max_frames)
    t_fit = time.time() - t0
    score = mcd(gmm_vc.convert(model, Xva), Yva)
    print(f"M={n_comp} кадров={max_frames or len(Xtr)} итераций={n_iter} "
          f"сошлось={converged} обучение={t_fit:.0f} с  MCD(val)={score:.3f} дБ")
    print(f"пик памяти {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} МБ")
    Path("models").mkdir(exist_ok=True)
    gmm_vc.save(model, f"models/gmm_M{n_comp}.npz")


if __name__ == "__main__":
    main()
