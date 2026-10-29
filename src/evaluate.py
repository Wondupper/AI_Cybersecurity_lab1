"""Оценка на отложенной тестовой выборке и запись metrics/baseline.json."""
import hashlib
import json
from pathlib import Path

import numpy as np

import gmm_vc
from mcd import mcd

tr = np.load("data/aligned/train.npz", allow_pickle=False)
te = np.load("data/aligned/test.npz", allow_pickle=False)
model = gmm_vc.load("models/gmm_M16.npz")
X, Y = te["X"], te["Y"]
res = {
    "metric": "MCD (дБ, кепстр WORLD, коэф. 0 не учитывается), меньше - лучше",
    "value_gmm": mcd(gmm_vc.convert(model, X), Y),
    "value_copy_source": mcd(X, Y),
    "value_mean_target": mcd(np.tile(tr["Y"].mean(0), (len(Y), 1)), Y),
    "n_test_phrases": int(len(te["ids"])),
    "n_test_frames": int(len(X)),
    "seed": 42,
    "n_components": 16,
    "train_frames_used": 100000,
    "data_version": "CMU ARCTIC v0.95 bdl/slt; splits.json sha256=" + hashlib.sha256(Path("data/splits.json").read_bytes()).hexdigest(),
}
Path("metrics").mkdir(exist_ok=True)
Path("metrics/baseline.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False))