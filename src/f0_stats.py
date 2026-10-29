"""Среднее и СКО log-F0 по озвученным кадрам train для обоих дикторов."""
import json
from pathlib import Path

import numpy as np

ids = json.loads(Path("data/splits.json").read_text(encoding="utf-8"))["train"]
out = {}
for key, spk in (("src", "bdl"), ("tgt", "slt")):
    v = np.concatenate([np.load(f"data/features/{spk}/{u}.npz")["f0"] for u in ids])
    lf = np.log(v[v > 0].astype(np.float64))
    out[key] = {"mean": float(lf.mean()), "std": float(lf.std())}
Path("models/f0_stats.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(out)