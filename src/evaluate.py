"""Однократная оценка на отложенной тестовой выборке.

Запуск: python src/evaluate.py [модель] [файл результата]
По умолчанию результат пишется в metrics/test_eval.json: файл metrics/baseline.json (эталон ЛР2)
и metrics/ci_baseline.json (эталон CI) этим скриптом не перезаписываются.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import gmm_vc
from mcd import mcd

model_path = sys.argv[1] if len(sys.argv) > 1 else "models/gmm_M16.npz"
out_path = Path(sys.argv[2] if len(sys.argv) > 2 else "metrics/test_eval.json")
if out_path.name in ("baseline.json", "ci_baseline.json"):
    sys.exit("эталонные файлы не перезаписываются автоматически")

tr = np.load("data/aligned/train.npz", allow_pickle=False)
te = np.load("data/aligned/test.npz", allow_pickle=False)
model = gmm_vc.load(model_path)
X, Y = te["X"], te["Y"]
splits_sha = hashlib.sha256(Path("data/splits.json").read_bytes()).hexdigest()
res = {
    "metric": "MCD (дБ, кепстр WORLD, коэф. 0 не учитывается), меньше - лучше",
    "value_gmm": mcd(gmm_vc.convert(model, X), Y),
    "value_copy_source": mcd(X, Y),
    "value_mean_target": mcd(np.tile(tr["Y"].mean(0), (len(Y), 1)), Y),
    "n_test_phrases": len(te["ids"]),
    "n_test_frames": len(X),
    "seed": 42,
    "n_components": 16,
    "train_frames_used": 100000,
    "data_version": "CMU ARCTIC v0.95 bdl/slt; splits.json sha256=" + splits_sha,
}
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False))
