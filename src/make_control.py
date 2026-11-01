"""Контрольная выборка для CI: 20 фраз из val (готовые выровненные пары кадров).

Запускается один раз на ВМ, где есть data/aligned/val.npz. Результат лежит в Git (data/control/).
Файл нельзя пересоздавать без обновления SHA256 и baseline: контрольная выборка должна быть неизменной.
"""
import json
from pathlib import Path

import numpy as np

import checksums

N_PHRASES = 20
SEED = 42
OUT = Path("data/control")

va = np.load("data/aligned/val.npz", allow_pickle=False)
ids, utt = va["ids"], va["utt"]
pick = np.sort(np.random.default_rng(SEED).choice(len(ids), size=N_PHRASES, replace=False))
mask = np.isin(utt, pick)
X, Y = va["X"][mask], va["Y"][mask]
new_utt = np.searchsorted(pick, utt[mask]).astype(np.int32)

OUT.mkdir(parents=True, exist_ok=True)
target = OUT / "control_val.npz"
np.savez_compressed(target, X=X, Y=Y, utt=new_utt, ids=ids[pick])
sha = checksums.sha256_file(target)
(OUT / "SHA256SUMS").write_text(f"{sha}  data/control/control_val.npz\n", encoding="utf-8")
splits_sha = checksums.sha256_file("data/splits.json")
info = {
    "phrases": N_PHRASES, "pairs_of_frames": int(len(X)), "file_bytes": target.stat().st_size,
    "sha256": sha, "splits_sha256": splits_sha,
    "X_min": float(X.min()), "X_max": float(X.max()), "Y_min": float(Y.min()), "Y_max": float(Y.max()),
    "ids": [str(i) for i in ids[pick]],
}
print(json.dumps(info, indent=1, ensure_ascii=False))
