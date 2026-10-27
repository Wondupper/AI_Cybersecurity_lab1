"""Разделение фраз ARCTIC на train/val/test (по фразам, не по кадрам)."""
import json
import random
from pathlib import Path

SEED = 42
RAW = Path("data/raw")
OUT = Path("data/splits.json")


def main():
    bdl = {p.stem for p in (RAW / "cmu_us_bdl_arctic" / "wav").glob("*.wav")}
    slt = {p.stem for p in (RAW / "cmu_us_slt_arctic" / "wav").glob("*.wav")}
    ids = sorted(bdl & slt)
    rng = random.Random(SEED)
    rng.shuffle(ids)
    n_test = n_val = round(0.1 * len(ids))
    split = {
        "seed": SEED,
        "test": sorted(ids[:n_test]),
        "val": sorted(ids[n_test:n_test + n_val]),
        "train": sorted(ids[n_test + n_val:]),
    }
    OUT.write_text(json.dumps(split, indent=1), encoding="utf-8")
    print({k: len(v) for k, v in split.items() if k != "seed"})
    assert not (set(split["train"]) & set(split["test"]))
    assert not (set(split["train"]) & set(split["val"]))
    assert not (set(split["val"]) & set(split["test"]))


if __name__ == "__main__":
    main()