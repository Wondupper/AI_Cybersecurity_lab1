"""Data quality gate: проверка контрольной выборки по правилам из configs/data_gate.json.

Код возврата 0 - данные в порядке, 1 - нарушена хотя бы одна проверка.
Запуск: python src/data_gate.py [--config configs/data_gate.json] [--file путь.npz]
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

import checksums


def check_arrays(cfg, data):
    """Проверки содержимого. data - словарь массивов. Возвращает список нарушений."""
    bad = []
    missing = [k for k in cfg["required_arrays"] if k not in data]
    if missing:
        return [f"нет обязательных массивов: {missing}"]
    x, y, utt, ids = data["X"], data["Y"], data["utt"], data["ids"]
    for name, arr in (("X", x), ("Y", y)):
        if arr.dtype != np.dtype(cfg["frame_dtype"]):
            bad.append(f"{name}: тип {arr.dtype}, нужен {cfg['frame_dtype']}")
        if arr.ndim != 2 or arr.shape[1] != cfg["n_features"]:
            bad.append(f"{name}: форма {arr.shape}, нужно (N, {cfg['n_features']})")
    if x.shape != y.shape:
        bad.append(f"X и Y разной формы: {x.shape} и {y.shape}")
    if bad:
        return bad
    for name, arr in (("X", x), ("Y", y)):
        if not np.all(np.isfinite(arr)):
            bad.append(f"{name}: есть NaN или Inf")
        elif np.abs(arr).max() > cfg["value_abs_max"]:
            bad.append(f"{name}: значение {np.abs(arr).max():.1f} больше допустимого {cfg['value_abs_max']}")
    n = len(x)
    if not cfg["min_frames"] <= n <= cfg["max_frames"]:
        bad.append(f"число пар кадров {n} вне [{cfg['min_frames']}, {cfg['max_frames']}]")
    if utt.ndim != 1 or len(utt) != n or not np.issubdtype(utt.dtype, np.integer):
        bad.append("utt: нужен целочисленный вектор длиной N")
        return bad
    if np.any(np.diff(utt) < 0):
        bad.append("utt: номера фраз должны идти по возрастанию")
    counts = np.bincount(utt) if utt.min() >= 0 else np.array([])
    n_phr = len(ids)
    if not cfg["min_phrases"] <= n_phr <= cfg["max_phrases"]:
        bad.append(f"число фраз {n_phr} вне [{cfg['min_phrases']}, {cfg['max_phrases']}]")
    if len(counts) != n_phr or np.any(counts == 0):
        bad.append("utt не соответствует списку ids")
    elif counts.min() < cfg["min_frames_per_phrase"]:
        bad.append(f"есть фраза с {counts.min()} кадрами, минимум {cfg['min_frames_per_phrase']}")
    if len(set(ids.tolist())) != n_phr:
        bad.append("в ids есть повторы")
    return bad


def run(cfg, file=None, root="."):
    root = Path(root)
    path = Path(file) if file else root / cfg["file"]
    bad = []
    if not path.is_file():
        return [f"файл {path} не найден"]
    sums = checksums.read_sums(root / cfg["checksum_file"])
    expected = sums.get(cfg["file"])
    if expected is None:
        bad.append(f"в {cfg['checksum_file']} нет эталона для {cfg['file']}")
    elif checksums.sha256_file(path) != expected:
        bad.append("SHA256 файла не совпадает с эталоном")
    try:
        with np.load(path, allow_pickle=False) as f:
            data = {k: f[k] for k in f.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        return [*bad, f"файл не читается как npz без pickle: {exc}"]
    return bad + check_arrays(cfg, data)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_gate.json")
    ap.add_argument("--file", default=None, help="проверить другой файл (например, копию с ошибкой)")
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    problems = run(cfg, args.file)
    print(f"data gate: версия данных «{cfg['data_version']}»")
    if problems:
        for p in problems:
            print("НАРУШЕНИЕ:", p)
        print("data gate: ОТКАЗ")
        return 1
    print("data gate: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
