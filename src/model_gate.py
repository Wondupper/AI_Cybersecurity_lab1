"""Model quality gate: сравнение кандидатной модели с baseline на контрольной выборке.

Метрика MCD (дБ), меньше - лучше. Кандидат проходит при candidate <= baseline * (1 + допуск).
Baseline и политика лежат в metrics/ci_baseline.json и сами не перезаписываются:
создать файл можно только командой --init-baseline, и только если его ещё нет.
Код возврата: 0 - прошла, 1 - отказ (в том числе при плохой или несовместимой метрике).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import checksums
import gmm_vc
from mcd import mcd

METRIC = "MCD (дБ, кепстр WORLD, коэф. 0 не учитывается), меньше - лучше"
DEFAULT_TOLERANCE = 0.05


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def compare(baseline, candidate, data_version, control_sha256):
    """Возвращает (прошла, подробности). Все сомнительные случаи - отказ."""
    reasons = []
    base = baseline.get("baseline_value")
    tol = baseline.get("relative_tolerance")
    abs_thr = baseline.get("absolute_threshold")
    if baseline.get("direction") != "lower_is_better":
        reasons.append("в baseline нужно direction = lower_is_better")
    if not is_number(candidate):
        reasons.append(f"метрика кандидата не число: {candidate!r}")
    if not is_number(base) or base < 0:
        reasons.append(f"некорректное baseline_value: {base!r}")
    if baseline.get("data_version") != data_version:
        reasons.append("версия данных не совпадает с baseline")
    if baseline.get("control_sha256") != control_sha256:
        reasons.append("SHA256 контрольной выборки не совпадает с baseline")
    threshold = None
    if not reasons:
        if base > 0 and is_number(tol) and tol >= 0:
            threshold = base * (1 + tol)
        elif is_number(abs_thr):
            threshold = abs_thr
        else:
            reasons.append("нет допуска: baseline = 0 и не задан absolute_threshold, либо relative_tolerance неверен")
    if not reasons and candidate > threshold:
        reasons.append(f"MCD кандидата {candidate:.4f} больше порога {threshold:.4f}")
    result = {
        "metric": METRIC, "candidate": candidate if is_number(candidate) else None,
        "baseline_value": base if is_number(base) else None, "threshold": threshold,
        "passed": not reasons, "reasons": reasons,
    }
    return not reasons, result


def candidate_mcd(model_path, control_path):
    model = gmm_vc.load(model_path)
    with np.load(control_path, allow_pickle=False) as f:
        x, y = f["X"], f["Y"]
    return mcd(gmm_vc.convert(model, x), y)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/gmm_M16.npz")
    ap.add_argument("--data-config", default="configs/data_gate.json")
    ap.add_argument("--baseline", default="metrics/ci_baseline.json")
    ap.add_argument("--out", default="reports/model_gate.json")
    ap.add_argument("--init-baseline", action="store_true", help="создать baseline по текущей модели (один раз)")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.data_config).read_text(encoding="utf-8"))
    control = Path(cfg["file"])
    control_sha = checksums.sha256_file(control)
    model_sha = checksums.sha256_file(args.model)

    if args.init_baseline:
        base_path = Path(args.baseline)
        if base_path.exists():
            print(f"{base_path} уже есть, baseline автоматически не перезаписывается")
            return 1
        value = candidate_mcd(args.model, control)
        baseline = {
            "metric": METRIC, "direction": "lower_is_better", "baseline_value": value,
            "relative_tolerance": DEFAULT_TOLERANCE, "absolute_threshold": None,
            "data_version": cfg["data_version"], "control_sha256": control_sha, "model_sha256": model_sha,
        }
        base_path.write_text(json.dumps(baseline, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"baseline записан: MCD {value:.4f} дБ")
        return 0

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    try:
        cand = candidate_mcd(args.model, control)
    except (OSError, ValueError, KeyError) as exc:
        cand = None
        load_error = f"не удалось оценить модель: {exc}"
    else:
        load_error = None
    ok, result = compare(baseline, cand, cfg["data_version"], control_sha)
    if load_error:
        result["reasons"].append(load_error)
        ok = result["passed"] = False
    result.update(model=args.model, model_sha256=model_sha, control_sha256=control_sha,
                  data_version=cfg["data_version"], exit_code=0 if ok else 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1, ensure_ascii=False))
    print("model gate:", "OK" if ok else "ОТКАЗ")
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
