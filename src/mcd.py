"""Метрика MCD (дБ) по кепстральным коэффициентам WORLD; коэффициент 0 не учитывается."""
import numpy as np

K = 10.0 / np.log(10.0) * np.sqrt(2.0)


def mcd(a, b):
    """a, b: пары кадров формы (N, 24). Возвращает средний MCD в дБ (меньше - лучше)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] == 0 or a.shape[1] < 2:
        raise ValueError("ожидаются два массива одинаковой формы (N, D), N > 0, D >= 2")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("в данных есть NaN или Inf")
    dist = np.sqrt(((a[:, 1:] - b[:, 1:]) ** 2).sum(axis=1))
    return float(K * dist.mean())