"""Выравнивание двух последовательностей признаков методом DTW."""
import numpy as np
from scipy.spatial.distance import cdist


def dtw_path(a, b):
    """a: (n, d), b: (m, d). Возвращает индексы пути (i, j) и полную стоимость."""
    n, m = len(a), len(b)
    c = cdist(a, b)                      # евклидовы расстояния между кадрами
    D = np.empty((n, m))
    D[0] = np.cumsum(c[0])
    for i in range(1, n):
        prev = D[i - 1]
        best = prev.copy()
        np.minimum(prev[1:], prev[:-1], out=best[1:])
        s = np.cumsum(c[i])
        D[i] = s + np.minimum.accumulate(c[i] + best - s)
    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            k = int(np.argmin((D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])))
            if k == 0:
                i, j = i - 1, j - 1
            elif k == 1:
                i -= 1
            else:
                j -= 1
        path.append((i, j))
    path.reverse()
    p = np.array(path)
    return p[:, 0], p[:, 1], float(D[-1, -1])
