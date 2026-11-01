import numpy as np

from dtw import dtw_path


def naive_cost(a, b):
    n, m = len(a), len(b)
    c = np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1))
    D = np.full((n, m), np.inf)
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                D[i, j] = c[0, 0]
                continue
            prev = min(D[i - 1, j - 1] if i and j else np.inf,
                       D[i - 1, j] if i else np.inf,
                       D[i, j - 1] if j else np.inf)
            D[i, j] = c[i, j] + prev
    return D[-1, -1]


def test_cost_matches_naive():
    rng = np.random.default_rng(0)
    for n, m in [(1, 1), (1, 5), (6, 1), (7, 9), (20, 13)]:
        a, b = rng.normal(size=(n, 3)), rng.normal(size=(m, 3))
        _, _, cost = dtw_path(a, b)
        assert np.isclose(cost, naive_cost(a, b))


def test_path_is_valid():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(15, 4)), rng.normal(size=(11, 4))
    i, j, _ = dtw_path(a, b)
    assert (i[0], j[0]) == (0, 0) and (i[-1], j[-1]) == (14, 10)
    di, dj = np.diff(i), np.diff(j)
    assert np.all((di >= 0) & (di <= 1) & (dj >= 0) & (dj <= 1))
    assert np.all(di + dj >= 1)


def test_identical_sequences_zero_cost():
    a = np.random.default_rng(2).normal(size=(10, 3))
    i, j, cost = dtw_path(a, a)
    assert cost == 0 and np.array_equal(i, j)
