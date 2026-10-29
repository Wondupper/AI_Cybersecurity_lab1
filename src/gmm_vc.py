"""Преобразование спектра голоса совместной гауссовой смесью (JD-GMM)."""
import numpy as np
from scipy.linalg import solve_triangular
from sklearn.mixture import GaussianMixture


def fit(X, Y, n_components, seed, max_frames=None):
    """X, Y: выровненные кадры train (N, d). Нормировка считается только по ним."""
    X, Y = X.astype(np.float64), Y.astype(np.float64)
    d = X.shape[1]
    mx, sx = X.mean(0), X.std(0) + 1e-8
    my, sy = Y.mean(0), Y.std(0) + 1e-8
    if max_frames is not None and len(X) > max_frames:
        idx = np.random.default_rng(seed).choice(len(X), size=max_frames, replace=False)
        X, Y = X[idx], Y[idx]
    Z = np.hstack([(X - mx) / sx, (Y - my) / sy])
    gmm = GaussianMixture(n_components=n_components, covariance_type="full",
                          reg_covar=1e-4, max_iter=100, random_state=seed)
    gmm.fit(Z)
    sxx = gmm.covariances_[:, :d, :d]
    syx = gmm.covariances_[:, d:, :d]
    model = {
        "w": gmm.weights_,
        "mux": gmm.means_[:, :d],
        "muy": gmm.means_[:, d:],
        "A": np.linalg.solve(sxx, syx.transpose(0, 2, 1)).transpose(0, 2, 1),
        "L": np.linalg.cholesky(sxx),
        "mx": mx, "sx": sx, "my": my, "sy": sy,
    }
    return model, bool(gmm.converged_), int(gmm.n_iter_)


def convert(model, x):
    """x: (N, d) кадры источника -> (N, d) оценка кадров цели."""
    xs = (np.asarray(x, dtype=np.float64) - model["mx"]) / model["sx"]
    d = xs.shape[1]
    n_comp = len(model["w"])
    logp = np.empty((len(xs), n_comp))
    for m in range(n_comp):
        z = solve_triangular(model["L"][m], (xs - model["mux"][m]).T, lower=True)
        half_logdet = np.log(np.diag(model["L"][m])).sum()
        logp[:, m] = (np.log(model["w"][m]) - 0.5 * (z ** 2).sum(axis=0)
                      - half_logdet - 0.5 * d * np.log(2 * np.pi))
    logp -= logp.max(axis=1, keepdims=True)
    post = np.exp(logp)
    post /= post.sum(axis=1, keepdims=True)
    ys = np.zeros_like(xs)
    for m in range(n_comp):
        ys += post[:, m:m + 1] * (model["muy"][m] + (xs - model["mux"][m]) @ model["A"][m].T)
    return ys * model["sy"] + model["my"]


def save(model, path):
    np.savez_compressed(path, **model)


def load(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}