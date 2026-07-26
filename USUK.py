"""
Causal ridge + light lead-lag blend + ALGO own-AR overlay.

LEADERS are frozen from the FIRST 700 days of the public prices.txt only
(not the eval/LB holdout). Each day betas are re-fit on prcSoFar.
Book signal = z(ridge) + PAIRS_W * z(sparse lead-lag).

Walk-forward note (1000d public file):
  pure ridge+ALGO$45k min≈524 across 3 windows
  ridge+ALGO$60k z>=0.5 min≈558
  freeze700 leaders + pw0.25 + ALGO$70k z>=0.5 → last-250 OOS ≈593
Target 800 not reached without lookahead; this is best validated causal step.
"""
import numpy as np

N_INST = 51
ALGO_IDX = 0
ALGO_CAP = 100_000.0
OTHER_CAP = 10_000.0
POSITION_CAPS = np.full(N_INST, OTHER_CAP)
POSITION_CAPS[ALGO_IDX] = ALGO_CAP

LAM_ENSEMBLE = [0.03, 0.1, 0.3]
MIN_TRAIN = 80
TOP_K = 25
PAIRS_W = 0.25
BETA_LOOKBACK = 200
BETA_LAM = 0.5

USE_ALGO_OVERLAY = True
ALGO_TRADE_CAP = 70_000.0
ALGO_Z_MIN = 0.5
ALGO_MIN_HIST = 40

# Frozen from public prices.txt days 0..700 (exclude last-250 holdout)
LEADERS = [
    [40, 33],
    [20, 25, 37],
    [3, 50, 10],
    [15],
    [33, 47, 0],
    [9, 33, 36],
    [49, 19],
    [40, 38],
    [50, 15, 33],
    [45, 16],
    [33],
    [33, 25],
    [2, 41, 21],
    [37, 9, 25],
    [26, 12, 32],
    [9, 0, 4],
    [37],
    [34, 44, 5],
    [24, 21],
    [45, 9, 46],
    [37, 1, 8],
    [14, 13, 16],
    [37],
    [33],
    [36, 43, 27],
    [40],
    [45, 9, 26],
    [2, 21],
    [20, 37, 22],
    [9, 3, 4],
    [40, 38, 11],
    [10],
    [13],
    [40, 38],
    [1, 44, 3],
    [32],
    [27],
    [22, 16],
    [9, 40, 47],
    [40, 37],
    [4, 47, 23],
    [37],
    [21],
    [4, 16],
    [13],
    [13],
    [10, 46, 50],
    [5],
    [39, 30, 44],
    [9, 47, 40],
    [47, 9, 38],
]

_algo_hist = []


def _ridge_multi(X, Y, lams):
    XtX, XtY = X.T @ X, X.T @ Y
    p = X.shape[1]
    acc = None
    for lam in lams:
        W = np.linalg.solve(XtX + lam * np.eye(p), XtY)
        acc = W if acc is None else acc + W
    return acc / len(lams)


def _zscore(p):
    p = np.asarray(p, dtype=float)
    p = p - p.mean()
    sd = p.std()
    return p / sd if sd > 1e-12 else p


def _ridge_signal(rets):
    W = _ridge_multi(rets[:-1], rets[1:], LAM_ENSEMBLE)
    return rets[-1] @ W


def _pairs_signal(rets):
    lb = min(BETA_LOOKBACK, rets.shape[0] - 1)
    A, B = rets[-lb - 1 : -1], rets[-lb:]
    x = rets[-1]
    pred = np.zeros(rets.shape[1])
    for j in range(rets.shape[1]):
        idx = np.asarray(LEADERS[j], dtype=int)
        X, y = A[:, idx], B[:, j]
        beta = np.linalg.solve(X.T @ X + BETA_LAM * np.eye(len(idx)), X.T @ y)
        pred[j] = x[idx] @ beta
    return pred


def _algo_own_ar(rets):
    x = rets[:-1, ALGO_IDX]
    y = rets[1:, ALGO_IDX]
    beta = (x * y).mean() / ((x * x).mean() + 1e-12)
    return float(rets[-1, ALGO_IDX] * beta)


def compute_signal(prcSoFar):
    rets = np.diff(np.log(prcSoFar), axis=1).T
    if rets.shape[0] < MIN_TRAIN + 1:
        return None
    return _zscore(_ridge_signal(rets)) + PAIRS_W * _zscore(_pairs_signal(rets))


def getMyPosition(prcSoFar):
    nins, nt = prcSoFar.shape
    if nt < 2:
        return np.zeros(nins, dtype=int)

    pred = compute_signal(prcSoFar)
    if pred is None:
        return np.zeros(nins, dtype=int)

    last_prices = prcSoFar[:, -1]
    others = np.arange(nins) != ALGO_IDX
    oi = np.where(others)[0]
    sig = pred[others] - pred[others].mean()
    order = np.argsort(sig)

    dollars = np.zeros(nins)
    dollars[oi[order[-TOP_K:]]] = OTHER_CAP
    dollars[oi[order[:TOP_K]]] = -OTHER_CAP

    if USE_ALGO_OVERLAY:
        rets = np.diff(np.log(prcSoFar), axis=1).T
        _algo_hist.append(_algo_own_ar(rets))
        if len(_algo_hist) >= ALGO_MIN_HIST:
            h = np.array(_algo_hist)
            sd = h.std()
            if sd > 1e-12:
                z = (h[-1] - h.mean()) / sd
                if abs(z) >= ALGO_Z_MIN:
                    dollars[ALGO_IDX] = np.sign(z) * ALGO_TRADE_CAP

    shares = np.round(dollars / last_prices).astype(int)
    mx = np.floor(POSITION_CAPS / last_prices).astype(int)
    return np.clip(shares, -mx, mx)
