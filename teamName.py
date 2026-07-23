"""
Lag-1 cross-sectional ridge ensemble; top/bottom-25 book at caps.

ALGO overlay (USE_ALGO_OVERLAY): trades ALGO at ALGO_TRADE_CAP ($45k) when
|z-score of ALGO pred vs its history| >= ALGO_Z_MIN. Full $100k primary was
unstable across windows; sized+threshold overlay raised walk-forward
min(w1,w2) from 566.6 (book-only) to 578.9 on prices.txt (750 days).
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

# Sized ALGO overlay (not full $100k): walk-forward pick vs book-only.
# Book-only min(w1,w2)=566.6; ALGO $45k @ |z|>=0.5 → min=578.9 (both windows).
USE_ALGO_OVERLAY = True
ALGO_TRADE_CAP = 45_000.0    # dollar size when trading ALGO (cap still $100k)
ALGO_Z_MIN = 0.5             # only trade when |z-score of pred| >= this
ALGO_MIN_HIST = 40           # prediction history required before trading ALGO

_algo_hist = []


def _ridge_multi(X, Y, lams):
    XtX, XtY = X.T @ X, X.T @ Y
    out = np.zeros(Y.shape[1] if Y.ndim > 1 else 1, dtype=float)
    p = X.shape[1]
    acc = None
    for lam in lams:
        W = np.linalg.solve(XtX + lam * np.eye(p), XtY)
        acc = W if acc is None else acc + W
    return acc / len(lams)


def compute_signal(prcSoFar):
    rets = np.diff(np.log(prcSoFar), axis=1).T
    if rets.shape[0] < MIN_TRAIN + 1:
        return None
    W = _ridge_multi(rets[:-1], rets[1:], LAM_ENSEMBLE)
    return rets[-1] @ W


def getMyPosition(prcSoFar):
    nins, nt = prcSoFar.shape
    if nt < 2:
        return np.zeros(nins, dtype=int)
    pred = compute_signal(prcSoFar)
    if pred is None:
        return np.zeros(nins, dtype=int)

    last_prices = prcSoFar[:, -1]
    others = np.arange(nins) != ALGO_IDX
    sig = pred[others] - pred[others].mean()
    order = np.argsort(sig)
    oi = np.where(others)[0]

    dollars = np.zeros(nins)
    dollars[oi[order[-TOP_K:]]] = OTHER_CAP
    dollars[oi[order[:TOP_K]]] = -OTHER_CAP

    if USE_ALGO_OVERLAY:
        _algo_hist.append(float(pred[ALGO_IDX]))
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
