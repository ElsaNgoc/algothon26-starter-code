"""
Lag-1 cross-sectional ridge ensemble + ALGO own-AR overlay.

Causal / no lookahead: multivariate ridge on expanding history for the
50 non-ALGO book (long/short top-25 at $10k). ALGO traded via own-AR(1)
z-scored predictor at ALGO_TRADE_CAP.

Hardcoded lead-lag LEADERS overfit public prices (local ~900, LB ~300).
Walk-forward on 1000d prices (windows 251-500 / 501-750 / 751-1000):
  hard_old pairs min≈264; ridge+ownAR $45k min≈524 (chosen).
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

USE_ALGO_OVERLAY = True
ALGO_TRADE_CAP = 45_000.0
ALGO_Z_MIN = 0.0
ALGO_MIN_HIST = 40

_algo_hist = []


def _ridge_multi(X, Y, lams):
    XtX, XtY = X.T @ X, X.T @ Y
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


def _algo_own_ar(rets):
    x = rets[:-1, ALGO_IDX]
    y = rets[1:, ALGO_IDX]
    beta = (x * y).mean() / ((x * x).mean() + 1e-12)
    return float(rets[-1, ALGO_IDX] * beta)


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
