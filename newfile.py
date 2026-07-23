"""
Lag-1 cross-sectional ridge ensemble; top/bottom-25 book at caps.

ALGO-PRIMARY MODE (USE_ALGO_PRIMARY): trades ALGO at full $100k by the sign
of a dedicated predictor -- ALGO's next return regressed on ALL assets'
lagged returns (others as correlation explanatory variables), z-scored vs
its own history -- layered on the book, which acts as the amplifier sleeve.
MEASURED: predictor hit rate 52.6% (coin flip). Mode scores 623.1 on days
501-750 but 564.4 on days 251-500 vs the book-only 566.6/612.6 -- a split
decision: the ALGO leg adds ~+/-$50/day of unpredictable P&L that won one
window by luck and lost the other. Default OFF: for the hidden 250 days a
coin-flip $100k position is variance without expected value.
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

USE_ALGO_PRIMARY = False     # True: trade ALGO at sign(pred) x $100k
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

    if USE_ALGO_PRIMARY:
        _algo_hist.append(float(pred[ALGO_IDX]))
        if len(_algo_hist) >= ALGO_MIN_HIST:
            h = np.array(_algo_hist)
            sd = h.std()
            if sd > 1e-12:
                z = (h[-1] - h.mean()) / sd
                dollars[ALGO_IDX] = np.sign(z) * ALGO_CAP

    shares = np.round(dollars / last_prices).astype(int)
    mx = np.floor(POSITION_CAPS / last_prices).astype(int)
    return np.clip(shares, -mx, mx)
