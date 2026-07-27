"""
Dynamic-lambda ridge lead-lag + ALGO own-AR overlay.

Book: lag-1 cross-sectional ridge; lambda regime switches on market trend
strength (|mean|/sd over TREND_WINDOW). Strong trend -> low lambda; break ->
high lambda. Falls back to static ensemble when history is short.

ALGO: separate own-AR(1) z-score overlay ($60k when |z|>=0.5).

Meta-ensemble update:
- book signal = z(dyn ridge) + 0.40 * z(static ridge)
- rank_blend=0.15 to reduce churn in cross-sectional ordering
- strength sizing tested (floor 0.0-0.5): worse than flat $10k legs
"""
import numpy as np

N_INST = 51
ALGO_IDX = 0
ALGO_CAP = 100_000.0
OTHER_CAP = 10_000.0
POSITION_CAPS = np.full(N_INST, OTHER_CAP)
POSITION_CAPS[ALGO_IDX] = ALGO_CAP

MIN_TRAIN = 80
TOP_K = 25

LAM_STATIC = [0.03, 0.1, 0.3]

REGIME_DYNAMIC = True
TREND_WINDOW = 20
TREND_THRESHOLD = 0.30
LAM_TREND = [0.03, 0.1]
LAM_BREAK = [0.3, 1.0]
STATIC_BLEND_W = 0.40
RANK_BLEND_W = 0.15

USE_ALGO_OVERLAY = True
ALGO_TRADE_CAP = 60_000.0
ALGO_Z_MIN = 0.5
ALGO_MIN_HIST = 40

_algo_hist = []
_prev_rank = None


def _algo_own_ar(rets):
    x = rets[:-1, ALGO_IDX]
    y = rets[1:, ALGO_IDX]
    beta = (x * y).mean() / ((x * x).mean() + 1e-12)
    return float(rets[-1, ALGO_IDX] * beta)


def _z(v):
    v = np.asarray(v, float) - np.mean(v)
    sd = np.std(v)
    return v / sd if sd > 1e-12 else v


def _ridge_pred(rets, lams):
    X, Y = rets[:-1], rets[1:]
    XtX, XtY = X.T @ X, X.T @ Y
    p = X.shape[1]
    acc = None
    for lam in lams:
        W = np.linalg.solve(XtX + lam * np.eye(p), XtY)
        acc = W if acc is None else acc + W
    return rets[-1] @ (acc / len(lams))


def _select_lambdas(rets):
    if not REGIME_DYNAMIC or rets.shape[0] < TREND_WINDOW + 1:
        return LAM_STATIC
    mkt = rets[-TREND_WINDOW:].mean(axis=1)
    strength = abs(mkt.mean()) / (mkt.std() + 1e-9)
    return LAM_TREND if strength > TREND_THRESHOLD else LAM_BREAK


def compute_signal(prcSoFar):
    rets = np.diff(np.log(prcSoFar), axis=1).T
    if rets.shape[0] < MIN_TRAIN + 1:
        return None
    pred_dyn = _ridge_pred(rets, _select_lambdas(rets))
    pred_static = _ridge_pred(rets, LAM_STATIC)
    return _z(pred_dyn) + STATIC_BLEND_W * _z(pred_static)


def getMyPosition(prcSoFar):
    global _prev_rank
    nins, nt = prcSoFar.shape
    if nt < 2:
        return np.zeros(nins, dtype=int)
    pred = compute_signal(prcSoFar)
    if pred is None:
        return np.zeros(nins, dtype=int)
    last_prices = prcSoFar[:, -1]
    others = np.arange(nins) != ALGO_IDX
    sig = pred[others] - pred[others].mean()
    oi = np.where(others)[0]
    order = np.argsort(sig)

    # Smooth rank ordering over time to reduce unnecessary churn.
    cur_rank = np.empty(len(oi), dtype=float)
    cur_rank[order] = np.arange(len(oi), dtype=float)
    if _prev_rank is not None:
        mix_rank = (1.0 - RANK_BLEND_W) * cur_rank + RANK_BLEND_W * _prev_rank
        order = np.argsort(mix_rank)
        _prev_rank = mix_rank
    else:
        _prev_rank = cur_rank

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
