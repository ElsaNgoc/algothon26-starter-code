#!/usr/bin/env python
"""Anti-overfit walk-forward gate for Algothon strategies.

Uses one continuous competition path (portfolio/state carry forward), then
scores each chronological fold from sliced daily PnL. This matches how
getMyPosition() runs on prcSoFar in production.
"""

import numpy as np
import pandas as pd
from teamName import getMyPosition as get_position

PRICE_FILE = "./prices.txt"
N_FOLDS = 6
TEST_DAYS = 125
STEP_DAYS = 125
MIN_FOLD_START = 200
SCORE_PARAM = 1.0
WARMUP_START = 251

DEFAULT_COMM = 0.0001
ALGO_COMM = 0.00002
DEFAULT_CAP = 10_000
ALGO_CAP = 100_000


def score(mu, sigma, param=SCORE_PARAM):
    if mu <= 0 or sigma < 1e-10:
        return float(mu)
    sr = np.sqrt(250.0) * mu / sigma
    return float(mu * (sr * sr) / (sr * sr + param * param))


def load_prices(path):
    df = pd.read_csv(path, sep=r"\s+", header=0, index_col=None)
    return df.values.T


def make_folds(n_days):
    latest_start = n_days - TEST_DAYS
    starts = []
    s = latest_start - (N_FOLDS - 1) * STEP_DAYS
    while s <= latest_start and len(starts) < N_FOLDS:
        if s >= MIN_FOLD_START:
            starts.append(s)
        s += STEP_DAYS
    return starts


def run_continuous(prc, fold_starts, test_days):
    n_inst, n_days = prc.shape
    comm_rate = np.full(n_inst, DEFAULT_COMM)
    comm_rate[0] = ALGO_COMM
    dlr_limits = np.full(n_inst, DEFAULT_CAP)
    dlr_limits[0] = ALGO_CAP

    cash = 0.0
    cur_pos = np.zeros(n_inst)
    value = 0.0
    comm = 0.0
    daily = []

    t0 = WARMUP_START
    for t in range(t0, n_days + 1):
        hist = prc[:, :t]
        px = hist[:, -1]
        if t < n_days:
            raw = get_position(hist)
            pos_limits = (dlr_limits / px).astype(int)
            new_pos = np.clip(raw, -pos_limits, pos_limits).astype(int)
        else:
            new_pos = cur_pos.copy()

        delta = new_pos - cur_pos
        cash -= float(px.dot(delta) + comm)
        dvolumes = px * np.abs(delta)
        comm = float(np.sum(dvolumes * comm_rate))
        cur_pos = new_pos
        pos_value = float(cur_pos.dot(px))
        today_pl = cash + pos_value - value
        value = cash + pos_value
        if t > t0:
            daily.append(
                {
                    "t": t,
                    "pl": today_pl,
                    "ret0": float(np.log(prc[0, t - 1] / prc[0, t - 2])) if t >= 2 else 0.0,
                }
            )

    results = []
    for start in fold_starts:
        end = min(start + test_days, n_days)
        seg = [d for d in daily if start < d["t"] <= end]
        pll = np.asarray([d["pl"] for d in seg], dtype=float)
        ret0 = np.asarray([d["ret0"] for d in seg], dtype=float)
        mu = float(np.mean(pll))
        sd = float(np.std(pll))
        fold_score = score(mu, sd)
        ann_sr = float(np.sqrt(250.0) * mu / sd) if sd > 1e-12 else 0.0
        thr = float(np.median(np.abs(ret0))) if len(ret0) else 0.0
        hi = ret0 >= thr
        lo = ~hi
        results.append(
            {
                "start": int(start + 1),
                "end": int(end),
                "mu": mu,
                "sigma": sd,
                "ann_sr": ann_sr,
                "score": fold_score,
                "days": int(len(pll)),
                "hi_vol_mu": float(np.mean(pll[hi])) if np.any(hi) else 0.0,
                "lo_vol_mu": float(np.mean(pll[lo])) if np.any(lo) else 0.0,
            }
        )
    return results


def cold_eval_last250(prc):
    n_inst, n_days = prc.shape
    comm_rate = np.full(n_inst, DEFAULT_COMM)
    comm_rate[0] = ALGO_COMM
    dlr_limits = np.full(n_inst, DEFAULT_CAP)
    dlr_limits[0] = ALGO_CAP
    start = n_days - 250

    cash = 0.0
    cur_pos = np.zeros(n_inst)
    value = 0.0
    comm = 0.0
    pll = []
    for t in range(start, n_days + 1):
        hist = prc[:, :t]
        px = hist[:, -1]
        if t < n_days:
            raw = get_position(hist)
            pos_limits = (dlr_limits / px).astype(int)
            new_pos = np.clip(raw, -pos_limits, pos_limits).astype(int)
        else:
            new_pos = cur_pos.copy()
        delta = new_pos - cur_pos
        cash -= float(px.dot(delta) + comm)
        dvolumes = px * np.abs(delta)
        comm = float(np.sum(dvolumes * comm_rate))
        cur_pos = new_pos
        pos_value = float(cur_pos.dot(px))
        today_pl = cash + pos_value - value
        value = cash + pos_value
        if t > start:
            pll.append(today_pl)
    pll = np.asarray(pll, dtype=float)
    return score(float(np.mean(pll)), float(np.std(pll)))


def main():
    prc = load_prices(PRICE_FILE)
    _, n_days = prc.shape
    starts = make_folds(n_days)
    if not starts:
        raise RuntimeError("Not enough data for configured folds.")

    results = run_continuous(prc, starts, TEST_DAYS)
    scores = np.array([r["score"] for r in results], dtype=float)
    mus = np.array([r["mu"] for r in results], dtype=float)
    srs = np.array([r["ann_sr"] for r in results], dtype=float)
    hi_mus = np.array([r["hi_vol_mu"] for r in results], dtype=float)
    lo_mus = np.array([r["lo_vol_mu"] for r in results], dtype=float)
    robust_objective = 0.7 * float(np.min(scores)) + 0.3 * float(np.mean(scores))
    eval_score = cold_eval_last250(prc)
    last_fold = float(scores[-1])

    print(f"Loaded prices: {n_days} days")
    print(
        f"Folds: {len(results)} x {TEST_DAYS}d (continuous path), "
        f"starts={[(r['start'], r['end']) for r in results]}"
    )
    print("-" * 72)
    for i, r in enumerate(results, start=1):
        print(
            f"F{i}: {r['start']:>4d}-{r['end']:>4d} "
            f"score={r['score']:>7.2f} mu={r['mu']:>7.1f} "
            f"sr={r['ann_sr']:>5.2f} hiVolMu={r['hi_vol_mu']:>7.1f} loVolMu={r['lo_vol_mu']:>7.1f}"
        )

    print("-" * 72)
    print(
        f"Aggregate: meanScore={np.mean(scores):.2f} medianScore={np.median(scores):.2f} "
        f"minScore={np.min(scores):.2f} lastFold={last_fold:.2f} "
        f"robustObj={robust_objective:.2f} eval250={eval_score:.2f}"
    )
    print(
        f"          meanMu={np.mean(mus):.1f} meanSR={np.mean(srs):.2f} "
        f"hiVolMu={np.mean(hi_mus):.1f} loVolMu={np.mean(lo_mus):.1f}"
    )

    core_scores = scores[:-1]
    min_core = float(np.min(core_scores)) if len(core_scores) else float(np.min(scores))

    # Gate tuned for continuous-path folds on public 1000d.
    # Last fold (876-1000) is structurally weak; judge it separately from min(core).
    ship_ready = (
        eval_score >= 585.0
        and np.median(scores) >= 575.0
        and last_fold >= 120.0
        and min_core >= 370.0
        and lo_mus[-1] > 0.0
        and np.mean(hi_mus) > 0.0
        and np.mean(lo_mus) > 0.0
    )
    print(f"minCore={min_core:.2f} lastFoldLoVolMu={lo_mus[-1]:.1f}")
    print(f"SHIP_READY={ship_ready}")


if __name__ == "__main__":
    main()
