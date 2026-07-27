#!/usr/bin/env python
"""Anti-overfit walk-forward gate for Algothon strategies.

Runs multiple chronological folds, reports robust aggregate metrics, and
flags whether current strategy is "ship-ready" under conservative criteria.
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


def run_fold(prc, start_day, test_days):
    n_inst, n_days = prc.shape
    end_day = min(start_day + test_days, n_days)

    comm_rate = np.full(n_inst, DEFAULT_COMM)
    comm_rate[0] = ALGO_COMM
    dlr_limits = np.full(n_inst, DEFAULT_CAP)
    dlr_limits[0] = ALGO_CAP

    cash = 0.0
    cur_pos = np.zeros(n_inst)
    value = 0.0
    comm = 0.0
    tot_dvol = 0.0
    daily_pl = []
    daily_ret0 = []

    for t in range(start_day, end_day + 1):
        hist = prc[:, :t]
        px = hist[:, -1]

        if t < end_day:
            raw = get_position(hist)
            pos_limits = (dlr_limits / px).astype(int)
            new_pos = np.clip(raw, -pos_limits, pos_limits).astype(int)
        else:
            new_pos = cur_pos.copy()

        delta = new_pos - cur_pos
        cash -= float(px.dot(delta) + comm)

        dvolumes = px * np.abs(delta)
        tot_dvol += float(np.sum(dvolumes))
        comm = float(np.sum(dvolumes * comm_rate))

        cur_pos = new_pos
        pos_value = float(cur_pos.dot(px))
        today_pl = cash + pos_value - value
        value = cash + pos_value

        if t > start_day:
            daily_pl.append(today_pl)
            if t >= 2:
                daily_ret0.append(float(np.log(prc[0, t - 1] / prc[0, t - 2])))
            else:
                daily_ret0.append(0.0)

    pll = np.asarray(daily_pl, dtype=float)
    mu = float(np.mean(pll))
    sd = float(np.std(pll))
    fold_score = score(mu, sd)
    ann_sr = float(np.sqrt(250.0) * mu / sd) if sd > 1e-12 else 0.0
    ret = float(value / tot_dvol) if tot_dvol > 0 else 0.0

    # Simple regime split by ALGO realized vol (median threshold inside fold).
    ret0 = np.asarray(daily_ret0, dtype=float)
    absr = np.abs(ret0)
    thr = float(np.median(absr)) if len(absr) > 0 else 0.0
    hi_mask = absr >= thr
    lo_mask = ~hi_mask
    hi_mu = float(np.mean(pll[hi_mask])) if np.any(hi_mask) else 0.0
    lo_mu = float(np.mean(pll[lo_mask])) if np.any(lo_mask) else 0.0

    return {
        "start": int(start_day + 1),
        "end": int(end_day),
        "mu": mu,
        "sigma": sd,
        "ann_sr": ann_sr,
        "score": fold_score,
        "ret": ret,
        "days": int(len(pll)),
        "hi_vol_mu": hi_mu,
        "lo_vol_mu": lo_mu,
    }


def make_folds(n_days):
    latest_start = n_days - TEST_DAYS
    starts = []
    s = latest_start - (N_FOLDS - 1) * STEP_DAYS
    while s <= latest_start and len(starts) < N_FOLDS:
        if s >= MIN_FOLD_START:
            starts.append(s)
        s += STEP_DAYS
    return starts


def main():
    prc = load_prices(PRICE_FILE)
    _, n_days = prc.shape
    starts = make_folds(n_days)
    if not starts:
        raise RuntimeError("Not enough data for configured folds.")

    results = [run_fold(prc, s, TEST_DAYS) for s in starts]
    scores = np.array([r["score"] for r in results], dtype=float)
    mus = np.array([r["mu"] for r in results], dtype=float)
    srs = np.array([r["ann_sr"] for r in results], dtype=float)
    hi_mus = np.array([r["hi_vol_mu"] for r in results], dtype=float)
    lo_mus = np.array([r["lo_vol_mu"] for r in results], dtype=float)

    robust_objective = 0.7 * float(np.min(scores)) + 0.3 * float(np.mean(scores))

    print(f"Loaded prices: {n_days} days")
    print(
        f"Folds: {len(results)} x {TEST_DAYS}d, starts={[(r['start'], r['end']) for r in results]}"
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
        f"minScore={np.min(scores):.2f} robustObj={robust_objective:.2f}"
    )
    print(
        f"          meanMu={np.mean(mus):.1f} meanSR={np.mean(srs):.2f} "
        f"hiVolMu={np.mean(hi_mus):.1f} loVolMu={np.mean(lo_mus):.1f}"
    )

    # Conservative gate tuned for this repo history.
    ship_ready = (
        np.min(scores) >= 550.0
        and np.median(scores) >= 575.0
        and robust_objective >= 560.0
        and np.mean(hi_mus) > 0.0
        and np.mean(lo_mus) > 0.0
    )
    print(f"SHIP_READY={ship_ready}")


if __name__ == "__main__":
    main()
