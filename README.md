# 📈 Algothon 2026 — Team USUK Trading Strategy

Our team's strategy for the **Susquehanna x UNSW FinTech Society Algothon 2026**, Australia's student-led algorithmic trading hackathon. This repository is a fork of the [official starter code](https://github.com/UNSW-FinTech-Society-IT/algothon26-starter-code), extended with our strategy, validation tooling and a full experiment log.

## 🧠 Approach

The strategy was developed through **AI-assisted iteration (vibe coding)** with Cursor. Every experiment, whether it shipped or not, is recorded in [AI_CHANGELOG.md](AI_CHANGELOG.md) with the idea, the results and the decision. This made the process transparent and easy for teammates to review.

Final strategy, in plain terms:

- **Cross-sectional ridge regression ensemble** — predicts next-day moves for each instrument, then goes long the top 25 and short the bottom 25, with a fixed dollar cap per position
- **Meta-ensemble** — blends a dynamic (rolling) model with a static model, plus a small rank-based blend to reduce unnecessary trading
- **Special-instrument overlay** — a separate, simpler model for the one instrument with lower commission and a higher position limit

## 🛡️ Validating Against Overfitting

An early version scored about **905** locally but only around **300** on the hidden leaderboard data, because it relied on relationships hand-picked from the public dataset.

To fix this, I built [`wf_gate.py`](wf_gate.py), a **walk-forward validation gate**:

- runs the strategy on one continuous path, exactly like the competition
- scores six chronological folds separately
- only marks a strategy `SHIP_READY` if it passes minimum, median and latest-fold thresholds

The lesson: a lower but **consistent** score across time periods is worth more than a high score on one window.

## 📊 Results (local evaluation)

| Version | Local score | Notes |
| --- | ---: | --- |
| Teammate's ridge ensemble (starting point) | ~567 | Integrated into `teamName.py` |
| Hand-picked lead-lag pairs | ~905 | Overfit; around 300 on hidden data, not shipped |
| Causal ridge + special-instrument overlay | ~573 | Robust baseline |
| **Meta-ensemble (shipped)** | **~590** | Passes the walk-forward gate |

## 🗂️ What's in This Repo

| File | Purpose |
| :--- | :--- |
| `teamName.py` | Our strategy, implementing `getMyPosition(prcSoFar)` |
| `USUK.py` | Submission copy of the strategy under our team name |
| `newfile.py` | Teammate's original ridge ensemble, kept for reference |
| `wf_gate.py` | Walk-forward anti-overfitting gate |
| `AI_CHANGELOG.md` | Experiment log of every AI-assisted change (Vietnamese) |
| `eval.py` | Official evaluation script — scores the last 250 days of `prices.txt` |
| `prices.txt` | Competition price data |
| `requirements-dev.txt` | Package set available at grading time |

## 🚀 Running It

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
pip install -r requirements-dev.txt

python eval.py      # official local backtest
python wf_gate.py   # walk-forward validation gate
```

## 🛠️ Skills Demonstrated

`Python` `NumPy` `pandas` `Ridge Regression` `Walk-Forward Validation` `Experiment Tracking` `AI-Assisted Development`

## 👩‍💻 Author

**Elsa Ngoc Dang** · [LinkedIn](https://www.linkedin.com/in/ha-bao-ngoc-dang/) · [GitHub](https://github.com/ElsaNgoc)

---

<details>
<summary><b>Original starter code README</b></summary>

Starter code for the Susquehanna x UNSW FinTech Society Algothon 2026 - the seventh year of Australia's first student-led algorithmic trading hackathon.

Full rules, scoring, schedule, and submission details live on the **[Algothon 2026 Wiki](https://wiki.algothon.au/)** - this README only covers what's in this repo and how to run it. If anything here ever seems to disagree with the wiki, the wiki is correct.

1. Implement `getMyPosition(prcSoFar)` in `teamName.py`. No need to rename anything or touch `eval.py` while developing - `eval.py` imports from `teamName` by default.
2. Run `python eval.py` to backtest locally.

Only when you're ready to submit: copy `teamName.py` to `<YourTeamName>.py` (matching your registered team name) and zip it up - `eval.py` and `prices.txt` are not part of the submission, only your algorithm file (and `requirements.txt`, if you used extra packages). See the [Submission Guide](https://wiki.algothon.au/submission/) for exact packaging requirements. Submit through the [live leaderboard](https://www.algothon.au/leaderboard).

</details>
