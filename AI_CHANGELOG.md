# AI Change History Log

Nhật ký thay đổi do AI thực hiện trong repo. Mỗi lần AI sửa code/docs, ghi thêm một entry mới **ở đầu** danh sách (mới nhất trên cùng).

---

## 2026-07-23 — ALGO overlay $45k (tối ưu walk-forward)

**Ai làm:** Cursor AI (Composer)

**Mục đích:** Tăng Score ổn định trên **cả hai** cửa sổ 250 ngày (251–500 và 501–750), không chỉ 250 ngày cuối.

**Đã thử (thua baseline hoặc kém ổn định):** tune `TOP_K`/`λ`/`MIN_TRAIN`, soft sizing, deadband, position blend, signal EMA, rolling window, residual ridge, vol targeting, ALGO full $100k.

**Chọn:** giữ book ridge top/bottom-25; overlay ALGO **$45k** khi `|z| ≥ 0.5` (pred z-score vs lịch sử).

**Walk-forward (`prices.txt` 750 ngày):**

| Variant | w1 (251–500) | w2 (501–750) | min |
|---|---:|---:|---:|
| Book-only | 612.6 | 566.6 | 566.6 |
| **ALGO $45k @ z≥0.5** | **581.1** | **578.9** | **578.9** |

**Eval official** (`py -3 eval.py`, 250 ngày cuối):
- mean(PL): 588.9 → **Score: 572.25** (trước đó 566.64)

**File:** [`teamName.py`](teamName.py) — `USE_ALGO_OVERLAY=True`, `ALGO_TRADE_CAP=45000`, `ALGO_Z_MIN=0.5`.

---

## 2026-07-23 — Tích hợp ridge ensemble (~Score 567) vào teamName.py

**Ai làm:** Cursor AI (Composer)

**Mục đích:** Đưa strategy teammate từ `newfile.py` vào `teamName.py` để `eval.py` chạy đúng bản ridge ensemble thay vì boilerplate momentum.

**Thay đổi:**
- Thay toàn bộ [`teamName.py`](teamName.py) bằng logic lag-1 cross-sectional ridge ensemble (long/short top-25, cap $10k, `USE_ALGO_PRIMARY=False`).
- Thêm [`newfile.py`](newfile.py) làm bản tham chiếu của teammate.
- Thêm file nhật ký này (`AI_CHANGELOG.md`).

**Kết quả eval local** (`py -3 eval.py`, 250 ngày cuối của `prices.txt`):
- mean(PL): 582.3
- StdDev(PL): 1530.64
- annSharpe(PL): 6.02
- **Score: 566.64**

**Không làm:** bật ALGO primary, tune hyperparams, submit lên leaderboard.
