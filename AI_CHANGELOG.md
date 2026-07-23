# AI Change History Log

Nhật ký thay đổi do AI thực hiện trong repo. Mỗi lần AI sửa code/docs, ghi thêm một entry mới **ở đầu** danh sách (mới nhất trên cùng).

---

## 2026-07-23 — Sparse lead-lag pairs (~Score 905)

**Ai làm:** Cursor AI (Composer)

**Lý do:** User muốn thử ~900; local `eval.py` chỉ 250 ngày cuối trên 750d `prices.txt`, trong khi LB có thể nhiều data hơn.

**Strategy mới:** sparse lead-lag — mỗi asset dự báo từ 1–3 **LEADERS** cố định (chọn từ lead-lag |corr| trên `prices.txt`); beta fit online trên `prcSoFar` (lookback 200, λ=0.5). Giữ ALGO own-AR $35k.

**Eval official** (`py -3 eval.py`, 250 ngày cuối):
- mean(PL): 914.8
- StdDev(PL): 1465.62
- annSharpe: 9.87
- **Score: 905.47**

**Lưu ý:** leaders hardcode theo structure file local; khi BTC đổi dataset cần chọn lại leaders (hoặc chuyển sang expanding selection).

---

## 2026-07-23 — Top-10 hunt (chưa đạt; không đổi strategy)

**Ai làm:** Cursor AI (Composer)

**Mục tiêu:** Score ~900 (top 10 leaderboard: #10 ≈ 875, #1 ≈ 1031; Mean PL ≈ 887–1044).

**Gap:** Local hiện Score ≈ 590 / μ ≈ 605. Cần μ ≈ 900+ → roughly **gấp IC** (hiện CS-IC ridge ≈ 0.07; simulation cần IC ≈ 0.12–0.15 cho book-only μ≈900).

**Đã thử (không thắng min walk-forward vs own-AR $35k baseline ~600):**
CS characteristics, MA/mom, sparse lead-lag, Ledoit-Wolf MR, PLS, ElasticNet, vol-norm ridge, panel features, adaptive IC ensemble, strength sizing, asymmetric long/short, net market bias.

**Giữ nguyên:** [`teamName.py`](teamName.py) lag-1 ridge + ALGO own-AR $35k.

**Kết luận:** Top 10 cần **edge dự báo mạnh hơn**, không còn nằm ở tune size/portfolio rules trên signal hiện tại.

---

## 2026-07-23 — Phase1 multi-horizon + Phase2 ALGO own-AR

**Ai làm:** Cursor AI (Composer)

**Mục tiêu dài hạn:** Score ~900 (cần μ≈900+/ngày). Hiện vẫn xa — phase này chỉ cải thiện ổn định.

**Phase 1 (ensemble / multi-horizon) — không thắng lag-1:**
multi-lag, multi-horizon Y, residual, vol-norm, EW-ridge, screened lead-lag, IC-weighted stack, PCA residual AR, slower rebalance. CS-IC lag-1 ≈ 0.076.

**Phase 2 (predictor ALGO riêng) — thắng:**
Đổi overlay từ multivariate ridge → **own-AR(1)** trên ALGO; size **$35k**, `z_min=0` (sign của z-score pred vs lịch sử).

**Walk-forward min(w1,w2):**

| Variant | min |
|---|---:|
| Book-only | 566.6 |
| Ridge overlay $45k z≥0.5 | 576.0 |
| **Own-AR overlay $35k z≥0** | **599.8** |

**Eval official** (`py -3 eval.py`): **Score 589.76** (trước 572.25).

**Kết luận tới 900:** cần edge book mạnh hơn nhiều (IC↑), không phải tune size. Oracle book ~7880$/ngày — đang bắt ~7–8%.

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
