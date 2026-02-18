# scripts/simulate_single_position_engine.py
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd


# =========================
# IO helpers
# =========================
def read_table(parq: str, csv: str) -> pd.DataFrame:
    p = Path(parq)
    c = Path(csv)
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def _pick_default_out_tag(picks_path: str, tag: str | None, suffix: str | None) -> tuple[str, str]:
    """
    Try to infer tag/suffix from picks filename. Fallback to passed values.
    """
    if tag and suffix:
        return tag, suffix

    name = Path(picks_path).name
    # e.g. picks_pt10_h40_sl10_ex30_gate_none_t0p20_q0p75_rutility.csv
    m = re.search(r"picks_(pt\d+_h\d+_sl\d+_ex\d+)_gate_(.+)\.(csv|parquet)$", name)
    if m:
        inferred_tag = m.group(1)
        inferred_suffix = m.group(2)
        return (tag or inferred_tag), (suffix or inferred_suffix)

    inferred_tag = tag or "run"
    inferred_suffix = suffix or Path(picks_path).stem.replace("picks_", "")
    return inferred_tag, inferred_suffix


def _parse_bool(x: str | bool) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "on")


# =========================
# Leverage cap
# =========================
def clamp_total_by_leverage(seed: float, entry_seed: float, desired_total: float, max_leverage_pct: float) -> float:
    """
    Enforce: seed_after >= - entry_seed * max_leverage_pct  (entry_seed = S0 at cycle entry)
    desired_total : total amount we'd like to spend today across ALL legs
    returns : allowed total spend (>=0), possibly reduced.
    """
    if desired_total <= 0:
        return 0.0
    if not np.isfinite(entry_seed) or entry_seed <= 0:
        # safest: no borrowing if entry seed invalid
        return float(min(desired_total, max(seed, 0.0)))

    borrow_limit = float(entry_seed) * float(max_leverage_pct)
    floor_seed = -borrow_limit
    room = float(seed) - float(floor_seed)  # seed + borrow_limit
    if room <= 0:
        return 0.0
    return float(min(desired_total, room))


# =========================
# Strategy state
# =========================
@dataclass
class Leg:
    ticker: str
    shares: float = 0.0
    invested: float = 0.0

    entry_date: pd.Timestamp | None = None
    holding_days: int = 0

    extending: bool = False

    # trailing / partial TP
    tp1_done: bool = False
    trail_peak: float = 0.0  # peak since tp1 (or entry if you want)
    last_reason: str = ""

    def avg_price(self) -> float:
        return (self.invested / self.shares) if (self.shares > 0 and self.invested > 0) else np.nan

    def value(self, close_px: float | None) -> float:
        if close_px is None or not np.isfinite(close_px) or close_px <= 0:
            return 0.0
        return float(self.shares) * float(close_px)


@dataclass
class Cycle:
    in_cycle: bool = False
    entry_seed: float = 0.0    # S0 at entry
    unit_total: float = 0.0    # S0 / max_days
    entry_date: pd.Timestamp | None = None

    legs: list[Leg] = field(default_factory=list)

    # cash balance can go negative
    seed: float = 0.0

    # tracking
    max_leverage_pct: float = 0.0  # max(-seed)/entry_seed observed during cycle
    max_equity: float = 0.0
    max_dd: float = 0.0

    # totals for cycle-level accounting
    buys_total: float = 0.0
    sells_total: float = 0.0

    def equity(self, prices_close: dict[str, float]) -> float:
        pos_val = 0.0
        for leg in self.legs:
            px = prices_close.get(leg.ticker)
            pos_val += leg.value(px)
        return float(self.seed) + float(pos_val)

    def update_drawdown(self, eq: float) -> None:
        if eq > self.max_equity:
            self.max_equity = eq
        if self.max_equity > 0:
            dd = (eq - self.max_equity) / self.max_equity
            if dd < self.max_dd:
                self.max_dd = dd

    def update_max_leverage(self, max_leverage_pct_cap: float) -> None:
        if self.entry_seed <= 0:
            return
        lev = max(0.0, -float(self.seed)) / float(self.entry_seed)
        if lev > self.max_leverage_pct:
            self.max_leverage_pct = float(lev)
        if self.max_leverage_pct > max_leverage_pct_cap + 1e-9:
            self.max_leverage_pct = float(max_leverage_pct_cap)


# =========================
# picks handling
# =========================
def load_picks(picks_path: str) -> pd.DataFrame:
    p = Path(picks_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing picks file: {p}")
    if picks_path.lower().endswith(".parquet"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError(f"picks must have Date/Ticker. cols={list(df.columns)[:50]}")
    df = df.copy()
    df["Date"] = _norm_date(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["Date", "Ticker"]).sort_values(["Date"]).reset_index(drop=True)
    return df


def select_topk_for_date(picks: pd.DataFrame, date: pd.Timestamp, k: int) -> list[str]:
    """
    Picks can be:
    - 1 row per Date (typical top-1): then this returns [Ticker]
    - multiple rows per Date (if you later extend predict_gate): then returns top-k in file order
      (or by a score column if present).
    """
    sub = picks[picks["Date"] == date].copy()
    if sub.empty:
        return []

    # if score columns exist, sort desc to pick top-k more sensibly
    for score_col in ["utility", "ret_score", "p_success", "EV"]:
        if score_col in sub.columns:
            sub[score_col] = pd.to_numeric(sub[score_col], errors="coerce").fillna(0.0)
            sub = sub.sort_values(score_col, ascending=False)
            break

    tickers = sub["Ticker"].astype(str).str.upper().tolist()
    # unique preserving order
    out = []
    seen = set()
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
        if len(out) >= k:
            break
    return out


def parse_weights(k: int, weights_spec: str) -> list[float]:
    """
    weights_spec examples:
      "1.0"
      "0.7,0.3"
      "0.6,0.4"
    If malformed, fallback to equal weights.
    """
    try:
        parts = [p.strip() for p in str(weights_spec).split(",") if p.strip()]
        w = [float(x) for x in parts]
        if len(w) != k or any(not np.isfinite(x) or x <= 0 for x in w):
            raise ValueError("bad weights")
        s = sum(w)
        return [x / s for x in w]
    except Exception:
        return [1.0 / k] * k


# =========================
# main simulation
# =========================
def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-leg (TopK) cycle engine with leverage cap + optional trailing exits.")

    ap.add_argument("--picks-path", required=True, type=str, help="CSV/Parquet with Date,Ticker (can be multiple per date).")
    ap.add_argument("--prices-parq", default="data/raw/prices.parquet", type=str)
    ap.add_argument("--prices-csv", default="data/raw/prices.csv", type=str)

    ap.add_argument("--initial-seed", default=40_000_000, type=float)

    ap.add_argument("--profit-target", required=True, type=float)   # e.g. 0.10
    ap.add_argument("--max-days", required=True, type=int)          # e.g. 40
    ap.add_argument("--stop-level", required=True, type=float)      # e.g. -0.10
    ap.add_argument("--max-extend-days", required=True, type=int)   # tag compatibility (no hard limit)

    ap.add_argument("--max-leverage-pct", default=1.0, type=float)  # 1.0 => 100% (max total spend = 2x S0)

    # top-k configs
    ap.add_argument("--topk", default=1, type=int, help="1 or 2 recommended.")
    ap.add_argument("--topk-weights", default="1.0", type=str, help="e.g. '1.0' or '0.7,0.3'")

    # trailing / partial TP
    ap.add_argument("--enable-trailing", default="true", type=str)
    ap.add_argument("--tp1-frac", default=0.50, type=float, help="fraction sold at profit target (0~1).")
    ap.add_argument("--trail-stop", default=0.10, type=float, help="trailing stop fraction, e.g. 0.10 => -10% from peak.")

    ap.add_argument("--tag", default="", type=str)
    ap.add_argument("--suffix", default="", type=str)
    ap.add_argument("--out-dir", default="data/signals", type=str)

    args = ap.parse_args()

    topk = int(args.topk)
    if topk < 1:
        topk = 1
    if topk > 2:
        # safety: keep it manageable
        topk = 2

    w = parse_weights(topk, args.topk_weights)
    enable_trailing = _parse_bool(args.enable_trailing)
    tp1_frac = float(args.tp1_frac)
    tp1_frac = max(0.0, min(1.0, tp1_frac))
    trail_stop = float(args.trail_stop)
    trail_stop = max(0.0, min(0.99, trail_stop))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag, suffix = _pick_default_out_tag(args.picks_path, args.tag or None, args.suffix or None)

    # load picks
    picks = load_picks(args.picks_path)

    # load prices
    prices = read_table(args.prices_parq, args.prices_csv).copy()
    if "Date" not in prices.columns or "Ticker" not in prices.columns:
        raise ValueError(f"prices must have Date/Ticker. cols={list(prices.columns)[:50]}")
    for c in ["Open", "High", "Low", "Close"]:
        if c not in prices.columns:
            raise ValueError(f"prices missing {c}")
    prices["Date"] = _norm_date(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    prices = prices.dropna(subset=["Date", "Ticker", "Close"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    grouped = prices.groupby("Date", sort=True)

    cycle = Cycle(
        in_cycle=False,
        seed=float(args.initial_seed),
        max_equity=float(args.initial_seed),
        max_dd=0.0,
    )

    cooldown_today = False
    curve = []
    trades = []

    def close_cycle(exit_date: pd.Timestamp, reason: str) -> None:
        """
        Close cycle only when all legs are closed already (or being forced).
        Here we assume legs are empty.
        """
        nonlocal cooldown_today, cycle, trades

        invested_total = float(cycle.buys_total)
        proceeds_total = float(cycle.sells_total)
        cycle_return = (proceeds_total - invested_total) / invested_total if invested_total > 0 else np.nan
        win = int(cycle_return > 0) if np.isfinite(cycle_return) else 0

        tickers = ",".join([leg.ticker for leg in cycle.legs])  # should be empty; kept for safety

        trades.append({
            "EntryDate": cycle.entry_date,
            "ExitDate": exit_date,
            "Ticker": tickers,
            "Tickers": tickers,
            "EntrySeed": cycle.entry_seed,
            "ProfitTarget": float(args.profit_target),
            "MaxDays": int(args.max_days),
            "StopLevel": float(args.stop_level),
            "MaxExtendDaysParam": int(args.max_extend_days),
            "TopK": int(topk),
            "TopKWeights": ",".join([f"{x:.6f}" for x in w]),
            "EnableTrailing": int(enable_trailing),
            "TP1Frac": float(tp1_frac),
            "TrailStop": float(trail_stop),

            "MaxLeveragePctCap": float(args.max_leverage_pct),
            "MaxLeveragePct": float(cycle.max_leverage_pct),

            "Invested": invested_total,
            "Proceeds": proceeds_total,
            "CycleReturn": cycle_return,
            "Win": win,

            "HoldingDays": int(max_hold_days_in_cycle),
            "Reason": reason,
            "MaxDrawdown": float(cycle.max_dd),
            "FinalSeed": float(cycle.seed),
        })

        # reset cycle state (seed already updated during sells)
        cycle.in_cycle = False
        cycle.entry_seed = 0.0
        cycle.unit_total = 0.0
        cycle.entry_date = None
        cycle.legs = []
        cycle.max_leverage_pct = 0.0
        cycle.buys_total = 0.0
        cycle.sells_total = 0.0

        cooldown_today = True

    # We need this in close_cycle; keep updated during loop
    max_hold_days_in_cycle = 0

    # ---------------------------
    # simulate day-by-day
    # ---------------------------
    for date, day_df in grouped:
        day_df = day_df.set_index("Ticker", drop=False)
        cooldown_today = False

        prices_close_today = {}

        # build close map for equity calc quickly
        for t, row in day_df.iterrows():
            try:
                prices_close_today[str(t)] = float(row["Close"])
            except Exception:
                continue

        # =======================
        # If in cycle: manage legs
        # =======================
        if cycle.in_cycle:
            # 1) advance holding days + evaluate exits for each leg
            legs_still_open: list[Leg] = []

            for leg in cycle.legs:
                if leg.ticker not in day_df.index:
                    # missing data today -> just keep open, no action
                    legs_still_open.append(leg)
                    continue

                row = day_df.loc[leg.ticker]
                close_px = float(row["Close"])
                high_px = float(row["High"])
                low_px = float(row["Low"])

                leg.holding_days += 1
                max_hold_days_in_cycle = max(max_hold_days_in_cycle, leg.holding_days)

                avg = leg.avg_price()

                # --- trailing / partial TP logic
                if enable_trailing:
                    # TP1 not done: if hit profit target intraday, sell tp1_frac at PT price
                    if (not leg.tp1_done) and np.isfinite(avg) and avg > 0 and np.isfinite(high_px):
                        pt_px = avg * (1.0 + float(args.profit_target))
                        if high_px >= pt_px and leg.shares > 0:
                            sell_shares = leg.shares * float(tp1_frac)
                            sell_shares = max(0.0, min(leg.shares, sell_shares))
                            if sell_shares > 0:
                                proceeds = sell_shares * pt_px
                                leg.shares -= sell_shares
                                cycle.seed += proceeds
                                cycle.sells_total += proceeds
                                leg.last_reason = "TP1"
                                leg.tp1_done = True
                                # start trailing peak at today's high (or pt_px)
                                leg.trail_peak = max(float(high_px), float(pt_px))

                    # TP1 done: trailing stop for remaining shares
                    if leg.tp1_done and leg.shares > 0:
                        # update peak
                        if np.isfinite(high_px):
                            leg.trail_peak = max(float(leg.trail_peak), float(high_px))

                        stop_px = float(leg.trail_peak) * (1.0 - float(trail_stop))
                        # If intraday low breaks stop, assume stop filled at stop_px
                        if np.isfinite(low_px) and low_px <= stop_px:
                            proceeds = leg.shares * stop_px
                            cycle.seed += proceeds
                            cycle.sells_total += proceeds
                            leg.shares = 0.0
                            leg.last_reason = "TRAIL_STOP"

                    # If fully sold, leg closes
                    if leg.shares <= 1e-12:
                        continue  # leg closed

                # --- Non-trailing or still pre-TP1: max-days / extend logic + DCA
                # If trailing is enabled AND TP1 already done, we DO NOT DCA (risk control)
                allow_dca = (not enable_trailing) or (enable_trailing and (not leg.tp1_done))

                # --- full take profit if trailing disabled
                if (not enable_trailing) and np.isfinite(avg) and avg > 0 and np.isfinite(high_px) and leg.shares > 0:
                    pt_px = avg * (1.0 + float(args.profit_target))
                    if high_px >= pt_px:
                        proceeds = leg.shares * pt_px
                        cycle.seed += proceeds
                        cycle.sells_total += proceeds
                        leg.shares = 0.0
                        leg.last_reason = "TP_FULL"
                        continue  # closed

                # --- reach max_days (only if not already in extending AND still open AND pre-TP1 if trailing)
                if leg.shares > 0 and (not leg.extending) and allow_dca and leg.holding_days >= int(args.max_days):
                    cur_ret = (close_px - avg) / avg if np.isfinite(avg) and avg != 0 else -np.inf
                    if cur_ret >= float(args.stop_level):
                        # close at close
                        proceeds = leg.shares * close_px
                        cycle.seed += proceeds
                        cycle.sells_total += proceeds
                        leg.shares = 0.0
                        leg.last_reason = "MAXDAY_CLOSE"
                        continue
                    else:
                        leg.extending = True

                # --- extending zone: exit on recovery to stop-level; else DCA (capped)
                if leg.shares > 0 and leg.extending and allow_dca and np.isfinite(avg) and avg > 0:
                    rec_px = avg * (1.0 + float(args.stop_level))
                    if np.isfinite(high_px) and high_px >= rec_px:
                        proceeds = leg.shares * rec_px
                        cycle.seed += proceeds
                        cycle.sells_total += proceeds
                        leg.shares = 0.0
                        leg.last_reason = "EXT_RECOVERY"
                        continue

                legs_still_open.append(leg)

            # remove closed legs
            cycle.legs = legs_still_open

            # 2) DCA buys (after exits), with leverage cap applied to total spends
            if cycle.in_cycle and len(cycle.legs) > 0:
                # compute desired per leg
                desired_map: dict[str, float] = {}

                for i, leg in enumerate(cycle.legs):
                    if leg.ticker not in day_df.index:
                        continue
                    row = day_df.loc[leg.ticker]
                    close_px = float(row["Close"])
                    avg = leg.avg_price()

                    # if trailing enabled and TP1 done -> no more buys
                    if enable_trailing and leg.tp1_done:
                        continue

                    # desired unit per leg = unit_total * weight_i (stable)
                    # If topk changes due to missing leg, still use original weights for existing legs by index.
                    w_i = w[i] if i < len(w) else (1.0 / max(1, len(cycle.legs)))
                    unit_leg = float(cycle.unit_total) * float(w_i)

                    desired = 0.0
                    if leg.extending:
                        desired = unit_leg
                    else:
                        if np.isfinite(avg) and np.isfinite(close_px) and close_px > 0 and avg > 0:
                            if close_px <= avg:
                                desired = unit_leg
                            elif close_px <= avg * 1.05:
                                desired = unit_leg / 2.0
                            else:
                                desired = 0.0

                    if desired > 0:
                        desired_map[leg.ticker] = float(desired)

                desired_total = float(sum(desired_map.values()))
                allowed_total = clamp_total_by_leverage(
                    seed=float(cycle.seed),
                    entry_seed=float(cycle.entry_seed),
                    desired_total=desired_total,
                    max_leverage_pct=float(args.max_leverage_pct),
                )

                if allowed_total > 0 and desired_total > 0:
                    scale = allowed_total / desired_total
                    for i, leg in enumerate(cycle.legs):
                        if leg.ticker not in desired_map:
                            continue
                        if leg.ticker not in day_df.index:
                            continue
                        close_px = float(day_df.loc[leg.ticker]["Close"])
                        if not (np.isfinite(close_px) and close_px > 0):
                            continue

                        invest = float(desired_map[leg.ticker]) * float(scale)
                        if invest <= 0:
                            continue

                        # execute buy
                        cycle.seed -= invest
                        cycle.buys_total += invest
                        leg.invested += invest
                        leg.shares += invest / close_px

                    cycle.update_max_leverage(float(args.max_leverage_pct))

            # 3) if all legs closed -> end cycle
            if cycle.in_cycle and len(cycle.legs) == 0:
                cycle.in_cycle = False
                # cycle.seed already includes proceeds
                # record trade row
                close_cycle(date, reason="ALL_LEGS_CLOSED")

        # =======================
        # If NOT in cycle: enter
        # =======================
        if (not cycle.in_cycle) and (not cooldown_today):
            tickers = select_topk_for_date(picks, date, topk)
            # require prices exist today
            tickers = [t for t in tickers if t in day_df.index]

            if len(tickers) > 0:
                # entry seed = current seed at entry
                S0 = float(cycle.seed)
                unit_total = (S0 / float(args.max_days)) if args.max_days > 0 else 0.0
                if unit_total <= 0:
                    # cannot enter if no unit budget
                    pass
                else:
                    # desired buys for entry day per ticker
                    desired_map = {}
                    for i, t in enumerate(tickers):
                        w_i = w[i] if i < len(w) else (1.0 / len(tickers))
                        desired_map[t] = float(unit_total) * float(w_i)

                    desired_total = float(sum(desired_map.values()))
                    allowed_total = clamp_total_by_leverage(
                        seed=float(cycle.seed),
                        entry_seed=float(S0),
                        desired_total=desired_total,
                        max_leverage_pct=float(args.max_leverage_pct),
                    )

                    if allowed_total > 0 and desired_total > 0:
                        scale = allowed_total / desired_total

                        new_legs: list[Leg] = []
                        for t in tickers:
                            close_px = float(day_df.loc[t]["Close"])
                            if not (np.isfinite(close_px) and close_px > 0):
                                continue
                            invest = float(desired_map[t]) * float(scale)
                            if invest <= 0:
                                continue

                            cycle.seed -= invest
                            cycle.buys_total += invest

                            leg = Leg(ticker=t, shares=invest / close_px, invested=invest, entry_date=date, holding_days=1)
                            leg.tp1_done = False
                            leg.trail_peak = float(close_px)  # start
                            new_legs.append(leg)

                        if len(new_legs) > 0:
                            cycle.in_cycle = True
                            cycle.entry_seed = float(S0)
                            cycle.unit_total = float(unit_total)
                            cycle.entry_date = date
                            cycle.legs = new_legs
                            cycle.max_leverage_pct = 0.0
                            cycle.update_max_leverage(float(args.max_leverage_pct))
                            max_hold_days_in_cycle = 1

        # =======================
        # record curve (daily)
        # =======================
        eq = cycle.equity(prices_close_today)
        cycle.update_drawdown(eq)

        # for reporting, store current tickers
        tickers_now = ",".join([leg.ticker for leg in cycle.legs]) if cycle.in_cycle else ""

        curve.append({
            "Date": date,
            "Equity": float(eq),
            "Seed": float(cycle.seed),
            "InCycle": int(cycle.in_cycle),
            "Tickers": tickers_now,
            "Positions": int(len(cycle.legs)) if cycle.in_cycle else 0,
            "EntrySeed": float(cycle.entry_seed) if cycle.in_cycle else np.nan,
            "UnitTotal": float(cycle.unit_total) if cycle.in_cycle else np.nan,
            "BuysTotal": float(cycle.buys_total),
            "SellsTotal": float(cycle.sells_total),
            "MaxLeveragePctCycle": float(cycle.max_leverage_pct) if cycle.in_cycle else 0.0,
            "MaxDrawdownPortfolio": float(cycle.max_dd),
        })

    # finalize dataframes
    trades_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(curve)

    if not curve_df.empty:
        curve_df["SeedMultiple"] = curve_df["Equity"] / float(args.initial_seed)

    # save
    trades_path = out_dir / f"sim_engine_trades_{tag}_gate_{suffix}.parquet"
    curve_path = out_dir / f"sim_engine_curve_{tag}_gate_{suffix}.parquet"

    trades_df.to_parquet(trades_path, index=False)
    curve_df.to_parquet(curve_path, index=False)

    print(f"[DONE] wrote trades: {trades_path} rows={len(trades_df)}")
    print(f"[DONE] wrote curve : {curve_path} rows={len(curve_df)}")

    if not curve_df.empty:
        final_mult = float(curve_df["SeedMultiple"].iloc[-1])
        print(f"[INFO] final SeedMultiple={final_mult:.4f} maxDD={float(cycle.max_dd):.4f}")


if __name__ == "__main__":
    main()