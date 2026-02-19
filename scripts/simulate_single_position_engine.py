# scripts/simulate_single_position_engine.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


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


def clamp_invest_by_leverage(seed: float, entry_seed: float, desired: float, max_leverage_pct: float) -> float:
    """
    Enforce: seed_after >= - entry_seed * max_leverage_pct
    desired : amount we'd like to spend today (>=0)
    """
    if desired <= 0:
        return 0.0
    if not np.isfinite(entry_seed) or entry_seed <= 0:
        return float(min(desired, max(seed, 0.0)))

    borrow_limit = float(entry_seed) * float(max_leverage_pct)
    floor_seed = -borrow_limit
    room = float(seed) - float(floor_seed)  # seed + borrow_limit
    if room <= 0:
        return 0.0
    return float(min(desired, room))


@dataclass
class Leg:
    ticker: str
    weight: float

    shares: float = 0.0
    invested: float = 0.0

    tp1_done: bool = False
    peak: float = 0.0  # for trailing after TP1

    def avg_price(self) -> float:
        return (self.invested / self.shares) if (self.shares > 0 and self.invested > 0) else np.nan

    def value(self, close_px: float) -> float:
        if not np.isfinite(close_px) or close_px <= 0:
            return 0.0
        return float(self.shares) * float(close_px)


@dataclass
class CycleState:
    in_cycle: bool = False
    entry_date: pd.Timestamp | None = None

    seed: float = 0.0         # cash, can go negative
    entry_seed: float = 0.0   # S0 at entry
    unit: float = 0.0         # daily buy budget = entry_seed / max_days
    holding_days: int = 0
    extending: bool = False

    max_leverage_pct: float = 0.0
    max_equity: float = 0.0
    max_dd: float = 0.0

    legs: list[Leg] = None  # type: ignore

    def equity(self, prices: dict[str, float]) -> float:
        v = 0.0
        if self.legs:
            for leg in self.legs:
                px = prices.get(leg.ticker, np.nan)
                if np.isfinite(px):
                    v += leg.value(float(px))
        return float(self.seed) + float(v)

    def update_dd(self, prices: dict[str, float]) -> None:
        eq = self.equity(prices)
        if eq > self.max_equity:
            self.max_equity = eq
        if self.max_equity > 0:
            dd = (eq - self.max_equity) / self.max_equity
            if dd < self.max_dd:
                self.max_dd = dd

    def update_lev(self, max_cap: float) -> None:
        if self.entry_seed <= 0:
            return
        lev = max(0.0, -self.seed) / self.entry_seed
        if lev > self.max_leverage_pct:
            self.max_leverage_pct = float(lev)
        if self.max_leverage_pct > max_cap + 1e-9:
            self.max_leverage_pct = float(max_cap)


def parse_weights(weights: str, topk: int) -> list[float]:
    parts = [p.strip() for p in str(weights).split(",") if p.strip()]
    ws = [float(p) for p in parts]
    if len(ws) != topk:
        raise ValueError(f"--weights must have {topk} numbers (got {len(ws)}): {weights}")
    s = sum(ws)
    if s <= 0:
        raise ValueError("--weights sum must be > 0")
    return [w / s for w in ws]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Single-cycle engine with TopK (1~2), TP1 partial, trailing stop, leverage cap on ALL buys.\n"
            "PLUS: SUB cycle (top1) funded by MAIN TP1 proceeds pool, with max_days_sub = max_days/2, trailing enabled.\n"
            "SUB entry uses SubOK==1 from picks if present (anti-chasing filter)."
        )
    )
    ap.add_argument("--picks-path", required=True, type=str, help="CSV with columns Date,Ticker (TopK rows/day).")
    ap.add_argument("--prices-parq", default="data/raw/prices.parquet", type=str)
    ap.add_argument("--prices-csv", default="data/raw/prices.csv", type=str)

    ap.add_argument("--initial-seed", default=40_000_000, type=float)

    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    ap.add_argument("--max-leverage-pct", default=1.0, type=float)

    ap.add_argument("--enable-trailing", default="true", type=str)
    ap.add_argument("--tp1-frac", default=0.50, type=float)
    ap.add_argument("--trail-stop", default=0.10, type=float)

    ap.add_argument("--topk", default=1, type=int)
    ap.add_argument("--weights", default="1.0", type=str)

    ap.add_argument("--tag", default="", type=str)
    ap.add_argument("--suffix", default="", type=str)
    ap.add_argument("--out-dir", default="data/signals", type=str)

    args = ap.parse_args()

    enable_trailing = str(args.enable_trailing).lower() in ("1", "true", "yes", "y")
    topk = int(args.topk)
    if topk < 1 or topk > 2:
        raise ValueError("--topk should be 1 or 2")
    weights = parse_weights(args.weights, topk=topk)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- picks
    picks_path = Path(args.picks_path)
    if not picks_path.exists():
        raise FileNotFoundError(f"Missing picks file: {picks_path}")

    picks = pd.read_csv(picks_path) if picks_path.suffix.lower() != ".parquet" else pd.read_parquet(picks_path)
    if "Date" not in picks.columns or "Ticker" not in picks.columns:
        raise ValueError(f"picks must have Date,Ticker. cols={list(picks.columns)[:50]}")

    picks = picks.copy()
    picks["Date"] = _norm_date(picks["Date"])
    picks["Ticker"] = picks["Ticker"].astype(str).str.upper().str.strip()

    # ✅ SubOK is optional; if missing, allow SUB by default.
    if "SubOK" not in picks.columns:
        picks["SubOK"] = 1
    picks["SubOK"] = pd.to_numeric(picks["SubOK"], errors="coerce").fillna(0).astype(int)

    picks = picks.dropna(subset=["Date", "Ticker"]).sort_values(["Date"]).reset_index(drop=True)

    # keep TopK per day (in case input contains more)
    picks = picks.groupby("Date", group_keys=False).head(topk).reset_index(drop=True)

    # MAIN picks: existing behavior (TopK tickers)
    picks_by_date: dict[pd.Timestamp, list[str]] = {}
    # SUB candidates: only SubOK==1, keep original order within the day's picks
    picks_by_date_sub: dict[pd.Timestamp, list[str]] = {}

    for d, g in picks.groupby("Date"):
        picks_by_date[d] = g["Ticker"].tolist()
        g2 = g[g["SubOK"] == 1]
        picks_by_date_sub[d] = g2["Ticker"].tolist()

    # ---------- prices
    prices = read_table(args.prices_parq, args.prices_csv).copy()
    if "Date" not in prices.columns or "Ticker" not in prices.columns:
        raise ValueError("prices must have Date,Ticker")
    prices["Date"] = _norm_date(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    for c in ["Open", "High", "Low", "Close"]:
        if c not in prices.columns:
            raise ValueError(f"prices missing {c}")
    prices = prices.dropna(subset=["Date", "Ticker", "Close"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    grouped = prices.groupby("Date", sort=True)

    # ---------- states
    st = CycleState(
        seed=float(args.initial_seed),
        max_equity=float(args.initial_seed),
        max_dd=0.0,
        legs=[]
    )

    sub = CycleState(
        seed=st.seed,  # shared conceptually
        max_equity=float(args.initial_seed),
        max_dd=0.0,
        legs=[]
    )

    max_days_sub = max(1, int(args.max_days) // 2)
    sub_used_in_this_main = False

    tp1_cash_pool = 0.0
    min_tp1_cash_pool = 0.0

    def track_pool() -> None:
        nonlocal min_tp1_cash_pool
        if tp1_cash_pool < min_tp1_cash_pool:
            min_tp1_cash_pool = float(tp1_cash_pool)

    cooldown_today = False
    trades: list[dict] = []
    curve: list[dict] = []

    def close_sub_cycle(exit_date: pd.Timestamp, day_prices_close: dict[str, float], reason: str) -> None:
        nonlocal tp1_cash_pool, sub, trades
        proceeds = 0.0
        invested_total = 0.0

        for leg in sub.legs:
            px = float(day_prices_close.get(leg.ticker, np.nan))
            if not np.isfinite(px) or px <= 0:
                continue
            proceeds += leg.shares * px
            invested_total += leg.invested

        cycle_return = (proceeds - invested_total) / invested_total if invested_total > 0 else np.nan
        win = int(cycle_return > 0) if np.isfinite(cycle_return) else 0

        trades.append({
            "CycleType": "SUB",
            "EntryDate": sub.entry_date,
            "ExitDate": exit_date,
            "Tickers": ",".join([l.ticker for l in sub.legs]),
            "Weights": ",".join([f"{l.weight:.4f}" for l in sub.legs]),
            "EntrySeed": sub.entry_seed,
            "ProfitTarget": args.profit_target,
            "TP1_Frac": float(args.tp1_frac),
            "TrailStop": float(args.trail_stop) if enable_trailing else np.nan,
            "MaxDays": max_days_sub,
            "StopLevel": np.nan,
            "MaxExtendDaysParam": 0,
            "MaxLeveragePctCap": args.max_leverage_pct,
            "MaxLeveragePct": sub.max_leverage_pct,
            "Invested": invested_total,
            "Proceeds": proceeds,
            "CycleReturn": cycle_return,
            "HoldingDays": sub.holding_days,
            "Extending": 0,
            "Reason": reason,
            "MaxDrawdown": sub.max_dd,
            "Win": win,
        })

        st.seed += proceeds

        sub.in_cycle = False
        sub.entry_date = None
        sub.entry_seed = 0.0
        sub.unit = 0.0
        sub.holding_days = 0
        sub.extending = False
        sub.max_leverage_pct = 0.0
        sub.legs = []
        sub.max_equity = float(st.seed)

        # reset pool on sub exit (as agreed)
        tp1_cash_pool = 0.0
        track_pool()

    def close_main_cycle(exit_date: pd.Timestamp, day_prices_close: dict[str, float], reason: str) -> None:
        nonlocal cooldown_today, st, trades, sub_used_in_this_main, tp1_cash_pool

        # if sub running, force close it first (same day close)
        if sub.in_cycle:
            close_sub_cycle(exit_date, day_prices_close, reason="MAIN_EXIT_FORCE_CLOSE_SUB")

        proceeds = 0.0
        invested_total = 0.0

        for leg in st.legs:
            px = float(day_prices_close.get(leg.ticker, np.nan))
            if not np.isfinite(px) or px <= 0:
                continue
            proceeds += leg.shares * px
            invested_total += leg.invested

        cycle_return = (proceeds - invested_total) / invested_total if invested_total > 0 else np.nan
        win = int(cycle_return > 0) if np.isfinite(cycle_return) else 0

        trades.append({
            "CycleType": "MAIN",
            "EntryDate": st.entry_date,
            "ExitDate": exit_date,
            "Tickers": ",".join([l.ticker for l in st.legs]),
            "Weights": ",".join([f"{l.weight:.4f}" for l in st.legs]),
            "EntrySeed": st.entry_seed,
            "ProfitTarget": args.profit_target,
            "TP1_Frac": float(args.tp1_frac),
            "TrailStop": float(args.trail_stop) if enable_trailing else np.nan,
            "MaxDays": args.max_days,
            "StopLevel": args.stop_level,
            "MaxExtendDaysParam": args.max_extend_days,
            "MaxLeveragePctCap": args.max_leverage_pct,
            "MaxLeveragePct": st.max_leverage_pct,
            "Invested": invested_total,
            "Proceeds": proceeds,
            "CycleReturn": cycle_return,
            "HoldingDays": st.holding_days,
            "Extending": int(st.extending),
            "Reason": reason,
            "MaxDrawdown": st.max_dd,
            "Win": win,
        })

        st.seed += proceeds

        st.in_cycle = False
        st.entry_date = None
        st.entry_seed = 0.0
        st.unit = 0.0
        st.holding_days = 0
        st.extending = False
        st.max_leverage_pct = 0.0
        st.legs = []

        sub_used_in_this_main = False

        tp1_cash_pool = 0.0
        track_pool()

        cooldown_today = True

    # ---------- simulate
    for date, day_df in grouped:
        day_df = day_df.set_index("Ticker", drop=False)
        cooldown_today = False

        day_prices_close: dict[str, float] = {}
        day_prices_high: dict[str, float] = {}
        day_prices_low: dict[str, float] = {}

        for t in day_df.index:
            r = day_df.loc[t]
            day_prices_close[t] = float(r["Close"])
            day_prices_high[t] = float(r["High"])
            day_prices_low[t] = float(r["Low"])

        # ===== SUB update
        if sub.in_cycle:
            sub.holding_days += 1

            # TP1 + trailing
            if enable_trailing:
                for leg in sub.legs:
                    if leg.ticker not in day_df.index:
                        continue
                    high_px = day_prices_high[leg.ticker]
                    low_px = day_prices_low[leg.ticker]
                    avg = leg.avg_price()

                    if (not leg.tp1_done) and np.isfinite(avg) and high_px >= avg * (1.0 + float(args.profit_target)):
                        tp_px = avg * (1.0 + float(args.profit_target))
                        sell_shares = leg.shares * float(args.tp1_frac)
                        sell_shares = float(min(leg.shares, max(0.0, sell_shares)))
                        proceeds = sell_shares * tp_px

                        leg.shares -= sell_shares
                        leg.invested *= (leg.shares / (leg.shares + sell_shares)) if (leg.shares + sell_shares) > 0 else 0.0

                        st.seed += proceeds
                        leg.tp1_done = True
                        leg.peak = float(max(high_px, tp_px))
                        sub.update_lev(float(args.max_leverage_pct))

                    if leg.tp1_done and leg.shares > 0:
                        if high_px > leg.peak:
                            leg.peak = float(high_px)
                        stop_px = leg.peak * (1.0 - float(args.trail_stop))
                        if np.isfinite(low_px) and low_px <= stop_px:
                            proceeds = leg.shares * stop_px
                            st.seed += proceeds
                            leg.shares = 0.0
                            leg.invested = 0.0

            # emptied by trail -> close
            if sub.in_cycle and all((leg.shares <= 0 for leg in sub.legs)):
                close_sub_cycle(date, day_prices_close, reason="SUB_TRAIL_EXIT_ALL")

            # max_days_sub hard close
            if sub.in_cycle and sub.holding_days >= max_days_sub:
                close_sub_cycle(date, day_prices_close, reason="SUB_MAXDAY_CLOSE")

            # DCA until TP1 done
            if sub.in_cycle:
                all_tp1_sub = all((leg.tp1_done for leg in sub.legs))
                if not all_tp1_sub:
                    desired_total = float(sub.unit)
                    for leg in sub.legs:
                        if leg.ticker not in day_df.index:
                            continue
                        close_px = day_prices_close[leg.ticker]
                        if not np.isfinite(close_px) or close_px <= 0:
                            continue

                        avg = leg.avg_price()
                        desired_leg = desired_total * float(leg.weight)

                        if np.isfinite(avg) and avg > 0:
                            if close_px <= avg:
                                pass
                            elif close_px <= avg * 1.05:
                                desired_leg = desired_leg / 2.0
                            else:
                                desired_leg = 0.0

                        invest_raw = clamp_invest_by_leverage(st.seed, sub.entry_seed, desired_leg, float(args.max_leverage_pct))
                        invest = float(min(invest_raw, tp1_cash_pool)) if tp1_cash_pool > 0 else 0.0

                        if invest > 0:
                            st.seed -= invest
                            tp1_cash_pool -= invest
                            track_pool()

                            leg.invested += invest
                            leg.shares += invest / close_px
                            sub.update_lev(float(args.max_leverage_pct))

            if sub.in_cycle:
                sub.update_dd(day_prices_close)

        # ===== MAIN update
        if st.in_cycle:
            st.holding_days += 1

            # TP1 + trailing
            if enable_trailing:
                for leg in st.legs:
                    if leg.ticker not in day_df.index:
                        continue
                    high_px = day_prices_high[leg.ticker]
                    low_px = day_prices_low[leg.ticker]
                    avg = leg.avg_price()

                    if (not leg.tp1_done) and np.isfinite(avg) and high_px >= avg * (1.0 + float(args.profit_target)):
                        tp_px = avg * (1.0 + float(args.profit_target))
                        sell_shares = leg.shares * float(args.tp1_frac)
                        sell_shares = float(min(leg.shares, max(0.0, sell_shares)))
                        proceeds = sell_shares * tp_px

                        leg.shares -= sell_shares
                        leg.invested *= (leg.shares / (leg.shares + sell_shares)) if (leg.shares + sell_shares) > 0 else 0.0

                        st.seed += proceeds

                        # MAIN TP1 proceeds -> pool
                        tp1_cash_pool += proceeds
                        track_pool()

                        leg.tp1_done = True
                        leg.peak = float(max(high_px, tp_px))
                        st.update_lev(float(args.max_leverage_pct))

                    if leg.tp1_done and leg.shares > 0:
                        if high_px > leg.peak:
                            leg.peak = float(high_px)
                        stop_px = leg.peak * (1.0 - float(args.trail_stop))
                        if np.isfinite(low_px) and low_px <= stop_px:
                            proceeds = leg.shares * stop_px
                            st.seed += proceeds
                            leg.shares = 0.0
                            leg.invested = 0.0

            # close if all emptied
            if st.in_cycle and all((leg.shares <= 0 for leg in st.legs)):
                close_main_cycle(date, day_prices_close, reason="TRAIL_EXIT_ALL")

            # max_days decision
            if st.in_cycle:
                if st.holding_days >= int(args.max_days) and (not st.extending):
                    rets = []
                    for leg in st.legs:
                        if leg.ticker in day_df.index and leg.shares > 0:
                            avg = leg.avg_price()
                            px = day_prices_close[leg.ticker]
                            if np.isfinite(avg) and avg > 0:
                                rets.append((px - avg) / avg)
                    cur_ret = float(np.mean(rets)) if rets else -np.inf

                    if cur_ret >= float(args.stop_level):
                        close_main_cycle(date, day_prices_close, reason="MAXDAY_CLOSE")
                    else:
                        st.extending = True

            # DCA until all TP1 done
            if st.in_cycle:
                all_tp1 = all((leg.tp1_done for leg in st.legs))
                if not all_tp1:
                    desired_total = float(st.unit)
                    for leg in st.legs:
                        if leg.ticker not in day_df.index:
                            continue
                        close_px = day_prices_close[leg.ticker]
                        if not np.isfinite(close_px) or close_px <= 0:
                            continue

                        avg = leg.avg_price()
                        desired_leg = desired_total * float(leg.weight)
                        if not st.extending:
                            if np.isfinite(avg) and avg > 0:
                                if close_px <= avg:
                                    pass
                                elif close_px <= avg * 1.05:
                                    desired_leg = desired_leg / 2.0
                                else:
                                    desired_leg = 0.0

                        invest = clamp_invest_by_leverage(st.seed, st.entry_seed, desired_leg, float(args.max_leverage_pct))
                        if invest > 0:
                            st.seed -= invest
                            leg.invested += invest
                            leg.shares += invest / close_px
                            st.update_lev(float(args.max_leverage_pct))

            if st.in_cycle:
                st.update_dd(day_prices_close)
            else:
                st.update_dd({})

        # ===== MAIN entry
        if (not st.in_cycle) and (not cooldown_today):
            picks_today = picks_by_date.get(date, [])
            if picks_today:
                valid = [t for t in picks_today if t in day_df.index and np.isfinite(day_prices_close.get(t, np.nan))]
                if len(valid) >= 1:
                    chosen = valid[:topk]

                    S0 = float(st.seed)
                    unit = (S0 / float(args.max_days)) if args.max_days > 0 else 0.0

                    legs: list[Leg] = []
                    for i, t in enumerate(chosen):
                        legs.append(Leg(ticker=t, weight=float(weights[i])))

                    invested_total = 0.0
                    for leg in legs:
                        px = day_prices_close[leg.ticker]
                        desired = float(unit) * float(leg.weight)
                        invest = clamp_invest_by_leverage(st.seed, S0, desired, float(args.max_leverage_pct))
                        if invest > 0:
                            st.seed -= invest
                            leg.invested += invest
                            leg.shares += invest / px
                            invested_total += invest

                    if invested_total > 0:
                        st.in_cycle = True
                        st.entry_date = date
                        st.entry_seed = S0
                        st.unit = float(unit)
                        st.holding_days = 1
                        st.extending = False
                        st.max_leverage_pct = 0.0
                        st.legs = legs
                        st.update_lev(float(args.max_leverage_pct))
                        st.update_dd(day_prices_close)

                        sub_used_in_this_main = False
                        tp1_cash_pool = 0.0
                        track_pool()

        # ===== SUB entry (only when main trailing zone, pool>0, not used) + ✅ SubOK==1 candidates only
        if st.in_cycle and (not sub.in_cycle) and (not cooldown_today) and (not sub_used_in_this_main):
            all_main_tp1 = all((leg.tp1_done for leg in st.legs)) if st.legs else False
            if all_main_tp1 and tp1_cash_pool > 0:
                picks_today_sub = picks_by_date_sub.get(date, [])
                if picks_today_sub:
                    t0 = None
                    for t in picks_today_sub:
                        if t in day_df.index and np.isfinite(day_prices_close.get(t, np.nan)):
                            t0 = t
                            break

                    if t0 is not None:
                        entry_pool_sub = float(tp1_cash_pool)
                        unit_sub = (entry_pool_sub / float(max_days_sub)) if max_days_sub > 0 else 0.0

                        leg = Leg(ticker=t0, weight=1.0)
                        px = day_prices_close[t0]

                        invest_raw = clamp_invest_by_leverage(st.seed, entry_pool_sub, unit_sub, float(args.max_leverage_pct))
                        invest = float(min(invest_raw, tp1_cash_pool)) if tp1_cash_pool > 0 else 0.0

                        if invest > 0:
                            st.seed -= invest
                            tp1_cash_pool -= invest
                            track_pool()

                            leg.invested += invest
                            leg.shares += invest / px

                            sub.in_cycle = True
                            sub.entry_date = date
                            sub.entry_seed = entry_pool_sub
                            sub.unit = float(unit_sub)
                            sub.holding_days = 1
                            sub.extending = False
                            sub.max_leverage_pct = 0.0
                            sub.legs = [leg]
                            sub.update_lev(float(args.max_leverage_pct))
                            sub.update_dd({t0: px})

                            sub_used_in_this_main = True

        # ===== record curve
        prices_for_eq = day_prices_close if st.in_cycle else {}
        eq = st.equity(prices_for_eq)

        curve.append({
            "Date": date,
            "Equity": eq,
            "Seed": st.seed,
            "InCycle": int(st.in_cycle),
            "InSubCycle": int(sub.in_cycle),
            "Tp1CashPool": float(tp1_cash_pool),
            "MinTp1CashPoolSoFar": float(min_tp1_cash_pool),
            "Tickers": ",".join([l.ticker for l in st.legs]) if st.in_cycle else "",
            "SubTicker": ",".join([l.ticker for l in sub.legs]) if sub.in_cycle else "",
            "HoldingDays": st.holding_days if st.in_cycle else 0,
            "SubHoldingDays": sub.holding_days if sub.in_cycle else 0,
            "Extending": int(st.extending) if st.in_cycle else 0,
            "MaxLeveragePctCycle": st.max_leverage_pct if st.in_cycle else 0.0,
            "MaxLeveragePctSubCycle": sub.max_leverage_pct if sub.in_cycle else 0.0,
            "MaxDrawdownPortfolio": st.max_dd,
        })

    trades_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(curve)
    if not curve_df.empty:
        curve_df["SeedMultiple"] = curve_df["Equity"] / float(args.initial_seed)

    tag = args.tag if args.tag else "run"
    suffix = args.suffix if args.suffix else picks_path.stem.replace("picks_", "")

    trades_path = Path(args.out_dir) / f"sim_engine_trades_{tag}_gate_{suffix}.parquet"
    curve_path = Path(args.out_dir) / f"sim_engine_curve_{tag}_gate_{suffix}.parquet"
    trades_df.to_parquet(trades_path, index=False)
    curve_df.to_parquet(curve_path, index=False)

    print(f"[DONE] wrote trades: {trades_path} rows={len(trades_df)}")
    print(f"[DONE] wrote curve : {curve_path} rows={len(curve_df)}")
    if not curve_df.empty:
        print(f"[INFO] final SeedMultiple={float(curve_df['SeedMultiple'].iloc[-1]):.4f} maxDD={float(st.max_dd):.4f}")

    # pool minimum check
    print(f"[INFO] min tp1_cash_pool observed = {float(min_tp1_cash_pool):.2f}")


if __name__ == "__main__":
    main()