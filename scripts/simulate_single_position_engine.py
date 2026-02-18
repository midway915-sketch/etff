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
    ap = argparse.ArgumentParser(description="Single-cycle engine with TopK (1~2), TP1 partial, trailing stop, leverage cap on ALL buys.")
    ap.add_argument("--picks-path", required=True, type=str, help="CSV with columns Date,Ticker (TopK rows/day). RankIdx optional.")
    ap.add_argument("--prices-parq", default="data/raw/prices.parquet", type=str)
    ap.add_argument("--prices-csv", default="data/raw/prices.csv", type=str)

    ap.add_argument("--initial-seed", default=40_000_000, type=float)

    ap.add_argument("--profit-target", required=True, type=float)   # PT (e.g. 0.10)
    ap.add_argument("--max-days", required=True, type=int)          # holding trigger (e.g. 40)
    ap.add_argument("--stop-level", required=True, type=float)      # stop threshold for extend decision (e.g. -0.10)
    ap.add_argument("--max-extend-days", required=True, type=int)   # tag compat (no hard limit)

    ap.add_argument("--max-leverage-pct", default=1.0, type=float)  # 1.0 => 100%

    ap.add_argument("--enable-trailing", default="true", type=str)
    ap.add_argument("--tp1-frac", default=0.50, type=float)         # fraction to sell at PT
    ap.add_argument("--trail-stop", default=0.10, type=float)       # peak drawdown for trailing (0.08~0.12)

    ap.add_argument("--topk", default=1, type=int)                  # 1 or 2
    ap.add_argument("--weights", default="1.0", type=str)           # "1.0" or "0.7,0.3" etc

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

    # load picks
    picks_path = Path(args.picks_path)
    if not picks_path.exists():
        raise FileNotFoundError(f"Missing picks file: {picks_path}")

    picks = pd.read_csv(picks_path) if picks_path.suffix.lower() != ".parquet" else pd.read_parquet(picks_path)
    if "Date" not in picks.columns or "Ticker" not in picks.columns:
        raise ValueError(f"picks must have Date,Ticker. cols={list(picks.columns)[:50]}")

    picks = picks.copy()
    picks["Date"] = _norm_date(picks["Date"])
    picks["Ticker"] = picks["Ticker"].astype(str).str.upper().str.strip()
    picks = picks.dropna(subset=["Date", "Ticker"]).sort_values(["Date"]).reset_index(drop=True)

    # keep TopK per day (in case input contains more)
    picks = picks.groupby("Date", group_keys=False).head(topk).reset_index(drop=True)

    # load prices
    prices = read_table(args.prices_parq, args.prices_csv).copy()
    if "Date" not in prices.columns or "Ticker" not in prices.columns:
        raise ValueError("prices must have Date,Ticker")
    prices["Date"] = _norm_date(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    for c in ["Open", "High", "Low", "Close"]:
        if c not in prices.columns:
            raise ValueError(f"prices missing {c}")
    prices = prices.dropna(subset=["Date", "Ticker", "Close"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # date -> list[tickers] picks
    picks_by_date: dict[pd.Timestamp, list[str]] = {}
    for d, g in picks.groupby("Date"):
        picks_by_date[d] = g["Ticker"].tolist()

    grouped = prices.groupby("Date", sort=True)

    st = CycleState(
        seed=float(args.initial_seed),
        max_equity=float(args.initial_seed),
        max_dd=0.0,
        legs=[]
    )

    cooldown_today = False
    trades = []
    curve = []

    def close_cycle(exit_date: pd.Timestamp, day_prices: dict[str, float], reason: str) -> None:
        nonlocal cooldown_today, st, trades
        proceeds = 0.0
        invested_total = 0.0

        # liquidate all legs at close price (or given prices)
        for leg in st.legs:
            px = float(day_prices.get(leg.ticker, np.nan))
            if not np.isfinite(px) or px <= 0:
                px = float(day_prices.get(leg.ticker + "_close_fallback", np.nan))
            if not np.isfinite(px) or px <= 0:
                # if missing price, skip (should be rare)
                continue
            proceeds += leg.shares * px
            invested_total += leg.invested

        cycle_return = (proceeds - invested_total) / invested_total if invested_total > 0 else np.nan
        win = int(cycle_return > 0) if np.isfinite(cycle_return) else 0

        trades.append({
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

        # reset cycle
        st.in_cycle = False
        st.entry_date = None
        st.entry_seed = 0.0
        st.unit = 0.0
        st.holding_days = 0
        st.extending = False
        st.max_leverage_pct = 0.0
        st.legs = []

        cooldown_today = True

    # simulate
    for date, day_df in grouped:
        day_df = day_df.set_index("Ticker", drop=False)
        cooldown_today = False

        # map prices for equity
        day_prices_close = {}
        day_prices_high = {}
        day_prices_low = {}

        for t in day_df.index:
            r = day_df.loc[t]
            day_prices_close[t] = float(r["Close"])
            day_prices_high[t] = float(r["High"])
            day_prices_low[t] = float(r["Low"])

        # ----- in cycle: update + exits + buys
        if st.in_cycle:
            st.holding_days += 1

            # 1) TP1 + trailing per leg
            if enable_trailing:
                for leg in st.legs:
                    if leg.ticker not in day_df.index:
                        continue
                    close_px = day_prices_close[leg.ticker]
                    high_px = day_prices_high[leg.ticker]
                    low_px = day_prices_low[leg.ticker]
                    avg = leg.avg_price()

                    # TP1 trigger
                    if (not leg.tp1_done) and np.isfinite(avg) and high_px >= avg * (1.0 + float(args.profit_target)):
                        tp_px = avg * (1.0 + float(args.profit_target))
                        sell_shares = leg.shares * float(args.tp1_frac)
                        sell_shares = float(min(leg.shares, max(0.0, sell_shares)))
                        proceeds = sell_shares * tp_px

                        leg.shares -= sell_shares
                        # invested bookkeeping: reduce proportionally
                        leg.invested *= (leg.shares / (leg.shares + sell_shares)) if (leg.shares + sell_shares) > 0 else 0.0

                        st.seed += proceeds
                        leg.tp1_done = True
                        leg.peak = float(max(high_px, tp_px))

                        st.update_lev(float(args.max_leverage_pct))

                    # trailing stop after TP1
                    if leg.tp1_done and leg.shares > 0:
                        if high_px > leg.peak:
                            leg.peak = float(high_px)
                        stop_px = leg.peak * (1.0 - float(args.trail_stop))
                        if np.isfinite(low_px) and low_px <= stop_px:
                            # sell remaining at stop_px
                            proceeds = leg.shares * stop_px
                            st.seed += proceeds
                            leg.shares = 0.0
                            leg.invested = 0.0

            # 2) if all legs emptied by TP1+trailing -> close cycle today
            if st.in_cycle and all((leg.shares <= 0 for leg in st.legs)):
                close_cycle(date, day_prices_close, reason="TRAIL_EXIT_ALL")

            # 3) max_days decision (only if still holding something)
            if st.in_cycle:
                # decide extend if at max_days and still below stop threshold (cycle-level, using weighted avg return)
                if st.holding_days >= int(args.max_days) and (not st.extending):
                    # compute weighted avg return vs avg price
                    rets = []
                    for leg in st.legs:
                        if leg.ticker in day_df.index and leg.shares > 0:
                            avg = leg.avg_price()
                            px = day_prices_close[leg.ticker]
                            if np.isfinite(avg) and avg > 0:
                                rets.append((px - avg) / avg)
                    cur_ret = float(np.mean(rets)) if rets else -np.inf

                    if cur_ret >= float(args.stop_level):
                        close_cycle(date, day_prices_close, reason="MAXDAY_CLOSE")
                    else:
                        st.extending = True

            # 4) buys (DCA) - leverage cap applies ALWAYS
            if st.in_cycle:
                # If already TP1 done on ALL legs, we don't DCA anymore (let trailing run)
                all_tp1 = all((leg.tp1_done for leg in st.legs))
                if not all_tp1:
                    # desired spend today = unit (cycle-level), split by weights
                    desired_total = float(st.unit)

                    # If extending, still DCA but capped
                    # If not extending, DCA only when price <= avg (or near avg)
                    for leg in st.legs:
                        if leg.ticker not in day_df.index:
                            continue
                        close_px = day_prices_close[leg.ticker]
                        if not np.isfinite(close_px) or close_px <= 0:
                            continue

                        avg = leg.avg_price()
                        # normal zone buy rule
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

            # update dd
            if st.in_cycle:
                st.update_dd(day_prices_close)
            else:
                st.update_dd({})

        # ----- entry: not in cycle and not cooldown day
        if (not st.in_cycle) and (not cooldown_today):
            picks_today = picks_by_date.get(date, [])
            if picks_today:
                # ensure all picks have prices today
                valid = [t for t in picks_today if t in day_df.index and np.isfinite(day_prices_close.get(t, np.nan))]
                if len(valid) >= 1:
                    # choose up to topk from valid (already ordered)
                    chosen = valid[:topk]

                    S0 = float(st.seed)
                    unit = (S0 / float(args.max_days)) if args.max_days > 0 else 0.0

                    # build legs
                    legs = []
                    for i, t in enumerate(chosen):
                        legs.append(Leg(ticker=t, weight=float(weights[i])))

                    # first day buy: unit split by weights
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

        # ----- record curve
        prices_for_eq = day_prices_close if st.in_cycle else {}
        eq = st.equity(prices_for_eq)

        curve.append({
            "Date": date,
            "Equity": eq,
            "Seed": st.seed,
            "InCycle": int(st.in_cycle),
            "Tickers": ",".join([l.ticker for l in st.legs]) if st.in_cycle else "",
            "HoldingDays": st.holding_days if st.in_cycle else 0,
            "Extending": int(st.extending) if st.in_cycle else 0,
            "MaxLeveragePctCycle": st.max_leverage_pct if st.in_cycle else 0.0,
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


if __name__ == "__main__":
    main()