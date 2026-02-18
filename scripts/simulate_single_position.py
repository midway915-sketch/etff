# scripts/simulate_single_position.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
LABEL_DIR = DATA_DIR / "labels"
SIGNAL_DIR = DATA_DIR / "signals"

# pick columns mapping
METHOD_TO_PICKCOL = {
    "ret_only": "pick_ret_only",
    "p_only": "pick_p_only",
    "gate_topk_then_ret": "pick_gate_ret",
    "utility": "pick_utility",
    "gate_topk_then_utility": "pick_gate_utility",
    "blend_rank": "pick_blend",
}

# label file name format
def fmt_tag(pt: float, max_days: int, sl: float, max_ext: int) -> str:
    pt_tag = int(round(pt * 100))
    sl_tag = int(round(abs(sl) * 100))
    return f"pt{pt_tag}_h{max_days}_sl{sl_tag}_ex{max_ext}"


def _dt(x) -> pd.Timestamp:
    return pd.to_datetime(x, errors="coerce")


def load_picks(tag: str) -> pd.DataFrame:
    p = SIGNAL_DIR / f"picks_{tag}_ks.csv"
    if not p.exists():
        raise FileNotFoundError(f"picks file not found: {p}")
    df = pd.read_csv(p)
    df["Date"] = pd.to_datetime(df["Date"])
    if "Skipped" not in df.columns:
        df["Skipped"] = 0
    return df.sort_values("Date").reset_index(drop=True)


def load_labels(tag: str) -> pd.DataFrame:
    p_parq = LABEL_DIR / f"strategy_labels_{tag}.parquet"
    p_csv = LABEL_DIR / f"strategy_labels_{tag}.csv"
    if p_parq.exists():
        df = pd.read_parquet(p_parq)
    elif p_csv.exists():
        df = pd.read_csv(p_csv)
    else:
        raise FileNotFoundError(f"labels not found: {p_parq} or {p_csv}")

    # normalize
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    if "ExitDate" in df.columns:
        df["ExitDate"] = pd.to_datetime(df["ExitDate"])
    else:
        raise ValueError("strategy_labels must contain ExitDate")
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()

    # ensure columns exist
    for c in ["CycleReturn", "MinCycleRet", "ExtendDays", "ForcedExitFlag"]:
        if c not in df.columns:
            df[c] = np.nan

    return df.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def build_label_map(labels: pd.DataFrame) -> Dict[Tuple[pd.Timestamp, str], dict]:
    m: Dict[Tuple[pd.Timestamp, str], dict] = {}
    for _, r in labels.iterrows():
        key = (r["Date"], r["Ticker"])
        m[key] = {
            "ExitDate": r["ExitDate"],
            "CycleReturn": float(r["CycleReturn"]) if pd.notna(r["CycleReturn"]) else np.nan,
            "MinCycleRet": float(r["MinCycleRet"]) if pd.notna(r["MinCycleRet"]) else np.nan,
            "ExtendDays": float(r["ExtendDays"]) if pd.notna(r["ExtendDays"]) else np.nan,
            "ForcedExitFlag": int(r["ForcedExitFlag"]) if pd.notna(r["ForcedExitFlag"]) else 0,
        }
    return m


@dataclass
class OpenTrade:
    entry_date: pd.Timestamp
    ticker: str
    planned_exit: pd.Timestamp
    entry_equity: float
    cycle_return: float
    min_cycle_ret: float
    forced_exit_flag: int
    extend_days: float


def safe_save(df: pd.DataFrame, path_parquet: Path, path_csv: Path) -> str:
    path_parquet.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path_parquet, index=False)
        return str(path_parquet)
    except Exception:
        df.to_csv(path_csv, index=False)
        return str(path_csv)


def simulate_one(
    picks: pd.DataFrame,
    label_map: Dict[Tuple[pd.Timestamp, str], dict],
    method_key: str,
    pick_col: str,
    initial_equity: float,
    shadow_on_skip: bool,
) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    equity = float(initial_equity)
    peak = equity
    max_dd = 0.0

    open_trade: Optional[OpenTrade] = None
    trades: List[dict] = []
    curve_rows: List[dict] = []

    skipped_days = 0
    ignored_days_holding = 0
    missing_label = 0
    entered_days = 0

    # Iterate all decision days (picks has one row per Date)
    for _, row in picks.iterrows():
        date = row["Date"]
        skipped = int(row.get("Skipped", 0))
        ticker_pick = row.get(pick_col, None)

        # 1) Close trade if we have passed/arrived at exit date (close at first available decision day >= planned_exit)
        if open_trade is not None and date >= open_trade.planned_exit:
            eq_before = equity
            equity = equity * (1.0 + open_trade.cycle_return)

            peak = max(peak, equity)
            dd = (equity - peak) / peak if peak > 0 else 0.0
            max_dd = min(max_dd, dd)

            trades.append({
                "Method": method_key,
                "EntryDate": open_trade.entry_date,
                "ExitDatePlanned": open_trade.planned_exit,
                "ExitDateActual": date,
                "Ticker": open_trade.ticker,
                "EquityBefore": eq_before,
                "EquityAfter": equity,
                "CycleReturn": open_trade.cycle_return,
                "PnL": equity - eq_before,
                "MinCycleRet": open_trade.min_cycle_ret,
                "ForcedExitFlag": open_trade.forced_exit_flag,
                "ExtendDays": open_trade.extend_days,
                "HoldDaysDecisionCalendar": (date - open_trade.entry_date).days,
            })
            open_trade = None

        # 2) If still holding (exit date after this date), ignore today's decision
        if open_trade is not None:
            ignored_days_holding += 1
            curve_rows.append({
                "Date": date,
                "Method": method_key,
                "Equity": equity,
                "InPosition": 1,
                "PickedTicker": np.nan,
                "Action": "HOLD",
            })
            continue

        # 3) Not in position: decide whether to enter
        if skipped == 1 and not shadow_on_skip:
            skipped_days += 1
            curve_rows.append({
                "Date": date,
                "Method": method_key,
                "Equity": equity,
                "InPosition": 0,
                "PickedTicker": np.nan,
                "Action": "SKIP",
            })
            peak = max(peak, equity)
            dd = (equity - peak) / peak if peak > 0 else 0.0
            max_dd = min(max_dd, dd)
            continue

        if ticker_pick is None or (isinstance(ticker_pick, float) and np.isnan(ticker_pick)) or str(ticker_pick).strip() == "":
            curve_rows.append({
                "Date": date,
                "Method": method_key,
                "Equity": equity,
                "InPosition": 0,
                "PickedTicker": np.nan,
                "Action": "NO_PICK",
            })
            continue

        ticker = str(ticker_pick).upper().strip()

        # if shadow_on_skip: on skipped day, force ticker=ret_only pick
        if skipped == 1 and shadow_on_skip:
            ticker = str(row.get("pick_ret_only", ticker)).upper().strip()

        info = label_map.get((date, ticker))
        if info is None:
            missing_label += 1
            curve_rows.append({
                "Date": date,
                "Method": method_key,
                "Equity": equity,
                "InPosition": 0,
                "PickedTicker": ticker,
                "Action": "MISSING_LABEL",
            })
            continue

        # Open trade
        cycle_ret = float(info["CycleReturn"]) if np.isfinite(info["CycleReturn"]) else 0.0
        min_ret = float(info["MinCycleRet"]) if np.isfinite(info["MinCycleRet"]) else np.nan
        planned_exit = info["ExitDate"]

        open_trade = OpenTrade(
            entry_date=date,
            ticker=ticker,
            planned_exit=planned_exit,
            entry_equity=equity,
            cycle_return=cycle_ret,
            min_cycle_ret=min_ret,
            forced_exit_flag=int(info.get("ForcedExitFlag", 0)),
            extend_days=float(info.get("ExtendDays", np.nan)),
        )
        entered_days += 1

        # Use MinCycleRet to update drawdown estimate (intra-trade worst)
        if np.isfinite(min_ret):
            trough = open_trade.entry_equity * (1.0 + min_ret)
            peak = max(peak, open_trade.entry_equity)  # entry equity can be new peak
            dd_trough = (trough - peak) / peak if peak > 0 else 0.0
            max_dd = min(max_dd, dd_trough)

        curve_rows.append({
            "Date": date,
            "Method": method_key,
            "Equity": equity,
            "InPosition": 1,
            "PickedTicker": ticker,
            "Action": "ENTER",
            "PlannedExit": planned_exit,
            "CycleReturn": cycle_ret,
            "MinCycleRet": min_ret,
        })

    # If still open at the end, leave it open (no mark). 기록만 남김.
    if open_trade is not None:
        trades.append({
            "Method": method_key,
            "EntryDate": open_trade.entry_date,
            "ExitDatePlanned": open_trade.planned_exit,
            "ExitDateActual": pd.NaT,
            "Ticker": open_trade.ticker,
            "EquityBefore": open_trade.entry_equity,
            "EquityAfter": np.nan,
            "CycleReturn": open_trade.cycle_return,
            "PnL": np.nan,
            "MinCycleRet": open_trade.min_cycle_ret,
            "ForcedExitFlag": open_trade.forced_exit_flag,
            "ExtendDays": open_trade.extend_days,
            "HoldDaysDecisionCalendar": (picks["Date"].max() - open_trade.entry_date).days,
            "NOTE": "OPEN_AT_END",
        })

    trades_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(curve_rows)

    # Summary
    closed = trades_df.dropna(subset=["EquityAfter"]).copy()
    n_trades = int(len(closed))
    win_rate = float((closed["CycleReturn"] > 0).mean()) if n_trades > 0 else 0.0
    avg_cycle_ret = float(closed["CycleReturn"].mean()) if n_trades > 0 else 0.0
    avg_hold_days = float(closed["HoldDaysDecisionCalendar"].mean()) if n_trades > 0 else 0.0
    forced_exit_rate = float((closed["ForcedExitFlag"] == 1).mean()) if n_trades > 0 else 0.0

    tail_threshold = -0.30
    if "MinCycleRet" in closed.columns and n_trades > 0:
        tail_rate = float((closed["MinCycleRet"] <= tail_threshold).mean())
    else:
        tail_rate = 0.0

    summary = {
        "method": method_key,
        "pick_col": pick_col,
        "initial_equity": initial_equity,
        "final_equity": float(equity),
        "total_return": float(equity / initial_equity - 1.0) if initial_equity != 0 else np.nan,
        "max_drawdown_est": float(max_dd),  # MinCycleRet 기반 intra-trade 포함(추정)
        "trades_closed": n_trades,
        "win_rate": win_rate,
        "avg_cycle_return": avg_cycle_ret,
        "avg_hold_days_calendar": avg_hold_days,
        "forced_exit_rate": forced_exit_rate,
        "tail_rate_minret_le_-0.30": tail_rate,
        "skipped_days": int(skipped_days),
        "ignored_days_while_holding": int(ignored_days_holding),
        "missing_label_days": int(missing_label),
        "entered_days": int(entered_days),
        "shadow_on_skip": bool(shadow_on_skip),
    }

    return trades_df, summary, curve_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)

    ap.add_argument("--method", type=str, default="gate_topk_then_utility",
                    help="ret_only|p_only|gate_topk_then_ret|utility|gate_topk_then_utility|blend_rank|all")
    ap.add_argument("--initial-equity", type=float, default=1.0)
    ap.add_argument("--shadow-on-skip", action="store_true", help="skipped day에도 ret_only로 진입했다고 가정(기회비용)")
    args = ap.parse_args()

    tag = fmt_tag(args.profit_target, args.max_days, args.stop_level, args.max_extend_days)

    picks = load_picks(tag)
    labels = load_labels(tag)
    label_map = build_label_map(labels)

    methods = []
    if args.method == "all":
        methods = list(METHOD_TO_PICKCOL.items())
    else:
        if args.method not in METHOD_TO_PICKCOL:
            raise ValueError(f"Unknown method: {args.method}. Use one of {list(METHOD_TO_PICKCOL.keys())} or all")
        methods = [(args.method, METHOD_TO_PICKCOL[args.method])]

    all_summ = []
    for method_key, pick_col in methods:
        trades_df, summary, curve_df = simulate_one(
            picks=picks,
            label_map=label_map,
            method_key=method_key,
            pick_col=pick_col,
            initial_equity=args.initial_equity,
            shadow_on_skip=args.shadow_on_skip,
        )

        # save
        out_trades_parq = SIGNAL_DIR / f"sim_singlepos_trades_{tag}_{method_key}.parquet"
        out_trades_csv = SIGNAL_DIR / f"sim_singlepos_trades_{tag}_{method_key}.csv"
        out_curve_parq = SIGNAL_DIR / f"sim_singlepos_curve_{tag}_{method_key}.parquet"
        out_curve_csv = SIGNAL_DIR / f"sim_singlepos_curve_{tag}_{method_key}.csv"
        out_summary_json = SIGNAL_DIR / f"sim_singlepos_summary_{tag}_{method_key}.json"

        saved_trades = safe_save(trades_df, out_trades_parq, out_trades_csv)
        saved_curve = safe_save(curve_df, out_curve_parq, out_curve_csv)
        SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
        out_summary_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

        print(f"[DONE] {method_key} trades -> {saved_trades}")
        print(f"[DONE] {method_key} curve  -> {saved_curve}")
        print(f"[DONE] {method_key} summary-> {out_summary_json}")

        all_summ.append(summary)

    # combined summary
    summ_df = pd.DataFrame(all_summ)
    comb_csv = SIGNAL_DIR / f"sim_singlepos_summary_{tag}_ALL.csv"
    summ_df.sort_values("final_equity", ascending=False).to_csv(comb_csv, index=False)
    print("\n=== SINGLE POSITION SUMMARY (sorted by final_equity) ===")
    print(summ_df.sort_values("final_equity", ascending=False).to_string(index=False))
    print(f"\n[COMBINED] {comb_csv}")


if __name__ == "__main__":
    main()
