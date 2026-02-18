# scripts/predict_select.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

DATA_DIR = Path("data")
FEATURE_DIR = DATA_DIR / "features"
LABEL_DIR = DATA_DIR / "labels"
SIGNAL_DIR = DATA_DIR / "signals"
MODEL_DIR = Path("app") / "model"

FEATURES_MODEL_PARQUET = FEATURE_DIR / "features_model.parquet"
FEATURES_MODEL_CSV = FEATURE_DIR / "features_model.csv"
FEATURES_PARQUET = FEATURE_DIR / "features.parquet"
FEATURES_CSV = FEATURE_DIR / "features.csv"


def fmt_tag(pt: float, max_days: int, sl: float, max_ext: int) -> str:
    pt_tag = int(round(pt * 100))
    sl_tag = int(round(abs(sl) * 100))
    return f"pt{pt_tag}_h{max_days}_sl{sl_tag}_ex{max_ext}"


def load_features() -> pd.DataFrame:
    if FEATURES_MODEL_PARQUET.exists():
        f = pd.read_parquet(FEATURES_MODEL_PARQUET)
    elif FEATURES_MODEL_CSV.exists():
        f = pd.read_csv(FEATURES_MODEL_CSV)
    else:
        if FEATURES_PARQUET.exists():
            f = pd.read_parquet(FEATURES_PARQUET)
        else:
            f = pd.read_csv(FEATURES_CSV)

    f["Date"] = pd.to_datetime(f["Date"])
    f["Ticker"] = f["Ticker"].astype(str).str.upper().str.strip()
    return f.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def load_labels(tag: str) -> pd.DataFrame:
    p = LABEL_DIR / f"strategy_labels_{tag}.parquet"
    c = LABEL_DIR / f"strategy_labels_{tag}.csv"
    if p.exists():
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(c)
    df["Date"] = pd.to_datetime(df["Date"])
    df["ExitDate"] = pd.to_datetime(df["ExitDate"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    return df.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def get_market_row(day_df: pd.DataFrame) -> dict:
    cols = [
        "VIX",
        "Market_Drawdown_60",
        "Market_Drawdown_252",
        "Market_realized_vol_20",
        "Market_ret_1d",
    ]
    row = {}
    first = day_df.iloc[0]
    for c in cols:
        row[c] = float(first[c]) if c in day_df.columns and pd.notna(first[c]) else np.nan
    return row


def apply_killswitch(market: dict, args) -> tuple[bool, int, float, str]:
    """
    returns: (skip, topk_used, lambda_used, reason)
    """
    vix = market.get("VIX", np.nan)
    dd60 = market.get("Market_Drawdown_60", np.nan)

    topk_used = args.topk
    lam_used = args.lambda_risk
    reason = ""

    # hard skip
    if np.isfinite(vix) and vix >= args.hard_vix:
        return True, topk_used, lam_used, f"hard_skip_vix>={args.hard_vix} (vix={vix:.2f})"
    if np.isfinite(dd60) and dd60 <= -abs(args.hard_dd60):
        return True, topk_used, lam_used, f"hard_skip_dd60<={-abs(args.hard_dd60):.2f} (dd60={dd60:.3f})"

    # soft risk mode
    if np.isfinite(vix) and vix >= args.soft_vix:
        topk_used = max(1, min(topk_used, args.soft_topk))
        lam_used = max(lam_used, args.soft_lambda)
        reason = f"soft_risk_vix>={args.soft_vix} (vix={vix:.2f})"
    elif np.isfinite(dd60) and dd60 <= -abs(args.soft_dd60):
        topk_used = max(1, min(topk_used, args.soft_topk))
        lam_used = max(lam_used, args.soft_lambda)
        reason = f"soft_risk_dd60<={-abs(args.soft_dd60):.2f} (dd60={dd60:.3f})"

    return False, topk_used, lam_used, reason


def summarize_eval(df: pd.DataFrame, tail_col: str = "TailFlag") -> pd.DataFrame:
    """
    df columns expected:
      - method
      - CycleReturn, MinCycleRet, ExtendDays, ForcedExitFlag, TailFlag
    """
    out = (
        df.dropna(subset=["CycleReturn"])
        .groupby("method")
        .agg(
            n=("CycleReturn", "count"),
            win_rate=("CycleReturn", lambda x: float((x > 0).mean())),
            avg_ret=("CycleReturn", "mean"),
            tail_rate=(tail_col, "mean"),
            worst_min_ret=("MinCycleRet", "min"),
            avg_extend_days=("ExtendDays", "mean"),
            forced_exit_rate=("ForcedExitFlag", "mean"),
        )
        .reset_index()
        .sort_values("avg_ret", ascending=False)
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)

    ap.add_argument("--tail-threshold", type=float, default=-0.30)
    ap.add_argument("--lambda-risk", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=5)

    # killswitch params
    ap.add_argument("--hard-vix", type=float, default=40.0)
    ap.add_argument("--hard-dd60", type=float, default=0.18)
    ap.add_argument("--soft-vix", type=float, default=30.0)
    ap.add_argument("--soft-dd60", type=float, default=0.12)
    ap.add_argument("--soft-topk", type=int, default=2)
    ap.add_argument("--soft-lambda", type=float, default=0.10)

    # shadow evaluation: skip day에는 ret_only로 들어간다고 가정
    ap.add_argument("--shadow-on-skip", action="store_true", help="skipped day에 ret_only trade를 넣은 shadow 성과도 출력")

    args = ap.parse_args()

    tag = fmt_tag(args.profit_target, args.max_days, args.stop_level, args.max_extend_days)
    thr_tag = int(round(abs(args.tail_threshold) * 100))

    clf_pos = joblib.load(MODEL_DIR / f"clf_pos_{tag}.pkl")
    clf_tail = joblib.load(MODEL_DIR / f"clf_tail_{tag}_thr{thr_tag}.pkl")
    reg = joblib.load(MODEL_DIR / f"reg_ret_{tag}.pkl")
    scaler = joblib.load(MODEL_DIR / f"scaler_{tag}.pkl")

    feat = load_features()
    lab = load_labels(tag)

    feature_cols = [c for c in feat.columns if c not in ("Date", "Ticker")]
    feature_cols = feat[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X = feat[feature_cols].to_numpy(dtype=float)
    Xs = scaler.transform(X)

    feat = feat.copy()
    feat["p_pos"] = clf_pos.predict_proba(Xs)[:, 1]
    feat["p_tail"] = clf_tail.predict_proba(Xs)[:, 1]
    feat["pred_ret"] = reg.predict(Xs)

    # 날짜별 picks (스킵이어도 "만약 거래했다면" 픽은 계산해둠)
    pick_rows = []
    for date, g in feat.groupby("Date", sort=False):
        g = g.reset_index(drop=True)

        market = get_market_row(g)
        skip, topk_used, lam_used, reason = apply_killswitch(market, args)

        g = g.copy()
        g["utility"] = g["pred_ret"] - float(lam_used) * g["p_tail"]

        # always compute "would-be picks"
        pick_ret = g.loc[g["pred_ret"].idxmax(), "Ticker"]
        pick_p = g.loc[g["p_pos"].idxmax(), "Ticker"]

        gg = g.sort_values("p_pos", ascending=False).head(max(1, min(topk_used, len(g))))
        pick_gate_ret = gg.loc[gg["pred_ret"].idxmax(), "Ticker"]
        pick_gate_utility = gg.loc[gg["utility"].idxmax(), "Ticker"]

        pick_utility = g.loc[g["utility"].idxmax(), "Ticker"]

        g2 = g.copy()
        g2["rank_p"] = g2["p_pos"].rank(ascending=False, method="average")
        g2["rank_r"] = g2["pred_ret"].rank(ascending=False, method="average")
        g2["blend_score"] = -(0.5 * g2["rank_p"] + 0.5 * g2["rank_r"])
        pick_blend = g2.loc[g2["blend_score"].idxmax(), "Ticker"]

        pick_rows.append(
            {
                "Date": date,
                "Skipped": 1 if skip else 0,
                "SkipReason": reason if skip else "",
                "topk_used": topk_used,
                "lambda_used": lam_used,
                # would-be picks (skip이어도 기록)
                "pick_ret_only": pick_ret,
                "pick_p_only": pick_p,
                "pick_gate_ret": pick_gate_ret,
                "pick_utility": pick_utility,
                "pick_gate_utility": pick_gate_utility,
                "pick_blend": pick_blend,
                "VIX": market.get("VIX", np.nan),
                "Market_Drawdown_60": market.get("Market_Drawdown_60", np.nan),
            }
        )

    picks = pd.DataFrame(pick_rows).sort_values("Date").reset_index(drop=True)

    # 라벨 준비
    lab_eval = lab[["Date", "Ticker", "CycleReturn", "MinCycleRet", "ExtendDays", "ForcedExitFlag"]].copy()
    lab_eval["TailFlag"] = (lab_eval["MinCycleRet"].astype(float) <= float(args.tail_threshold)).astype(int)

    # ====== 1) 기존 방식: "거래한 날만" 평가 ======
    def attach_traded(method_col: str, name: str) -> pd.DataFrame:
        tmp = picks[picks["Skipped"] == 0][["Date", method_col]].rename(columns={method_col: "Ticker"})
        out = tmp.merge(lab_eval, on=["Date", "Ticker"], how="left")
        out["method"] = name
        out["Skipped"] = 0
        return out

    eval_traded = pd.concat(
        [
            attach_traded("pick_ret_only", "ret_only"),
            attach_traded("pick_p_only", "p_only"),
            attach_traded("pick_gate_ret", "gate_topk_then_ret"),
            attach_traded("pick_utility", "utility"),
            attach_traded("pick_gate_utility", "gate_topk_then_utility"),
            attach_traded("pick_blend", "blend_rank"),
        ],
        ignore_index=True,
    )
    summary_traded = summarize_eval(eval_traded)

    # ====== 2) "스킵은 0 수익" 포함한 일별 성과(기회비용 확인용) ======
    def attach_daily_zero(method_col: str, name: str) -> pd.DataFrame:
        tmp = picks[["Date", "Skipped", method_col]].rename(columns={method_col: "Ticker"})
        out = tmp.merge(lab_eval, on=["Date", "Ticker"], how="left")
        out["method"] = name
        # 스킵은 0 수익으로 간주(그리고 리스크 지표도 neutral 처리)
        out.loc[out["Skipped"] == 1, "CycleReturn"] = 0.0
        out.loc[out["Skipped"] == 1, "TailFlag"] = 0
        out.loc[out["Skipped"] == 1, "ForcedExitFlag"] = 0
        # ExtendDays/MinCycleRet는 NaN로 둬도 되고 0으로 둬도 되는데, 여기선 NaN 유지
        return out

    eval_daily_zero = pd.concat(
        [
            attach_daily_zero("pick_ret_only", "ret_only_daily0"),
            attach_daily_zero("pick_p_only", "p_only_daily0"),
            attach_daily_zero("pick_gate_ret", "gate_topk_then_ret_daily0"),
            attach_daily_zero("pick_utility", "utility_daily0"),
            attach_daily_zero("pick_gate_utility", "gate_topk_then_utility_daily0"),
            attach_daily_zero("pick_blend", "blend_rank_daily0"),
        ],
        ignore_index=True,
    )

    # daily0 summary는 "일 단위"로 보는게 목적이라, n=전체일수에 가까워야 함
    summary_daily0 = (
        eval_daily_zero.dropna(subset=["CycleReturn"])
        .groupby("method")
        .agg(
            days=("CycleReturn", "count"),
            avg_daily_ret=("CycleReturn", "mean"),
            cum_ret=("CycleReturn", lambda x: float((1.0 + x).prod() - 1.0)),
            win_rate=("CycleReturn", lambda x: float((x > 0).mean())),
            tail_rate=("TailFlag", "mean"),
        )
        .reset_index()
        .sort_values("avg_daily_ret", ascending=False)
    )

    # ====== 3) Shadow: "스킵한 날은 ret_only로 거래했다면?" ======
    eval_shadow = None
    summary_shadow = None
    if args.shadow_on_skip:
        def attach_shadow(method_col: str, name: str) -> pd.DataFrame:
            # non-skip: method_col, skip: ret_only
            # ⚠️ method_col이 pick_ret_only면 컬럼 중복이 생겨 Series가 아니라 DF가 되어 np.where가 터짐
            if method_col == "pick_ret_only":
                use = picks[["Date", "Skipped", "pick_ret_only"]].copy()
                use["Ticker"] = use["pick_ret_only"]
            else:
                use = picks[["Date", "Skipped", "pick_ret_only", method_col]].copy()
                use = use.rename(columns={method_col: "_method_pick"})
                use["Ticker"] = np.where(
                    use["Skipped"].to_numpy() == 1,
                    use["pick_ret_only"].to_numpy(),
                    use["_method_pick"].to_numpy(),
                )

            out = use[["Date", "Skipped", "Ticker"]].merge(lab_eval, on=["Date", "Ticker"], how="left")
            out["method"] = name
            out["ShadowRule"] = "ret_only_on_skips"
            return out

        eval_shadow = pd.concat(
            [
                attach_shadow("pick_ret_only", "ret_only_shadow"),  # 사실상 동일
                attach_shadow("pick_p_only", "p_only_shadow"),
                attach_shadow("pick_gate_ret", "gate_topk_then_ret_shadow"),
                attach_shadow("pick_utility", "utility_shadow"),
                attach_shadow("pick_gate_utility", "gate_topk_then_utility_shadow"),
                attach_shadow("pick_blend", "blend_rank_shadow"),
            ],
            ignore_index=True,
        )


        summary_shadow = (
            eval_shadow.dropna(subset=["CycleReturn"])
            .groupby("method")
            .agg(
                days=("CycleReturn", "count"),
                avg_daily_ret=("CycleReturn", "mean"),
                cum_ret=("CycleReturn", lambda x: float((1.0 + x).prod() - 1.0)),
                win_rate=("CycleReturn", lambda x: float((x > 0).mean())),
                tail_rate=("TailFlag", "mean"),
                worst_min_ret=("MinCycleRet", "min"),
                forced_exit_rate=("ForcedExitFlag", "mean"),
            )
            .reset_index()
            .sort_values("avg_daily_ret", ascending=False)
        )

    # 스킵 통계
    skip_stats = {
        "total_days": int(len(picks)),
        "skipped_days": int(picks["Skipped"].sum()),
        "skip_rate": float(picks["Skipped"].sum() / max(1, len(picks))),
    }

    # 저장
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    picks_path = SIGNAL_DIR / f"picks_{tag}_ks.csv"
    eval_traded_path = SIGNAL_DIR / f"eval_traded_{tag}_ks.csv"
    summary_traded_path = SIGNAL_DIR / f"summary_traded_{tag}_ks.csv"
    summary_daily0_path = SIGNAL_DIR / f"summary_daily0_{tag}_ks.csv"
    skip_path = SIGNAL_DIR / f"skip_{tag}_ks.json"

    picks.to_csv(picks_path, index=False)
    eval_traded.to_csv(eval_traded_path, index=False)
    summary_traded.to_csv(summary_traded_path, index=False)
    summary_daily0.to_csv(summary_daily0_path, index=False)
    skip_path.write_text(pd.Series(skip_stats).to_json(), encoding="utf-8")

    if args.shadow_on_skip and eval_shadow is not None and summary_shadow is not None:
        eval_shadow_path = SIGNAL_DIR / f"eval_shadow_{tag}_ks.csv"
        summary_shadow_path = SIGNAL_DIR / f"summary_shadow_{tag}_ks.csv"
        eval_shadow.to_csv(eval_shadow_path, index=False)
        summary_shadow.to_csv(summary_shadow_path, index=False)
        print("Saved:", eval_shadow_path)
        print("Saved:", summary_shadow_path)

    print("Saved:", picks_path)
    print("Saved:", eval_traded_path)
    print("Saved:", summary_traded_path)
    print("Saved:", summary_daily0_path)
    print("Saved:", skip_path)

    print("\nSkip stats:", skip_stats)
    print("\n=== Traded-only summary ===")
    print(summary_traded.to_string(index=False))
    print("\n=== Daily( skip=0 ) summary ===")
    print(summary_daily0.to_string(index=False))
    if args.shadow_on_skip and summary_shadow is not None:
        print("\n=== Shadow( ret_only on skips ) summary ===")
        print(summary_shadow.to_string(index=False))


if __name__ == "__main__":
    main()
