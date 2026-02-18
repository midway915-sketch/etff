# scripts/build_ev_features.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
FEATURE_DIR = DATA_DIR / "features"
LABEL_DIR = DATA_DIR / "labels"

FEATURES_PARQUET = FEATURE_DIR / "features.parquet"
FEATURES_CSV = FEATURE_DIR / "features.csv"

OUT_MODEL_PARQUET = FEATURE_DIR / "features_model.parquet"
OUT_MODEL_CSV = FEATURE_DIR / "features_model.csv"
META_JSON = FEATURE_DIR / "features_model_meta.json"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _dt_ns(x: pd.Series) -> pd.Series:
    # merge_asof dtype 단위까지 같아야 해서 ns로 통일
    s = pd.to_datetime(x, errors="coerce")
    # 혹시 timezone이 섞였으면 제거
    try:
        s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s.astype("datetime64[ns]")

def load_features() -> pd.DataFrame:
    if FEATURES_PARQUET.exists():
        f = pd.read_parquet(FEATURES_PARQUET)
    elif FEATURES_CSV.exists():
        f = pd.read_csv(FEATURES_CSV)
    else:
        raise FileNotFoundError("features not found. Run scripts/build_features.py first.")
    f = f.copy()
    f["Date"] = pd.to_datetime(f["Date"])
    f["Ticker"] = f["Ticker"].astype(str).str.upper().str.strip()
    return f.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def load_strategy_labels(tag: str) -> pd.DataFrame:
    p = LABEL_DIR / f"strategy_labels_{tag}.parquet"
    c = LABEL_DIR / f"strategy_labels_{tag}.csv"
    if p.exists():
        df = pd.read_parquet(p)
    elif c.exists():
        df = pd.read_csv(c)
    else:
        raise FileNotFoundError(f"strategy labels not found for tag={tag}. Run build_strategy_labels.py first.")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["ExitDate"] = pd.to_datetime(df["ExitDate"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    return df


def compute_event_ev(events: pd.DataFrame, win_events: int) -> pd.DataFrame:
    """
    events: ExitDate 기준으로 정렬된 이벤트(각 entry가 exit한 시점에 결과가 확정되는 샘플)
    win_events: 최근 N개 이벤트 기준 롤링 통계(이벤트 수 기준)
    반환: ExitDate, Ticker, EV_* 컬럼
    """
    e = events.sort_values(["Ticker", "ExitDate"]).reset_index(drop=True).copy()

    # 이벤트 시계열(ExitDate가 시간축)
    grp = e.groupby("Ticker", group_keys=False)

    # rolling은 "이벤트 수" 기준
    e[f"EV_ret_mean_{win_events}e"] = grp["CycleReturn"].transform(
        lambda x: x.shift(1).rolling(win_events, min_periods=max(5, win_events // 5)).mean()
    )
    e[f"EV_winrate_{win_events}e"] = grp["CycleReturn"].transform(
        lambda x: (x.shift(1) > 0).rolling(win_events, min_periods=max(5, win_events // 5)).mean()
    )
    e[f"EV_minret_mean_{win_events}e"] = grp["MinCycleRet"].transform(
        lambda x: x.shift(1).rolling(win_events, min_periods=max(5, win_events // 5)).mean()
    )
    e[f"EV_extend_rate_{win_events}e"] = grp["ExtendedFlag"].transform(
        lambda x: x.shift(1).rolling(win_events, min_periods=max(5, win_events // 5)).mean()
    )

    # 이벤트당 1행이므로 그대로 사용
    cols = ["ExitDate", "Ticker"] + [c for c in e.columns if c.startswith("EV_")]
    return e[cols]


def merge_ev_to_daily(features: pd.DataFrame, ev_events: pd.DataFrame) -> pd.DataFrame:
    """
    merge_asof는 키 정렬 + 컬럼 충돌에 민감.
    - 티커별로 split 해서 merge_asof
    - ev_events의 Ticker는 merge 전에 제거(충돌 방지)
    - merge 후 Ticker 컬럼 강제 복구
    """
    features = features.copy()
    ev_events = ev_events.copy()

    def _dt_ns(x: pd.Series) -> pd.Series:
        s = pd.to_datetime(x, errors="coerce")
        try:
            s = s.dt.tz_localize(None)
        except Exception:
            pass
        return s.astype("datetime64[ns]")

    if "Date" not in features.columns or "Ticker" not in features.columns:
        raise ValueError(f"features must have Date,Ticker. cols={list(features.columns)}")

    features["Date"] = _dt_ns(features["Date"])
    features["Ticker"] = features["Ticker"].astype(str).str.upper().str.strip()

    # ev_events datetime key 자동 탐지
    if "EventDate" in ev_events.columns:
        right_key = "EventDate"
    elif "ExitDate" in ev_events.columns:
        right_key = "ExitDate"
    elif "Date" in ev_events.columns:
        right_key = "Date"
    else:
        raise ValueError(f"ev_events has no datetime key col. cols={list(ev_events.columns)}")

    if "Ticker" not in ev_events.columns:
        raise ValueError(f"ev_events must have Ticker. cols={list(ev_events.columns)}")

    ev_events[right_key] = _dt_ns(ev_events[right_key])
    ev_events["Ticker"] = ev_events["Ticker"].astype(str).str.upper().str.strip()

    # 정렬 안정화: NaT 제거 + 중복 제거
    features = (
        features.dropna(subset=["Date", "Ticker"])
        .sort_values(["Ticker", "Date"])
        .drop_duplicates(["Ticker", "Date"], keep="last")
        .reset_index(drop=True)
    )

    ev_events = (
        ev_events.dropna(subset=[right_key, "Ticker"])
        .sort_values(["Ticker", right_key])
        .drop_duplicates(["Ticker", right_key], keep="last")
        .reset_index(drop=True)
    )

    # 티커별 이벤트 맵
    ev_map = {t: g for t, g in ev_events.groupby("Ticker", sort=False)}

    out_parts = []
    for t, f in features.groupby("Ticker", sort=False):
        f = f.sort_values("Date").reset_index(drop=True)

        e = ev_map.get(t)
        if e is None or e.empty:
            out_parts.append(f)
            continue

        # ✅ 충돌 방지: 이벤트쪽 Ticker 제거
        e2 = e.drop(columns=["Ticker"], errors="ignore").sort_values(right_key).reset_index(drop=True)

        merged_t = pd.merge_asof(
            f,
            e2,
            left_on="Date",
            right_on=right_key,
            direction="backward",
            allow_exact_matches=True,
        )

        # ✅ merge 결과에서 Ticker가 깨졌을 때 복구
        if "Ticker" not in merged_t.columns:
            if "Ticker_x" in merged_t.columns:
                merged_t = merged_t.rename(columns={"Ticker_x": "Ticker"})
                if "Ticker_y" in merged_t.columns:
                    merged_t = merged_t.drop(columns=["Ticker_y"])
            else:
                merged_t["Ticker"] = t

        out_parts.append(merged_t)

    merged = pd.concat(out_parts, ignore_index=True)

    # 최종 안전장치: Ticker가 또 없으면 만들기
    if "Ticker" not in merged.columns:
        if "Ticker_x" in merged.columns:
            merged = merged.rename(columns={"Ticker_x": "Ticker"})
            if "Ticker_y" in merged.columns:
                merged = merged.drop(columns=["Ticker_y"])
        else:
            raise RuntimeError(f"merged has no Ticker column. cols={list(merged.columns)}")

    merged = merged.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return merged


def add_cross_sectional_ranks(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    일자별 cross-sectional rank(0~1) 추가: 모델이 '상대적 우위'를 더 잘 잡게 됨.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        out[f"{c}_cs_rank"] = out.groupby("Date")[c].rank(pct=True)
    return out


def save(df: pd.DataFrame) -> str:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(OUT_MODEL_PARQUET, index=False)
        return f"parquet:{OUT_MODEL_PARQUET}"
    except Exception as e:
        print(f"[WARN] parquet save failed ({e}), saving csv: {OUT_MODEL_CSV}")
        df.to_csv(OUT_MODEL_CSV, index=False)
        return f"csv:{OUT_MODEL_CSV}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-tag", type=str, required=True, help="strategy_labels_{tag} 의 tag")
    ap.add_argument("--ev-windows", type=str, default="50,200", help="event window sizes, comma-separated")
    ap.add_argument("--add-cs-rank", action="store_true")
    args = ap.parse_args()

    features = load_features()
    labels = load_strategy_labels(args.label_tag)

    # 이벤트(ExitDate 기준 결과 확정) 테이블
    events = labels.dropna(subset=["CycleReturn", "ExitDate"]).copy()
    events = events[["Ticker", "ExitDate", "CycleReturn", "MinCycleRet", "ExtendedFlag"]]

    win_list = [int(x.strip()) for x in args.ev_windows.split(",") if x.strip()]
    ev_all = []
    for w in win_list:
        ev_all.append(compute_event_ev(events, w))
    ev_events = ev_all[0]
    for x in ev_all[1:]:
        ev_events = ev_events.merge(x, on=["ExitDate", "Ticker"], how="outer")

    merged = merge_ev_to_daily(features, ev_events)

    ev_cols = [c for c in merged.columns if c.startswith("EV_")]
    if args.add_cs_rank:
        merged = add_cross_sectional_ranks(merged, ev_cols)

    saved_to = save(merged)

    meta = {
        "updated_at_utc": now_utc_iso(),
        "saved_to": saved_to,
        "label_tag": args.label_tag,
        "ev_windows": win_list,
        "rows": int(len(merged)),
        "min_date": str(merged["Date"].min().date()),
        "max_date": str(merged["Date"].max().date()),
        "ev_cols": ev_cols,
        "cs_rank_added": bool(args.add_cs_rank),
    }
    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[DONE] features_model saved={saved_to} rows={len(merged)} ev_cols={len(ev_cols)}")
    print(merged.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
