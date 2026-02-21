#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# sys.path guard (avoid "No module named 'scripts'")
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FEAT_DIR = DATA_DIR / "features"
LABEL_DIR = DATA_DIR / "labels"
META_DIR = DATA_DIR / "meta"

PRICES_PARQ = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

# ✅ 강제: tail labels는 features_model 기준으로만 align
FEATURES_MODEL_PARQ = FEAT_DIR / "features_model.parquet"
FEATURES_MODEL_CSV = FEAT_DIR / "features_model.csv"


# ------------------------------------------------------------
# SSOT feature cols resolution (forced 18 cols, consistent with feature_spec)
# ------------------------------------------------------------
def _validate_18_cols(cols: list[str], src: str) -> None:
    if len(cols) != 18:
        raise RuntimeError(f"SSOT feature cols must be 18, got {len(cols)} from {src}: {cols}")

    # ✅ 현재 SSOT: Sector_Ret_20 제거, RelStrength는 필수
    need = ["RelStrength"]
    miss = [c for c in need if c not in cols]
    if miss:
        raise RuntimeError(
            f"SSOT feature cols must include required features {need}. Missing={miss}. src={src}"
        )


def resolve_feature_cols_forced(args_feature_cols: str) -> tuple[list[str], str]:
    """
    ✅ 18개 SSOT 강제 (Sector_Ret_20 제거 반영)

    규칙:
      - --feature-cols override는 허용 안 함(라벨/모델 불일치 방지)
      - SSOT(meta/feature_cols.json) 있으면 그걸 사용
      - 없으면 get_feature_cols(sector_enabled=False) 사용
      - 무엇을 쓰든 반드시:
          (1) len == 18
          (2) RelStrength 포함
    """
    if str(args_feature_cols or "").strip():
        raise ValueError(
            "--feature-cols override is not allowed when SSOT 18 features are enforced.\n"
            "Remove --feature-cols and rely on SSOT/meta + feature_spec."
        )

    try:
        from scripts.feature_spec import read_feature_cols_meta, get_feature_cols  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import scripts.feature_spec: {e}")

    cols, _sector_enabled = read_feature_cols_meta()
    if cols:
        src = "data/meta/feature_cols.json"
        _validate_18_cols(cols, src)
        return cols, src

    cols = get_feature_cols(sector_enabled=False)
    src = "scripts/feature_spec.py:get_feature_cols(sector_enabled=False)"
    _validate_18_cols(cols, src)
    return cols, src


# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------
def read_prices() -> pd.DataFrame:
    if PRICES_PARQ.exists():
        df = pd.read_parquet(PRICES_PARQ)
    elif PRICES_CSV.exists():
        df = pd.read_csv(PRICES_CSV)
    else:
        raise FileNotFoundError(f"Missing prices: {PRICES_PARQ} (or {PRICES_CSV})")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.dropna(subset=["Date", "Ticker", "Close", "High", "Low"])
        .sort_values(["Ticker", "Date"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )
    return df


def read_features_model() -> tuple[pd.DataFrame, str]:
    """
    ✅ 강제: features_model만 사용 (features_scored 절대 사용 안 함)
    """
    if FEATURES_MODEL_PARQ.exists():
        df = pd.read_parquet(FEATURES_MODEL_PARQ).copy()
        src = str(FEATURES_MODEL_PARQ)
    elif FEATURES_MODEL_CSV.exists():
        df = pd.read_csv(FEATURES_MODEL_CSV).copy()
        src = str(FEATURES_MODEL_CSV)
    else:
        raise FileNotFoundError(
            "Missing features_model. Expected one of:\n"
            f"- {FEATURES_MODEL_PARQ}\n"
            f"- {FEATURES_MODEL_CSV}\n"
        )

    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError(f"features_model missing Date/Ticker: {src}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["Date", "Ticker"]).reset_index(drop=True)
    return df, src


# ------------------------------------------------------------
# Tail label logic
# ------------------------------------------------------------
def compute_trade_path(
    g: pd.DataFrame,
    profit_target: float,
    max_days: int,
    stop_level: float,
    max_extend_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tail definition:
      - If SL hit first within H, and then TP is reached within extension window => p_tail=1
      - else p_tail=0

    Returns:
      y_tail (0/1)
      y_pmax: max profit within extension window (diagnostic)
    """
    g = g.sort_values("Date").reset_index(drop=True)
    close = pd.to_numeric(g["Close"], errors="coerce").to_numpy(dtype=float)

    n = len(g)
    y_tail = np.zeros(n, dtype=int)
    y_pmax = np.full(n, np.nan, dtype=float)

    H = int(max_days)
    EX = int(max_extend_days)

    for i in range(n):
        if i + 1 >= n:
            continue
        entry = close[i]
        if not np.isfinite(entry) or entry <= 0:
            continue

        end_main = min(n - 1, i + H)
        win = close[i + 1 : end_main + 1]

        tp_hit = None
        sl_hit = None
        for j, px in enumerate(win, start=1):
            ret = (px / entry) - 1.0
            if tp_hit is None and ret >= profit_target:
                tp_hit = i + j
            if sl_hit is None and ret <= stop_level:
                sl_hit = i + j
            if tp_hit is not None or sl_hit is not None:
                break

        end_ext = min(n - 1, end_main + EX)
        win_ext = close[i + 1 : end_ext + 1]
        if len(win_ext) > 0:
            y_pmax[i] = float(np.nanmax((win_ext / entry) - 1.0))

        # already success or no event: tail=0
        if tp_hit is None and sl_hit is None:
            y_tail[i] = 0
        elif tp_hit is not None and (sl_hit is None or tp_hit <= sl_hit):
            y_tail[i] = 0
        else:
            # SL first -> check recovery to TP within extension
            recovered = False
            start = (sl_hit + 1) if sl_hit is not None else (end_main + 1)
            if start <= end_ext:
                for k in range(start, end_ext + 1):
                    if ((close[k] / entry) - 1.0) >= profit_target:
                        recovered = True
                        break
            y_tail[i] = 1 if recovered else 0

    return y_tail, y_pmax


def main() -> None:
    ap = argparse.ArgumentParser(description="Build tail labels (p_tail) aligned to features_model only.")
    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)

    ap.add_argument("--feature-cols", type=str, default="", help="(DISALLOWED) override; kept for compatibility")
    args = ap.parse_args()

    pt = float(args.profit_target)
    H = int(args.max_days)
    sl = float(args.stop_level)
    ex = int(args.max_extend_days)

    cols, src = resolve_feature_cols_forced(args.feature_cols)
    print(f"[INFO] SSOT feature cols source={src} cols={cols}")

    prices = read_prices()
    feats, feats_src = read_features_model()
    print(f"[INFO] features source={feats_src} rows={len(feats)}")

    pt100 = int(round(pt * 100))
    sl100 = int(round(abs(sl) * 100))
    tag = f"pt{pt100}_h{H}_sl{sl100}_ex{ex}"

    out_parq = LABEL_DIR / f"labels_tail_{tag}.parquet"
    out_csv = LABEL_DIR / f"labels_tail_{tag}.csv"
    meta_json = LABEL_DIR / f"labels_tail_{tag}_meta.json"

    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    y_list = []
    pmax_list = []
    key_list = []

    for tkr, g in prices.groupby("Ticker", sort=False):
        y_tail, y_pmax = compute_trade_path(
            g,
            profit_target=pt,
            max_days=H,
            stop_level=sl,
            max_extend_days=ex,
        )
        dts = pd.to_datetime(g["Date"], errors="coerce").dt.tz_localize(None).to_numpy()
        tkrs = np.array([tkr] * len(g))
        key_list.append(pd.DataFrame({"Date": dts, "Ticker": tkrs}))
        y_list.append(pd.Series(y_tail))
        pmax_list.append(pd.Series(y_pmax))

    keys = pd.concat(key_list, ignore_index=True)
    y_tail_all = pd.concat(y_list, ignore_index=True).to_numpy(dtype=int)
    y_pmax_all = pd.concat(pmax_list, ignore_index=True).to_numpy(dtype=float)

    labels = keys.copy()
    labels["p_tail"] = y_tail_all
    labels["tail_pmax"] = y_pmax_all

    labels["Date"] = pd.to_datetime(labels["Date"], errors="coerce").dt.tz_localize(None)
    labels["Ticker"] = labels["Ticker"].astype(str).str.upper().str.strip()
    labels = labels.dropna(subset=["Date", "Ticker"]).reset_index(drop=True)

    # ✅ Align to features_model keys only
    merged = feats[["Date", "Ticker"]].merge(labels, on=["Date", "Ticker"], how="inner")
    if merged.empty:
        raise RuntimeError(
            "No overlap between features_model and computed tail labels. "
            "Check universe tickers & date ranges."
        )

    out = merged[["Date", "Ticker", "p_tail", "tail_pmax"]].copy()
    out = out.sort_values(["Date", "Ticker"]).drop_duplicates(["Date", "Ticker"], keep="last").reset_index(drop=True)

    try:
        out.to_parquet(out_parq, index=False)
        print(f"[DONE] wrote parquet: {out_parq} rows={len(out)}")
    except Exception as e:
        print(f"[WARN] parquet save failed ({e}) -> writing csv only")

    out.to_csv(out_csv, index=False)
    print(f"[DONE] wrote csv: {out_csv} rows={len(out)}")

    meta = {
        "tag": tag,
        "profit_target": pt,
        "max_days": H,
        "stop_level": sl,
        "max_extend_days": ex,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "ssot_feature_cols_source": src,
        "ssot_feature_cols": cols,
        "features_source_used": feats_src,
        "label_cols": ["p_tail", "tail_pmax"],
        "notes": "features_model only. Sector_Ret_20 removed; RelStrength required.",
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[DONE] wrote meta: {meta_json}")


if __name__ == "__main__":
    main()
