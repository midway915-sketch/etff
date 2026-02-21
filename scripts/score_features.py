# scripts/score_features.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DATA_DIR = Path("data")
FEATURE_DIR = DATA_DIR / "features"
META_DIR = DATA_DIR / "meta"


# build_features.py 기준(16) + (옵션) 섹터(2)
DEFAULT_FEATURES = [
    # base (9)
    "Drawdown_252",
    "Drawdown_60",
    "ATR_ratio",
    "Z_score",
    "MACD_hist",
    "MA20_slope",
    "Market_Drawdown",
    "Market_ATR_ratio",
    "ret_score",
    # new (7)
    "ret_5",
    "ret_10",
    "ret_20",
    "breakout_20",
    "vol_surge",
    "trend_align",
    "beta_60",
    # optional sector (2)
    "Sector_Ret_20",
    "RelStrength",
]


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def ensure_features_exist(df: pd.DataFrame, feat_cols: list[str], *, warn_prefix: str = "") -> pd.DataFrame:
    """
    scoring 단계는 운영 편의상:
      - 컬럼 없으면 0.0으로 생성
      - 단, report/SSOT 기반일 때는 경고 출력
    """
    out = df.copy()
    missing = [c for c in feat_cols if c not in out.columns]
    if missing:
        pfx = f"{warn_prefix} " if warn_prefix else ""
        print(f"[WARN]{pfx}missing feature cols -> filled with 0.0: {missing}")
        for c in missing:
            out[c] = 0.0

    for c in feat_cols:
        out[c] = (
            pd.to_numeric(out[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
    return out


def _load_feature_cols_from_report(report_path: Path) -> list[str] | None:
    if not report_path.exists():
        return None
    try:
        j = json.loads(report_path.read_text(encoding="utf-8"))
        cols = j.get("feature_cols")
        if isinstance(cols, list) and cols:
            return [str(c) for c in cols if str(c).strip()]
    except Exception:
        return None
    return None


def _load_feature_cols_from_ssot() -> list[str] | None:
    """
    data/meta/feature_cols.json (SSOT) 우선 사용.
    형식:
      {"feature_cols":[...], "sector_enabled": true/false, ...}
    """
    p = META_DIR / "feature_cols.json"
    if not p.exists():
        return None
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
        cols = j.get("feature_cols")
        if isinstance(cols, list) and cols:
            return [str(c) for c in cols if str(c).strip()]
    except Exception:
        return None
    return None


def load_ps_feature_cols(tag: str) -> list[str] | None:
    return _load_feature_cols_from_report(META_DIR / f"train_model_report_{tag}.json")


def load_tail_feature_cols(tag: str) -> list[str] | None:
    return _load_feature_cols_from_report(META_DIR / f"train_tail_report_{tag}.json")


def load_tau_feature_cols(tag: str) -> list[str] | None:
    return _load_feature_cols_from_report(META_DIR / f"train_tau_report_{tag}.json")


def parse_tau_h_map(s: str) -> list[int]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    if not parts:
        # ✅ 기본은 30/40/50 (비교 실험 공통 고정)
        return [30, 40, 50]
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(float(p)))
        except Exception:
            pass
    return out if out else [30, 40, 50]


def class_to_h(cls: int, hmap: list[int]) -> int:
    if not hmap:
        return 40
    if cls < 0:
        return hmap[0]
    if cls >= len(hmap):
        return hmap[-1]
    return hmap[cls]


def parse_csv_cols(s: str) -> list[str]:
    return [c.strip() for c in (s or "").split(",") if c.strip()]


def resolve_model_paths(base_model: str, base_scaler: str, tag: str) -> tuple[str, str]:
    """
    tag가 있으면:
      app/model_{tag}.pkl / app/scaler_{tag}.pkl 우선
    없으면:
      주어진 base_model/base_scaler 그대로
    단, tag 파일이 없으면 base로 폴백
    """
    bm = Path(base_model)
    bs = Path(base_scaler)

    if tag:
        cand_m = bm.with_name(f"{bm.stem}_{tag}{bm.suffix}")
        cand_s = bs.with_name(f"{bs.stem}_{tag}{bs.suffix}")
        if cand_m.exists() and cand_s.exists():
            return str(cand_m), str(cand_s)

    return str(bm), str(bs)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score features_model with p_success, p_tail, and tau_class/tau_H -> save features_scored."
    )
    ap.add_argument("--tag", required=True, type=str, help="e.g. pt10_h40_sl10_ex30")

    ap.add_argument("--features-parq", default=str(FEATURE_DIR / "features_model.parquet"), type=str)
    ap.add_argument("--features-csv", default=str(FEATURE_DIR / "features_model.csv"), type=str)

    ap.add_argument("--out-parq", default=str(FEATURE_DIR / "features_scored.parquet"), type=str)
    ap.add_argument("--out-csv", default=str(FEATURE_DIR / "features_scored.csv"), type=str)

    # p_success
    ap.add_argument("--ps-model", default="app/model.pkl", type=str)
    ap.add_argument("--ps-scaler", default="app/scaler.pkl", type=str)
    ap.add_argument(
        "--ps-features",
        default="",
        type=str,
        help="comma-separated override. default=report(train_model_report_{tag}) -> SSOT(meta/feature_cols.json) -> DEFAULT_FEATURES",
    )

    # p_tail
    ap.add_argument("--tail-model", default="app/tail_model.pkl", type=str)
    ap.add_argument("--tail-scaler", default="app/tail_scaler.pkl", type=str)
    ap.add_argument(
        "--tail-features",
        default="",
        type=str,
        help="comma-separated override. default=report(train_tail_report_{tag}) -> SSOT -> ps-features",
    )

    # tau
    ap.add_argument("--tau-model", default="app/tau_model.pkl", type=str)
    ap.add_argument("--tau-scaler", default="app/tau_scaler.pkl", type=str)
    ap.add_argument(
        "--tau-features",
        default="",
        type=str,
        help="comma-separated override. default=report(train_tau_report_{tag}) -> SSOT -> ps-features",
    )
    ap.add_argument(
        "--tau-h-map",
        default="30,40,50",  # ✅ 비교 실험 공통: 30/40/50 고정
        type=str,
        help="tau_class->H mapping. e.g. '30,40,50' means class0=30 class1=40 class2=50",
    )

    args = ap.parse_args()

    feats = read_table(Path(args.features_parq), Path(args.features_csv)).copy()
    print("[DEBUG] features_model cols(head):", list(feats.columns)[:30], " index.name:", feats.index.name)

    if "Date" not in feats.columns or "Ticker" not in feats.columns:
        raise ValueError("features_model must contain Date,Ticker")

    feats["Date"] = norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    ssot_cols = _load_feature_cols_from_ssot()

    # -------------------------
    # p_success (report -> SSOT -> DEFAULT)
    # -------------------------
    ps_model_path, ps_scaler_path = resolve_model_paths(args.ps_model, args.ps_scaler, args.tag)
    if not Path(ps_model_path).exists() or not Path(ps_scaler_path).exists():
        raise FileNotFoundError(
            f"Missing p_success model/scaler.\n"
            f"-> tried: {ps_model_path} / {ps_scaler_path}\n"
            f"-> (tag fallback) you may need to run train_model.py first."
        )

    ps_model = joblib.load(ps_model_path)
    ps_scaler = joblib.load(ps_scaler_path)

    if args.ps_features.strip():
        ps_cols = parse_csv_cols(args.ps_features)
        ps_cols_src = "--ps-features"
    else:
        ps_cols = load_ps_feature_cols(args.tag)
        if ps_cols:
            ps_cols_src = f"report(train_model_report_{args.tag}.json)"
        elif ssot_cols:
            ps_cols = ssot_cols
            ps_cols_src = "SSOT(data/meta/feature_cols.json)"
        else:
            ps_cols = DEFAULT_FEATURES
            ps_cols_src = "DEFAULT_FEATURES"

    ps_cols = [c for c in ps_cols if c]
    feats_ps = ensure_features_exist(feats, ps_cols, warn_prefix=f"[p_success:{ps_cols_src}]")
    X_ps = feats_ps[ps_cols].to_numpy(dtype=float)
    X_ps_s = ps_scaler.transform(X_ps)
    feats["p_success"] = ps_model.predict_proba(X_ps_s)[:, 1].astype(float)

    # -------------------------
    # p_tail (optional; report -> SSOT -> ps_cols)
    # -------------------------
    tail_model_path = Path(args.tail_model)
    tail_scaler_path = Path(args.tail_scaler)
    if tail_model_path.exists() and tail_scaler_path.exists():
        tail_model = joblib.load(tail_model_path)
        tail_scaler = joblib.load(tail_scaler_path)

        if args.tail_features.strip():
            tail_cols = parse_csv_cols(args.tail_features)
            tail_cols_src = "--tail-features"
        else:
            tail_cols = load_tail_feature_cols(args.tag)
            if tail_cols:
                tail_cols_src = f"report(train_tail_report_{args.tag}.json)"
            elif ssot_cols:
                tail_cols = ssot_cols
                tail_cols_src = "SSOT(data/meta/feature_cols.json)"
            else:
                tail_cols = ps_cols
                tail_cols_src = "fallback(ps_cols)"

        tail_cols = [c for c in tail_cols if c]
        feats_tail = ensure_features_exist(feats, tail_cols, warn_prefix=f"[p_tail:{tail_cols_src}]")
        X_t = feats_tail[tail_cols].to_numpy(dtype=float)
        X_t_s = tail_scaler.transform(X_t)
        feats["p_tail"] = tail_model.predict_proba(X_t_s)[:, 1].astype(float)
    else:
        feats["p_tail"] = 0.0

    # -------------------------
    # tau_class / tau_H (optional; report -> SSOT -> ps_cols)
    # -------------------------
    tau_model_path = Path(args.tau_model)
    tau_scaler_path = Path(args.tau_scaler)
    hmap = parse_tau_h_map(args.tau_h_map)

    if tau_model_path.exists() and tau_scaler_path.exists():
        tau_model = joblib.load(tau_model_path)
        tau_scaler = joblib.load(tau_scaler_path)

        if args.tau_features.strip():
            tau_cols = parse_csv_cols(args.tau_features)
            tau_cols_src = "--tau-features"
        else:
            tau_cols = load_tau_feature_cols(args.tag)
            if tau_cols:
                tau_cols_src = f"report(train_tau_report_{args.tag}.json)"
            elif ssot_cols:
                tau_cols = ssot_cols
                tau_cols_src = "SSOT(data/meta/feature_cols.json)"
            else:
                tau_cols = ps_cols
                tau_cols_src = "fallback(ps_cols)"

        tau_cols = [c for c in tau_cols if c]
        feats_tau = ensure_features_exist(feats, tau_cols, warn_prefix=f"[tau:{tau_cols_src}]")
        X_tau = feats_tau[tau_cols].to_numpy(dtype=float)
        X_tau_s = tau_scaler.transform(X_tau)

        try:
            tau_class = tau_model.predict(X_tau_s)
            tau_class = np.asarray(tau_class).astype(int)
        except Exception:
            proba = tau_model.predict_proba(X_tau_s)
            tau_class = np.argmax(proba, axis=1).astype(int)

        feats["tau_class"] = tau_class.astype(int)

        try:
            proba = tau_model.predict_proba(X_tau_s)
            feats["tau_pmax"] = np.max(proba, axis=1).astype(float)
        except Exception:
            feats["tau_pmax"] = 0.0

        feats["tau_H"] = feats["tau_class"].apply(lambda x: class_to_h(int(x), hmap)).astype(int)
    else:
        feats["tau_class"] = -1
        feats["tau_pmax"] = 0.0
        # ✅ tau 비활성 폴백은 '중앙 H'가 더 안전(비교 실험/운영 둘 다)
        if hmap and len(hmap) >= 2:
            feats["tau_H"] = int(hmap[1])  # 40
        else:
            feats["tau_H"] = 40

    # -------------------------
    # write outputs
    # -------------------------
    out_parq = Path(args.out_parq)
    out_csv = Path(args.out_csv)
    out_parq.parent.mkdir(parents=True, exist_ok=True)

    try:
        feats.to_parquet(out_parq, index=False)
        print(f"[DONE] wrote: {out_parq} rows={len(feats)}")
    except Exception as e:
        print(f"[WARN] parquet save failed ({e}) -> writing csv")
        feats.to_csv(out_csv, index=False)
        print(f"[DONE] wrote: {out_csv} rows={len(feats)}")

    print("=" * 60)
    print("[INFO] scoring summary")
    print("tag:", args.tag)
    print("p_success model/scaler:", ps_model_path, "/", ps_scaler_path)
    print("p_success cols source:", ps_cols_src, "cols:", ps_cols)
    print("p_tail enabled:", bool(tail_model_path.exists() and tail_scaler_path.exists()))
    print("tau enabled:", bool(tau_model_path.exists() and tau_scaler_path.exists()))
    print("tau_h_map:", hmap)
    print("=" * 60)


if __name__ == "__main__":
    main()