# scripts/feature_spec.py
from __future__ import annotations

import json
from pathlib import Path


META_DIR = Path("data/meta")
FEATURE_COLS_JSON = META_DIR / "feature_cols.json"


def get_feature_cols(sector_enabled: bool = False) -> list[str]:
    """
    SSOT: feature column order (18 columns)

    ✅ 변경사항
    - Sector_Ret_20 제거 (섹터/그룹 집계 기반 피처 제거)
    - Market_ret_20 추가 (시장 20일 수익률)
    - RelStrength는 "시장(UPRO) 기준 상대강도"로 쓰되 컬럼명은 그대로 RelStrength 유지
      -> RelStrength = ret_20 - ret_20(UPRO)
    """
    return [
        "Drawdown_252",
        "Drawdown_60",
        "ATR_ratio",
        "Z_score",
        "MACD_hist",
        "MA20_slope",
        "Market_Drawdown",
        "Market_ATR_ratio",
        "Market_ret_20",   # ✅ NEW (SPY 기준)
        "ret_score",
        "ret_5",
        "ret_10",
        "ret_20",
        "breakout_20",
        "vol_surge",
        "trend_align",
        "beta_60",
        "RelStrength",     # ✅ now = ret_20 - UPRO_ret_20 (name kept)
    ]


def write_feature_cols_meta(cols: list[str], sector_enabled: bool) -> Path:
    META_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_cols": cols,
        "sector_enabled": bool(sector_enabled),
        "ssot_version": "v3_relstrength_upro",
    }
    FEATURE_COLS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return FEATURE_COLS_JSON


def read_feature_cols_meta() -> tuple[list[str], bool]:
    if not FEATURE_COLS_JSON.exists():
        return ([], False)
    try:
        payload = json.loads(FEATURE_COLS_JSON.read_text(encoding="utf-8"))
        cols = payload.get("feature_cols", [])
        sector_enabled = bool(payload.get("sector_enabled", False))
        if isinstance(cols, list) and all(isinstance(x, str) for x in cols):
            return (cols, sector_enabled)
        return ([], sector_enabled)
    except Exception:
        return ([], False)