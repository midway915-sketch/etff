# universe.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

# =========================================
# Universe definition (edit here)
# =========================================

UNIVERSE = [
    # Ticker, Group, Provider, Type(ETF/ETN), Direction(Bull/Bear), Leverage
    {"Ticker": "SOXL", "Group": "Semiconductors", "Provider": "Direxion", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "BULZ", "Group": "Tech",          "Provider": "MicroSectors/BMO", "Type": "ETN", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "TQQQ", "Group": "Nasdaq100",     "Provider": "ProShares", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "TECL", "Group": "Tech",          "Provider": "Direxion", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "WEBL", "Group": "Internet",      "Provider": "Direxion", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "UPRO", "Group": "SP500",         "Provider": "ProShares", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "WANT", "Group": "ConsumerDisc",  "Provider": "Direxion", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "HIBL", "Group": "HighBeta",      "Provider": "Direxion", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "FNGU", "Group": "FANG",          "Provider": "MicroSectors/BMO", "Type": "ETN", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "TNA",  "Group": "Russell2000",   "Provider": "Direxion", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "RETL", "Group": "Retail",        "Provider": "Direxion", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "UDOW", "Group": "Dow",           "Provider": "ProShares", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "NAIL", "Group": "Homebuilders",  "Provider": "Direxion", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "LABU", "Group": "Biotech",       "Provider": "Direxion", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "PILL", "Group": "Healthcare",    "Provider": "Direxion", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "MIDU", "Group": "Midcap",        "Provider": "Direxion", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "CURE", "Group": "Healthcare",    "Provider": "Direxion", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "FAS",  "Group": "Financials",    "Provider": "Direxion", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "TPOR", "Group": "Transports",    "Provider": "Direxion", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "DRN",  "Group": "REITs",         "Provider": "Direxion", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "DUSL", "Group": "Industrials",   "Provider": "Direxion", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "DFEN", "Group": "Defense",       "Provider": "Direxion", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "UTSL", "Group": "Utilities",     "Provider": "Direxion", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "BNKU", "Group": "Banks",         "Provider": "MicroSectors/BMO", "Type": "ETN", "Direction": "Bull", "Leverage": 3},
    {"Ticker": "DPST", "Group": "RegionalBanks", "Provider": "Direxion", "Type": "ETF", "Direction": "Bull", "Leverage": 3},
]

DEFAULT_COLUMNS = {
    # 운영/필터용 기본값 (나중에 eligibility filter에서 그대로 씀)
    "Enabled": True,
    "Currency": "USD",
    "Region": "US",
    "MinHistoryDays": 756,          # 대략 3년(거래일)
    "MinAvgDollarVol20": 2_000_000, # 20일 평균 거래대금 하한 (원하면 조정)
    "Notes": "",
}

OUTPUT_PATH = Path("data/universe.csv")


def build_universe_df() -> pd.DataFrame:
    df = pd.DataFrame(UNIVERSE).copy()

    # 기본 컬럼 채우기
    for col, val in DEFAULT_COLUMNS.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = df[col].fillna(val)

    # 최소 검증
    required = ["Ticker", "Type", "Direction", "Leverage"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()

    # 중복 티커 방지
    dup = df["Ticker"][df["Ticker"].duplicated()].tolist()
    if dup:
        raise ValueError(f"Duplicate tickers found: {dup}")

    # 정렬(가독성)
    sort_cols = [c for c in ["Enabled", "Type", "Group", "Ticker"] if c in df.columns]
    df = df.sort_values(sort_cols, ascending=[False, True, True, True]).reset_index(drop=True)

    return df


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = build_universe_df()
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}  (rows={len(df)})")
    print(df[["Ticker", "Type", "Direction", "Leverage", "Group", "Provider"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
