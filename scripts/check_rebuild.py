# scripts/check_rebuild.py
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    return sha256_bytes(path.read_bytes())


def sha256_dir(paths: list[Path]) -> str:
    items = []
    for p in sorted(paths, key=lambda x: str(x)):
        items.append(f"{p}:{sha256_file(p)}")
    return sha256_bytes("\n".join(items).encode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprint-path", default="data/meta/build_fingerprint.json")
    ap.add_argument("--label-spec", default="")  # ex) pt=0.10,h=40,sl=-0.10,ex=30

    ap.add_argument("--features-files", default="")
    ap.add_argument("--labels-files", default="")
    ap.add_argument("--universe-files", default="")
    ap.add_argument("--strategy-files", default="")
    args = ap.parse_args()

    fp_path = Path(args.fingerprint_path)
    fp_path.parent.mkdir(parents=True, exist_ok=True)

    def parse_list(s: str) -> list[Path]:
        s = (s or "").strip()
        if not s:
            return []
        return [Path(x.strip()) for x in s.split(",") if x.strip()]

    feat_files = parse_list(args.features_files)
    label_files = parse_list(args.labels_files)
    uni_files = parse_list(args.universe_files)
    strat_files = parse_list(args.strategy_files)

    new_fp = {
        "features_code_hash": sha256_dir(feat_files),
        "labels_code_hash": sha256_dir(label_files),
        "universe_hash": sha256_dir(uni_files),
        "strategy_code_hash": sha256_dir(strat_files),
        "label_spec": args.label_spec.strip(),
    }

    old = None
    if fp_path.exists():
        try:
            old = json.loads(fp_path.read_text(encoding="utf-8"))
        except Exception:
            old = None

    changed = {
        "features_code_hash": False,
        "labels_code_hash": False,
        "universe_hash": False,
        "strategy_code_hash": False,
        "label_spec": False,
        "no_previous_fingerprint": False,
    }

    if old is None:
        changed["no_previous_fingerprint"] = True
        changed["features_code_hash"] = True
        changed["labels_code_hash"] = True
        changed["universe_hash"] = True
        changed["strategy_code_hash"] = True
        changed["label_spec"] = True
    else:
        for k in ["features_code_hash", "labels_code_hash", "universe_hash", "strategy_code_hash", "label_spec"]:
            changed[k] = str(old.get(k, "")) != str(new_fp.get(k, ""))

    # 부분 리빌드 정책
    rebuild_prices = changed["universe_hash"] or changed["no_previous_fingerprint"]
    rebuild_features = rebuild_prices or changed["features_code_hash"]
    rebuild_model = rebuild_features or changed["labels_code_hash"] or changed["label_spec"]

    # ✅ signals는 “전략 코드 변경” 또는 “어떤 rebuild라도 발생” 시 초기화 권장
    clear_signals = (
        changed["strategy_code_hash"]
        or rebuild_prices
        or rebuild_features
        or rebuild_model
    )

    reason = []
    if changed["no_previous_fingerprint"]:
        reason.append("no_previous_fingerprint")
    for k in ["universe_hash", "features_code_hash", "labels_code_hash", "strategy_code_hash", "label_spec"]:
        if changed[k]:
            reason.append(f"changed:{k}")

    out = {
        "rebuild_prices": rebuild_prices,
        "rebuild_features": rebuild_features,
        "rebuild_model": rebuild_model,
        "clear_signals": clear_signals,
        "changed": changed,
        "reason": reason,
        "fingerprint_path": str(fp_path),
    }

    fp_path.write_text(json.dumps(new_fp, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

    # GHA env lines (마지막 4줄 고정)
    print(f"REBUILD_PRICES={'1' if rebuild_prices else '0'}")
    print(f"REBUILD_FEATURES={'1' if rebuild_features else '0'}")
    print(f"REBUILD_MODEL={'1' if rebuild_model else '0'}")
    print(f"CLEAR_SIGNALS={'1' if clear_signals else '0'}")


if __name__ == "__main__":
    main()