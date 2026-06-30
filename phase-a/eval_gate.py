"""Phase A.4 — CI eval gate.

Reads phase-a/ragas_summary.json and fails (exit 1) if any metric is below its
target threshold. Used by .github/workflows/eval-gate.yml to block merges.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SUMMARY = Path(__file__).resolve().parent / "ragas_summary.json"

THRESHOLDS = {
    "faithfulness": 0.60,
    "context_precision": 0.70,
    "context_recall": 0.70,
    "avg_score": 0.60,
}


def main() -> int:
    if not SUMMARY.exists():
        print(f"::error::{SUMMARY} not found — run phase-a/a2_run_ragas.py first")
        return 1
    overall = json.loads(SUMMARY.read_text(encoding="utf-8"))["overall"]
    failures = []
    print("RAGAS eval gate")
    for metric, target in THRESHOLDS.items():
        value = float(overall.get(metric, 0.0))
        ok = value >= target
        print(f"  {'PASS' if ok else 'FAIL'}  {metric:18} = {value:.3f}  (target >= {target})")
        if not ok:
            failures.append(f"{metric}={value:.3f} < {target}")
    if failures:
        print("::error::eval gate failed: " + "; ".join(failures))
        return 1
    print("eval gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
