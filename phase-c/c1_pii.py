"""Phase C.1 — PII redaction test (Presidio + VN regex).

10 mixed EN+VN inputs (+ empty / very long edge cases). Reports detection rate and
latency P95. Output: phase-c/pii_test_results.csv (input, output, pii_found, latency_ms).
"""
from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from guards import PIIScanner  # type: ignore  # noqa: E402
from src.common import load_env  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "pii_test_results.csv"

# (text, expected_entity_present)
CASES = [
    ("CCCD của tôi là 012345678901, gọi tôi qua 0987654321.", True),
    ("My email is john.doe@company.com, please reply.", True),
    ("Số điện thoại liên hệ: 0912345678.", True),
    ("Mã số thuế công ty là 0312345678 theo hồ sơ.", True),
    ("Contact Nguyen at nguyen.van.a@corp.vn or +84987654321.", True),
    ("Nhân viên Trần Thị B sống tại 123 Lê Lợi, Quận 1.", True),
    ("Please update my CMND 123456789 in the system.", True),
    ("Gửi lương về số 0356789012 nhé anh.", True),
    ("", False),                                   # edge: empty
    ("Chính sách nghỉ phép năm là 15 ngày. " * 200 + " Liên hệ 0987001002.", True),  # edge: very long
]


def main() -> None:
    load_env()
    scanner = PIIScanner()
    scanner.scan("warmup 0987654321")  # load Presidio/spaCy before timing

    rows, hits = [], 0
    expected_total = sum(1 for _, e in CASES if e)
    for text, expected in CASES:
        t0 = time.perf_counter()
        res = scanner.scan(text)
        dt = (time.perf_counter() - t0) * 1000
        if expected and res.has_pii:
            hits += 1
        rows.append({
            "input": text[:80] + ("…" if len(text) > 80 else ""),
            "output": res.anonymized[:80] + ("…" if len(res.anonymized) > 80 else ""),
            "pii_found": ";".join(res.entities),
            "latency_ms": round(dt, 2),
        })

    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["input", "output", "pii_found", "latency_ms"])
        w.writeheader(); w.writerows(rows)

    # latency P95 over realistic short inputs (exclude empty + pathological long case)
    normal = [t for t, _ in CASES if 0 < len(t) <= 200]
    lats = []
    for _ in range(6):
        for t in normal:
            t0 = time.perf_counter(); scanner.scan(t); lats.append((time.perf_counter() - t0) * 1000)
    p95 = statistics.quantiles(lats, n=20)[18]
    long_lat = next(r["latency_ms"] for r, (t, _) in zip(rows, CASES) if len(t) > 1000)

    detection_rate = hits / expected_total
    print(f"detection_rate = {hits}/{expected_total} = {detection_rate:.0%}  (target >= 80%)")
    print(f"latency (short)  P50={statistics.median(lats):.2f}ms  P95={p95:.2f}ms  (target P95 < 50ms)")
    print(f"robustness: empty input OK, very-long input ({long_lat:.0f}ms) handled without error")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
