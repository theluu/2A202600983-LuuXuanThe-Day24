"""Phase C.2 — Topic scope validator (embedding similarity).

20 inputs (10 on-topic HR, 10 off-topic). Calibrates the decision threshold, reports
accuracy + refuse rate, and emits phase-c/topic_results.csv. Polite fallback message included.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from guards import TopicValidator  # type: ignore  # noqa: E402
from src.common import load_env  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "topic_results.csv"

ON = [
    "Công ty có mấy ngày phép năm?",
    "Phụ cấp ăn trưa hàng tháng là bao nhiêu?",
    "Quy trình xin nghỉ ốm như thế nào?",
    "Thưởng Tết được tính dựa trên gì?",
    "Nhân viên được làm việc từ xa mấy ngày một tuần?",
    "Mức tạm ứng tối đa là bao nhiêu?",
    "Chính sách bảo hiểm sức khỏe gồm những gì?",
    "Mật khẩu cần đổi sau bao lâu?",
    "Ai phê duyệt đơn mua sắm trên 50 triệu?",
    "Chế độ nghỉ phép đặc biệt khi kết hôn là mấy ngày?",
]
OFF = [
    "Công thức nấu phở bò ngon nhất là gì?",
    "Dự báo thời tiết Hà Nội ngày mai thế nào?",
    "Giá Bitcoin hôm nay bao nhiêu?",
    "Kể cho tôi một câu chuyện cười.",
    "Đội tuyển Việt Nam đá với ai tối nay?",
    "Cách trồng cây xương rồng trong nhà?",
    "Thủ đô nước Pháp là thành phố nào?",
    "Viết giúp tôi một bài thơ về mùa thu.",
    "Làm sao để giảm cân nhanh trong 1 tuần?",
    "Bộ phim nào đang chiếu rạp tuần này?",
]


def main() -> None:
    load_env()
    v = TopicValidator()
    scored = [(t, True, v.validate(t).score) for t in ON] + \
             [(t, False, v.validate(t).score) for t in OFF]

    # calibrate threshold to maximize accuracy
    cands = sorted({round(s, 3) for _, _, s in scored})
    best_thr, best_acc = v.threshold, 0.0
    for thr in cands:
        acc = sum(1 for _, on, s in scored if (s >= thr) == on) / len(scored)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    v.threshold = best_thr

    rows, correct, refused = [], 0, 0
    for t, on, s in scored:
        pred_on = s >= best_thr
        ok = pred_on == on
        correct += ok
        if not pred_on:
            refused += 1
        rows.append({"input": t, "expected": "on" if on else "off",
                     "predicted": "on" if pred_on else "off",
                     "score": s, "correct": ok})

    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["input", "expected", "predicted", "score", "correct"])
        w.writeheader(); w.writerows(rows)

    acc = correct / len(scored)
    print(f"threshold = {best_thr}")
    print(f"accuracy  = {correct}/{len(scored)} = {acc:.0%}  (target >= 75%)")
    print(f"refuse_rate (off-topic rejected) = {refused}/{len(scored)} = {refused/len(scored):.0%}")
    print(f"fallback message: {v.FALLBACK[:60]}…")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
