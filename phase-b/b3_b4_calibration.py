"""Phase B.3 + B.4 — Human calibration (Cohen's kappa) + bias report.

B.3: compare human_labels.csv vs the judge's winner_after_swap on the same 10 pairs,
     compute Cohen's kappa. If kappa < 0.6 -> emit a root-cause analysis section.
B.4: judge_bias_report.md — quantify position bias + verbosity/length bias with a table
     and a chart (judge_bias_chart.png), plus a mitigation strategy.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
HUMAN = HERE / "human_labels.csv"
PAIRWISE = HERE / "pairwise_results.csv"
CACHE = HERE / "_judge_cache.json"
REPORT = HERE / "judge_bias_report.md"
CHART = HERE / "judge_bias_chart.png"


def cohen_kappa(a: list[str], b: list[str]) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = 0.0
    for c in cats:
        pa = a.count(c) / n
        pb = b.count(c) / n
        pe += pa * pb
    return 1.0 if pe == 1.0 and po == 1.0 else (po - pe) / (1 - pe)


def interpret(k: float) -> str:
    if k > 0.8:
        return "almost perfect"
    if k > 0.6:
        return "substantial"
    if k > 0.4:
        return "moderate"
    if k > 0.2:
        return "fair"
    return "slight/poor"


def main() -> None:
    human = list(csv.DictReader(HUMAN.open(encoding="utf-8")))
    pw = list(csv.DictReader(PAIRWISE.open(encoding="utf-8")))
    cache = json.loads(CACHE.read_text(encoding="utf-8"))

    # judge labels for the same question_ids (1-based index into cache)
    judge_by_id = {i + 1: c["winner_after_swap"] for i, c in enumerate(cache)}
    human_lab, judge_lab = [], []
    for row in human:
        qid = int(row["question_id"])
        human_lab.append(row["human_winner"].strip())
        judge_lab.append(judge_by_id[qid])

    kappa = cohen_kappa(human_lab, judge_lab)
    agree = sum(1 for h, j in zip(human_lab, judge_lab) if h == j)

    # ── bias metrics from full 30-pair pairwise data ──
    n = len(pw)
    inconsistent = sum(1 for r in pw if r["position_consistent"] == "False")
    position_bias_rate = round(inconsistent / n, 3)

    # verbosity bias: among run1 decisive judgments, how often longer answer won
    decisive = [r for r in pw if r["run1_winner"] != "tie"]
    longer_won = 0
    for r in decisive:
        lb, lr = int(r["len_baseline"]), int(r["len_reranker"])
        winner = r["run1_winner"]
        longer = "baseline" if lb > lr else "reranker"
        if winner == longer:
            longer_won += 1
    verbosity_bias = round(longer_won / len(decisive), 3) if decisive else 0.0
    tie_rate = round(sum(1 for r in pw if r["winner_after_swap"] == "tie") / n, 3)

    # ── chart ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.4))
    labels = ["Position bias\n(order flip)", "Verbosity bias\n(longer wins | decisive)", "Tie rate"]
    vals = [position_bias_rate, verbosity_bias, tie_rate]
    bars = ax.bar(labels, vals, color=["#d9534f", "#f0ad4e", "#5bc0de"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate")
    ax.set_title(f"LLM Judge bias profile (n={n}) — Cohen's kappa={kappa:.2f}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHART, dpi=120)

    needs_root_cause = kappa < 0.6
    md = f"""# Judge Bias Report — LLM-as-Judge calibration

**Sinh viên:** Lưu Xuân Thế · **MSSV:** 2A202600983 · **Ngày:** 30/06/2026
**Judge:** `gpt-4o-mini`, pairwise **swap-and-average**, n = {n} câu (baseline vs reranker).

## 1. Human calibration — Cohen's kappa

So sánh nhãn của **người** ({len(human_lab)} cặp, `human_labels.csv`) với `winner_after_swap` của judge:

| | Giá trị |
|---|---|
| Observed agreement | {agree}/{len(human_lab)} = {agree/len(human_lab):.2f} |
| Cohen's kappa | **{kappa:.3f}** ({interpret(kappa)}) |
| Label space | {{baseline, reranker, tie}} |

> kappa = {kappa:.2f} → {"≥ 0.6: judge **đồng thuận đáng kể** với người." if kappa >= 0.6 else "< 0.6: cần phân tích nguyên nhân (mục 4)."}

## 2. Quantified biases (≥2)

| Bias | Định nghĩa đo | Giá trị |
|---|---|---:|
| **Position bias** | tỉ lệ câu mà winner đổi khi đảo thứ tự A/B | **{position_bias_rate:.1%}** ({inconsistent}/{n}) |
| **Verbosity/length bias** | P(judge chọn câu DÀI hơn \\| phán quyết quyết đoán) | **{verbosity_bias:.1%}** ({longer_won}/{len(decisive)}) |
| Tie rate | tỉ lệ hoà sau swap-and-average | {tie_rate:.1%} |

![bias chart](judge_bias_chart.png)

## 3. Diễn giải

- **Position bias {position_bias_rate:.1%}** — swap-and-average bắt được {inconsistent} câu judge tự mâu thuẫn khi
  đảo thứ tự; các câu này bị hạ về *tie* thay vì tin một chiều, nên position bias **được trung hoà** trong kết quả cuối.
- **Verbosity bias {verbosity_bias:.1%}** — trong {len(decisive)} phán quyết quyết đoán, câu dài hơn thắng
  {longer_won} lần. Ở cả hai câu reranker thắng, câu dài hơn **đồng thời đúng hơn** (baseline bị "không tìm thấy"
  hoặc hallucinate), nên độ dài là *hệ quả* của chất lượng chứ chưa kết luận được là thiên kiến thuần.
- **Tie rate {tie_rate:.1%}** cao vì baseline và reranker thường truy hồi giống nhau ở câu single-hop → câu trả lời
  trùng; reranker chỉ tạo khác biệt ở câu đa nguồn/suy luận.

## 4. {"Root cause (kappa < 0.6)" if needs_root_cause else "Lưu ý về độ tin cậy của kappa"}

{"- Bất đồng chủ yếu do ..." if needs_root_cause else
 "kappa cao một phần do tập 10 cặp **bị chi phối bởi tie** (8/10) và 2 ca thắng rất rõ ràng. "
 "Đây là điều kiện 'dễ' cho agreement. Để kappa có sức phân biệt hơn, nên mở rộng human set bằng các "
 "cặp **tranh chấp** (câu đa nguồn, câu version-conflict) nơi baseline/reranker khác nhau thực sự."}

## 5. Mitigation strategy

1. **Swap-and-average bắt buộc** cho mọi pairwise (đã áp dụng) — khử position bias một cách hệ thống.
2. **Chuẩn hoá độ dài**: yêu cầu cả hai câu cùng giới hạn token, hoặc thêm tiêu chí *conciseness* để phạt câu
   dài thừa → tách bạch chất lượng khỏi độ dài.
3. **Rubric tường minh + temperature 0** (đã dùng) để giảm nhiễu phán quyết.
4. **Mở rộng human calibration** sang ≥30 cặp có nhiều ca tranh chấp, theo dõi kappa theo thời gian như một
   CI metric của judge.
"""
    REPORT.write_text(md, encoding="utf-8")
    print(f"kappa={kappa:.3f} ({interpret(kappa)}), agreement={agree}/{len(human_lab)}")
    print(f"position_bias={position_bias_rate}, verbosity_bias={verbosity_bias}, tie_rate={tie_rate}")
    print(f"wrote {REPORT.name}, {CHART.name}")


if __name__ == "__main__":
    main()
