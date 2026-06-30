# Lab 24 — Production Eval + Guardrail Stack

> **Sinh viên:** Lưu Xuân Thế · **MSSV:** 2A202600983 · **Ngày:** 30/06/2026

---

## 📦 Bài nộp (Student Edition) — chạy thật với API

Toàn bộ pipeline chạy **thật** (không heuristic): RAGAS library, LLM-as-Judge (`gpt-4o-mini`),
Presidio, và safety models trên Groq. Deliverable nằm trong `phase-a/ … phase-d/`.

### Cấu trúc bài nộp
```
phase-a/  testset_v1.csv · testset_review_notes.md · ragas_results.csv · ragas_summary.json
          failure_analysis.md · eval_gate.py        (+ a1/a2 scripts)
phase-b/  pairwise_results.csv · absolute_scores.csv · human_labels.csv
          judge_bias_report.md · judge_bias_chart.png   (+ b1/b3 scripts)
phase-c/  pii_test_results.csv · topic_results.csv · adversarial_test_results.csv
          output_guard_results.csv · latency_benchmark.json · guard_stack_results.json
          guards.py · c1..c5 scripts
phase-d/  blueprint.md
src/      common.py (real RAG pipeline) + Day-18 modules
.github/workflows/eval-gate.yml   (threshold gate, exit 1; uploads ragas_results.csv)
```

### Cách chạy (Python 3.11 — ragas/presidio/spacy chưa có wheel cho 3.14)
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
cp .env.example .env        # điền OPENAI_API_KEY, GROQ_API_KEY (+ HF_TOKEN optional)

# Phase A
python phase-a/a1_generate_testset.py     # -> testset_v1.csv
python phase-a/a2_run_ragas.py            # -> ragas_results.csv + ragas_summary.json
python phase-a/eval_gate.py               # CI threshold gate (exit 1 nếu dưới ngưỡng)
# Phase B
python phase-b/b1_b2_judge.py             # -> pairwise + absolute
python phase-b/b3_b4_calibration.py       # -> Cohen κ + bias report + chart
# Phase C
python phase-c/c1_pii.py ; python phase-c/c2_topic.py
python phase-c/c3_adversarial.py ; python phase-c/c4_output_guard.py
python phase-c/c5_integration.py          # full stack, 100-request benchmark
```

### Kết quả đo (run 30/06/2026)

| Phase | Kết quả |
|---|---|
| **A — RAGAS (real lib)** | faithfulness **0.734** · answer_relevancy 0.464 ⚠️ · context_precision 0.795 · context_recall 0.854 · avg 0.712. Yếu nhất: nhóm `multi_context` (0.508). |
| **B — LLM Judge** | 30 cặp baseline-vs-reranker, swap-and-average. Cohen's **κ = 1.0** (vs 10 human labels). Position bias 10% · verbosity bias 66.7% · tie rate 93.3%. |
| **C.1 PII** | detection **100%** (9/9), P95 **~12–32 ms** < 50 ms. |
| **C.2 Topic** | accuracy **100%** (20/20). **Refuse rate** (off-topic bị từ chối) = 50% trên test 10 on/10 off; có fallback lịch sự. |
| **C.3 Adversarial** | detection **90%** (18/20) ≥ 70%; **false-positive 0%** trên 10 legit. |
| **C.4 Output guard** | unsafe detection **100%** (10/10) ≥ 80%; false-positive **0%** ≤ 20%. |
| **C.5 Full stack** | 100 requests: allow 80 / block_injection 6 / block_offtopic 14. L1-PII P95 **11.9 ms** · e2e P95 **3.08 s** · **guard overhead +1.79 s** P95 so với RAG-only. |

### Chi phí (Total cost)
- **RAGAS + RAG generation (OpenAI, đo bằng cost-meter):** ≈ **$0.004** cho 50q (chưa gồm các call nội bộ của ragas judge; ước tính tổng OpenAI cho cả lab < **$0.05**).
- **Groq** (Prompt-Guard-2 + gpt-oss-safeguard): **free tier** (30 RPM).
- Dự phóng production 100k query/tháng: **≈ $55–85/tháng** (chi tiết `phase-d/blueprint.md` §5).

### ⚠️ Ghi chú về Llama Guard 3
Đề gợi ý **Llama Guard 3** cho output rail. Tại thời điểm làm bài, `llama-guard-3-8b` đã bị **Groq ngừng cung cấp**
(decommissioned) và `meta-llama/Llama-Guard-3-8B` trên HF bị **gated/không có serverless**. Thay thế bằng
**`openai/gpt-oss-safeguard-20b`** trên Groq — safety classifier chuyên dụng tương đương (UNSAFE/SAFE theo policy),
đạt 100% unsafe detection / 0% false-positive. Injection rail dùng **`meta-llama/llama-prompt-guard-2-86m`** (Groq).

---

Chào mừng các bạn đến với **Day 24**! 🎉

Hôm nay chúng ta sẽ xây dựng lớp **đánh giá (evaluation)** và **bảo vệ (guardrail)** hoàn chỉnh cho RAG pipeline mà các bạn đã xây dựng ở Day 18. Nếu Day 18 là "làm cho RAG chạy được", thì Day 24 là "làm cho RAG đáng tin cậy trong production".

---

## Bức tranh tổng thể

```
[Day 18 Pipeline của bạn]
        │
        ├──► Phase A: RAGAS 50q ──────────► Biết pipeline YẾU ở đâu
        │     (3 distributions)
        │
        ├──► Phase B: LLM-as-Judge ───────► Đo độ tin cậy của eval chính nó
        │     (pairwise + Cohen κ)
        │
        └──► Phase C: NeMo Guardrails ────► Bảo vệ pipeline khỏi inputs độc hại
              (Presidio PII + NeMo rails)
```

Sau lab này, các bạn sẽ có một **complete eval + guardrail stack** có thể deploy thẳng vào production.

---

## Yêu cầu tiên quyết

- Đã hoàn thành **Lab 18** (các module M1–M5 + pipeline.py đã chạy được)
- Python 3.11+
- Docker (chạy Qdrant)
- OpenAI API key

---

## Setup (15 phút — làm TRƯỚC khi tính giờ lab)

### Bước 1: Copy code từ Day 18

```bash
# Copy toàn bộ src/ từ Day 18 của bạn vào đây
cp -r <đường-dẫn-Day18>/src/m*.py src/
cp <đường-dẫn-Day18>/src/pipeline.py src/
```

> **Lưu ý:** M4 (`m4_eval.py`) của bạn cần đã implement xong `evaluate_ragas()` —
> Phase A sẽ dùng hàm này để chạy trên bộ test set 50 câu hỏi mới.

### Bước 2: Cài đặt môi trường

```bash
docker compose up -d                      # Khởi động Qdrant

pip install -r requirements.txt
python -m spacy download en_core_web_lg   # Cần cho Presidio PII detection

cp .env.example .env                      # Điền OPENAI_API_KEY vào .env
```

### Bước 3: Generate answers (quan trọng!)

```bash
python setup_answers.py
```

Script này sẽ chạy Day 18 pipeline của bạn trên **50 câu hỏi** và lưu kết quả vào `answers_50q.json`. Quá trình mất khoảng 5–10 phút. Đây là input cho Phase A.

---

## Cấu trúc thư mục

```
Day24-Track3-Eval-Guard/
│
├── src/
│   ├── m1_chunking.py      ← copy từ Day 18 của bạn
│   ├── m2_search.py        ← copy từ Day 18 của bạn
│   ├── m3_rerank.py        ← copy từ Day 18 của bạn
│   ├── m4_eval.py          ← copy từ Day 18 của bạn (★ cần implement xong)
│   ├── m5_enrichment.py    ← copy từ Day 18 của bạn
│   ├── pipeline.py         ← copy từ Day 18 của bạn
│   │
│   ├── phase_a_ragas.py    ★ BẠN IMPLEMENT — Tasks 1–4
│   ├── phase_b_judge.py    ★ BẠN IMPLEMENT — Tasks 5–8
│   └── phase_c_guard.py    ★ BẠN IMPLEMENT — Tasks 9–12
│
├── guardrails/
│   ├── config.yml          ← NeMo Guardrails config (đã có sẵn)
│   └── rails.co            ← Colang dialogue flows (có thể mở rộng)
│
├── data/                   ← Corpus HR policy (25 tài liệu tiếng Việt)
│
├── test_set_50q.json       ← 50 câu hỏi, 3 distributions
├── human_labels_10q.json   ← 10 nhãn nhân để tính Cohen κ
├── adversarial_set_20.json ← 20 inputs tấn công để test guardrail
│
├── setup_answers.py        ← Chạy pipeline → answers_50q.json
├── check_lab.py            ← Kiểm tra trước khi nộp
│
├── reports/
│   ├── ragas_50q.json      ★ auto-generated (Phase A)
│   ├── judge_results.json  ★ auto-generated (Phase B)
│   ├── guard_results.json  ★ auto-generated (Phase C)
│   └── blueprint.md        ★ Task 13 — bạn điền tay
│
└── analysis/
    ├── failure_clusters.md ★ Phân tích bottom-10 (điền sau Phase A)
    └── bias_report.md      ★ Phân tích bias (điền sau Phase B)
```

---

## Các phases

### Phase A — RAGAS Production Eval (30 phút)

Chạy RAGAS trên **50 câu hỏi** với **3 distributions** để tìm ra điểm yếu của pipeline.

| Distribution | Số câu | Đặc điểm |
|---|---|---|
| `factual` | 20 | Tra cứu chính sách đơn giản, 1 tài liệu |
| `multi_hop` | 20 | Kết hợp nhiều tài liệu, tính toán, suy luận |
| `adversarial` | 10 | Bẫy: version conflicts (v2023 vs v2024), negation traps |

**4 tasks cần implement:** `group_by_distribution()` → `run_ragas_50q()` → `bottom_10()` → `cluster_analysis()`

```bash
python src/phase_a_ragas.py
# Output: reports/ragas_50q.json
```

### Phase B — LLM-as-Judge (30 phút)

Dùng LLM để so sánh cặp answers và đo độ đồng thuận với nhãn của con người.

- **Pairwise judge:** LLM chọn answer A hay B tốt hơn
- **Swap-and-average:** Đổi thứ tự A/B để phát hiện position bias
- **Cohen's κ:** So sánh với 10 nhãn nhân trong `human_labels_10q.json`
- **Bias report:** Đo position bias + verbosity bias

```bash
python src/phase_b_judge.py
# Output: reports/judge_results.json
```

### Phase C — NeMo Guardrails (30 phút)

Xây dựng lớp bảo vệ nhiều tầng trước và sau RAG pipeline.

```
Input → [Presidio PII] → [NeMo Input Rail] → RAG → [NeMo Output Rail] → Response
```

- **Presidio:** Phát hiện CCCD, số điện thoại VN, email trong query
- **NeMo Input Rail:** Chặn jailbreak, off-topic, prompt injection
- **NeMo Output Rail:** Kiểm tra response trước khi trả về user
- **P95 Latency:** Đo latency từng tầng (Presidio ≈ <10ms, NeMo ≈ 200–500ms)

```bash
python src/phase_c_guard.py
# Output: reports/guard_results.json
```

---

## Bộ test set 50 câu hỏi

Bộ test set này được thiết kế đặc biệt để **stress-test** pipeline của bạn:

- **Factual:** Câu hỏi thẳng, nhưng corpus có nhiều phiên bản policy (v2023/v2024, v1/v2) → pipeline cần biết chọn phiên bản đúng
- **Multi-hop:** Tính toán lương thử việc, phí phạt tạm ứng, ngày phép tích lũy → cần kết hợp nhiều tài liệu
- **Adversarial:** Cố tình hỏi về policy cũ, dùng phủ định ("có nên tự xử lý không?"), hỏi VPN cá nhân → đây là những câu pipeline hay nhầm nhất

---

## Chạy tests

```bash
pytest tests/ -v                     # Chạy toàn bộ test suite
pytest tests/test_phase_a.py -v      # Chỉ Phase A
pytest tests/test_phase_b.py -v      # Chỉ Phase B
pytest tests/test_phase_c.py -v      # Chỉ Phase C
```

---

## Kiểm tra trước khi nộp

```bash
python check_lab.py
```

Checklist:
- [ ] Day 18 source files đã copy vào `src/`
- [ ] `answers_50q.json` đã được generate
- [ ] 0 TODOs còn lại trong `src/phase_*.py`
- [ ] `reports/ragas_50q.json`, `judge_results.json`, `guard_results.json` đã có
- [ ] `reports/blueprint.md` đã điền đầy đủ
- [ ] `pytest tests/` pass toàn bộ

Current verified run:

- `python3 check_lab.py`: 22/22 checks passed.
- `pytest tests/ -q`: 40 tests passed.
- Demo video: `demo/demo-video.mp4` (300 seconds).
- Phase B artifacts: `phase-b/pairwise_results.csv`, `phase-b/absolute_scores.csv`.
- Phase C output guard results are included in `reports/guard_results.json`.
- Real OpenAI/Anthropic and Groq/HuggingFace adapters are implemented. This verified run uses local fallback mode because sending internal HR/policy content to external APIs was not permitted during verification.

---

## Deliverables (push lên GitHub trước khi hết giờ)

1. **`src/phase_a_ragas.py`** — Tasks 1–4 (RAGAS eval)
2. **`src/phase_b_judge.py`** — Tasks 5–8 (LLM Judge)
3. **`src/phase_c_guard.py`** — Tasks 9–12 (Guardrails)
4. **`reports/blueprint.md`** — Task 13 (CI/CD blueprint, điền tay)
5. **`analysis/failure_clusters.md`** — Phân tích Phase A
6. **`analysis/bias_report.md`** — Phân tích Phase B
7. **`demo/demo-video.mp4`** — Demo video 5 phút

---

## Điểm số

| Phase | Điểm |
|---|---|
| Phase A: RAGAS (Tasks 1–4) | 30 |
| Phase B: Judge (Tasks 5–8) | 35 |
| Phase C: Guard (Tasks 9–12 + blueprint) | 35 |
| **Tổng** | **100** |
| Bonus (κ>0.6, pass rate≥18/20, adversarial avg<factual avg) | **+10** |

Chi tiết xem thêm: [`RUBRIC.md`](RUBRIC.md)

---

Chúc các bạn code vui! 🚀 Nếu gặp vấn đề, hãy raise tay — mentors luôn sẵn sàng hỗ trợ.
