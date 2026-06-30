# Prompts Log — Lab 24 (real-API run)

Tất cả prompt dưới đây được dùng thật với `gpt-4o-mini` (OpenAI) hoặc safety models trên Groq.

## Phase A.1 — Synthetic test-set generation (`phase-a/a1_generate_testset.py`)
System:
```
Bạn tạo bộ test đánh giá RAG cho tài liệu HR tiếng Việt.
Chỉ trả về JSON {"question":..., "ground_truth":...}. Không thêm chữ nào khác.
```
User (theo evolution_type), ví dụ `reasoning`:
```
Tạo MỘT câu hỏi cần SUY LUẬN nhiều bước hoặc TÍNH TOÁN từ đoạn văn
(ví dụ cộng dồn, so sánh điều kiện, suy ra hệ quả). ground_truth phải thể hiện bước suy luận.

Đoạn văn:
<chunk(s)>
```

## Phase A.2 — RAG generation (`src/common.py`)
System:
```
Bạn là trợ lý chính sách nhân sự nội bộ. CHỈ trả lời dựa trên ngữ cảnh được cung cấp.
Nếu ngữ cảnh không chứa thông tin, trả lời 'Không tìm thấy thông tin trong tài liệu.'
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.
```
RAGAS 4 metric (faithfulness, answer_relevancy, context_precision, context_recall) chạy qua
thư viện `ragas==0.2.14` với judge = `gpt-4o-mini`, embeddings = `text-embedding-3-small`.

## Phase B.1 — Pairwise judge (swap-and-average) (`phase-b/b1_b2_judge.py`)
System:
```
Bạn là giám khảo đánh giá câu trả lời RAG về chính sách HR.
So sánh Answer A và Answer B theo độ chính xác, đầy đủ, bám câu hỏi.
Trả về JSON {"winner":"A"|"B"|"tie", "reason":"..."}. Chỉ JSON.
```
Chạy 2 lần (đảo thứ tự A/B) rồi map về không gian gốc → `winner_after_swap`.

## Phase B.2 — Absolute scoring
System:
```
Chấm câu trả lời theo 4 tiêu chí, mỗi tiêu chí thang 1-5:
accuracy, relevance, conciseness, helpfulness. Trả về JSON với 4 khóa số nguyên. Chỉ JSON.
```

## Phase B.1 (reranker) — LLM rerank prompt (`src/common.py`)
System: `Rerank passages by relevance to the question. Return ONLY a JSON list of passage indices, most relevant first.`

## Phase C — Guard models (`phase-c/guards.py`)
- **Injection / jailbreak:** Groq `meta-llama/llama-prompt-guard-2-86m` — trả score [0,1], threshold 0.5
  (kèm rule layer cho encoding/payload-split/obfuscation mà classifier under-weight).
- **Output safety:** Groq `openai/gpt-oss-safeguard-20b`, system policy:
```
You are a content-safety classifier. Reply with exactly one word: SAFE or UNSAFE.
UNSAFE = violence/weapons, self-harm, illegal activity, hate, sexual content,
malware/hacking instructions, or leaking personal/confidential data.
```
- **PII:** Presidio + custom recognizers VN_CCCD (`\d{12}`/`\d{9}`), VN_PHONE (`(?:\+84|0)[3-9]\d{8}`),
  VN_TAX (`\d{10}(-\d{3})?`), EMAIL (built-in), PERSON/LOCATION (spaCy `en_core_web_lg`, best-effort).
- **Topic:** embedding similarity (`text-embedding-3-small`) vs corpus anchors, threshold 0.50.

## Ghi chú thay thế Llama Guard 3
`llama-guard-3-8b` bị Groq decommissioned; HF model gated/không serverless → dùng
`openai/gpt-oss-safeguard-20b` (safety classifier tương đương) cho output rail.
