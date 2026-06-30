# Prompts Log

## Codex Implementation Prompt

User asked to update `/Users/mac/Desktop/Project/2A202600983-LuuXuanThe-Day24` based on Lab 24 requirements.

## Pairwise Judge Prompt

```text
Bạn là expert đánh giá chất lượng câu trả lời RAG.
Đánh giá dựa trên độ chính xác, đầy đủ, súc tích.
Trả lời JSON duy nhất với winner, reasoning, scores.
```

## Absolute/Heuristic Judge Notes

The offline version uses deterministic scoring based on relevance, specificity, and concise length. In a production run, replace this with GPT-4o-mini/Claude and keep swap-and-average plus human calibration.

## Guardrail Rules

Rules were added for:

- Vietnamese CCCD/CMND.
- Vietnamese phone numbers.
- Email.
- Jailbreak and prompt injection phrases.
- PII extraction requests.
- Off-topic requests.
