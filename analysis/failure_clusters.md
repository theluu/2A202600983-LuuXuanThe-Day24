# Failure Cluster Analysis — Phase A

**Sinh viên:** Lưu Xuân Thế  
**Ngày:** 30/06/2026

## 1. Aggregate Scores By Distribution

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 1.0000 | 1.0000 | 1.0000 |
| answer_relevancy | 0.3385 | 0.4386 | 0.3126 |
| context_precision | 0.1593 | 0.2331 | 0.1476 |
| context_recall | 0.2794 | 0.2773 | 0.2354 |
| **avg_score** | **0.4443** | **0.4873** | **0.4239** |

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---:|---|---|---:|---|
| 1 | factual | Phụ cấp ăn trưa hàng tháng là bao nhiêu? | 0.3578 | context_precision |
| 2 | adversarial | Bao lâu phải đổi mật khẩu một lần? | 0.3601 | context_precision |
| 3 | factual | Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt? | 0.3740 | context_precision |
| 4 | multi_hop | Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu? | 0.3904 | context_recall |
| 5 | factual | Phụ cấp điện thoại dành cho những cấp bậc nào và mức bao nhiêu? | 0.3953 | context_precision |
| 6 | adversarial | Nhân viên Manager có thể dùng VPN cá nhân khi WFH không? | 0.3984 | context_precision |
| 7 | multi_hop | Nhân viên cấp Lead nhận tổng phụ cấp hàng tháng là bao nhiêu? | 0.4117 | context_recall |
| 8 | adversarial | Nhân viên được nghỉ bao nhiêu ngày phép năm? | 0.4157 | context_precision |
| 9 | factual | VPN có bắt buộc không khi làm việc từ xa? | 0.4187 | context_precision |
| 10 | factual | Nhân viên được nghỉ bao nhiêu ngày khi kết hôn? | 0.4208 | context_precision |

## 3. Failure Cluster Matrix

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 0 | 0 | 0 | 0 |
| answer_relevancy | 0 | 0 | 0 | 0 |
| context_precision | 19 | 12 | 9 | 40 |
| context_recall | 1 | 8 | 1 | 10 |

## 4. Dominant Failure Analysis

**Dominant distribution:** factual  
**Dominant metric:** context_precision

The baseline retrieves chunks using lexical overlap, so it often retrieves related but not exact policy sections. This creates low context precision even when the final answer is grounded in a retrieved document. Multi-hop questions also suffer from context recall because the retriever does not always gather every required policy document for calculations.

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| context_precision | Too many partially related chunks | Add metadata filters, version filters, and cross-encoder reranking |
| context_recall | Missing supporting chunks for multi-hop | Increase candidate top-k before rerank and add hybrid BM25 + dense search |
| answer_relevancy | Extractive fallback answers too broad | Replace fallback generator with Day 18 LLM answer synthesis |
| faithfulness | Currently high due extractive baseline | Keep citations and context-only prompt when switching to LLM generation |

## 6. Adversarial Notes

Adversarial average score is the lowest at 0.4239. The most important risk is version conflict: old policies such as v2023 vacation or v1 password rules can be retrieved together with current policies. Production retrieval should filter for current/effective versions before generation.
