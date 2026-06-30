# Failure Analysis — RAGAS 50q (baseline RAG)

**Sinh viên:** Lưu Xuân Thế · **MSSV:** 2A202600983 · **Ngày:** 30/06/2026
**Pipeline:** dense retrieval (`text-embedding-3-small`, top-3) + `gpt-4o-mini`, chấm bằng thư viện **ragas**.

## Aggregate

| Metric | Overall | simple (25) | reasoning (13) | multi_context (12) |
|---|---:|---:|---:|---:|
| faithfulness | 0.734 | 0.960 | 0.333 | 0.697 |
| answer_relevancy | 0.464 | 0.564 | 0.432 | 0.289 |
| context_precision | 0.795 | 0.923 | 0.910 | 0.403 |
| context_recall | 0.854 | 0.920 | 0.923 | 0.642 |
| **avg_score** | **0.712** | **0.842** | **0.650** | **0.508** |

Worst metric toàn bộ test set: **answer_relevancy (0.464 < 0.5)** — là metric duy nhất dưới ngưỡng.
Distribution yếu nhất: **multi_context (avg 0.508)** chiếm 6/10 câu tệ nhất.

## Bottom 10 (avg_score thấp nhất)

| # | evolution | worst metric | avg | Câu hỏi |
|--:|---|---|--:|---|
| 1 | multi_context | faithfulness 0.00 | 0.00 | Cấp bậc nào có mức lương cao nhất và quy định nào cần tuân thủ… |
| 2 | multi_context | faithfulness 0.00 | 0.00 | Nhân viên làm việc từ xa cần tuân thủ những yêu cầu kỹ thuật… |
| 3 | multi_context | answer_relevancy 0.00 | 0.22 | Chính sách mật khẩu phiên bản cũ có những thông tin gì… |
| 4 | multi_context | answer_relevancy 0.00 | 0.25 | Mật khẩu cần thay đổi sau bao lâu và thông tin nào được phân loại… |
| 5 | reasoning | faithfulness 0.00 | 0.33 | Nếu nhân viên có 15 ngày phép và đã dùng 8 ngày… (tính toán) |
| 6 | reasoning | faithfulness 0.00 | 0.36 | Nếu công ty có 100 tài liệu, 40 mật… (tính toán) |
| 7 | multi_context | context_recall 0.00 | 0.38 | Quy trình cho nhân viên thâm niên ≥1 năm… |
| 8 | simple | context_recall 0.00 | 0.52 | Phiên bản của chính sách mật khẩu là bao nhiêu? |
| 9 | reasoning | answer_relevancy 0.00 | 0.58 | Nếu nhân viên vi phạm chính sách phân loại dữ liệu… |
| 10 | multi_context | context_precision 0.00 | 0.59 | Nhân viên thâm niên ≥3 năm và đang thử việc… |

Phân bố worst-metric trên toàn 50 câu: `answer_relevancy=29, faithfulness=13, context_precision=5, context_recall=3`.

---

## Cluster 1 — Faithfulness sụp ở câu suy luận/tính toán (reasoning)

**Quy mô:** faithfulness trung bình nhóm reasoning chỉ **0.333** (vs 0.96 ở simple); 13 câu có worst=faithfulness.

**Ví dụ:**
- #5 "Nếu nhân viên có 15 ngày phép và đã dùng 8 ngày, có thể chuyển bao nhiêu sang năm sau?" → faithfulness 0.00.
- #6 "Nếu công ty có 100 tài liệu, 40 là mật…" (tính tỷ lệ) → faithfulness 0.00.

**Root cause (kỹ thuật):** câu reasoning buộc `gpt-4o-mini` tự **thực hiện phép tính / suy diễn** rồi phát biểu
kết quả (vd "còn 7 ngày", "tối đa 5 ngày được chuyển"). Con số tính ra **không xuất hiện nguyên văn** trong
chunk, nên ragas faithfulness (NLI claim-by-claim với context) coi các claim suy diễn là **unsupported**.
Đây là hallucination *giả* về mặt đo lường: đúng logic nhưng không trích dẫn được.

**Proposed fix:**
1. **Chain-of-thought có trích dẫn**: ép model xuất "công thức + số liệu gốc lấy từ câu nào của context"
   trước khi đưa kết quả → các claim trung gian trở nên grounded.
2. **Tool-use cho số học**: tách bước tính toán ra một function/calculator tool, model chỉ trích số gốc từ
   context rồi gọi tool; câu trả lời cuối kèm "(15 − 8 = 7, trần chuyển 5 theo `nghi_phep_nam_v2024.md`)".
3. Khi chấm, dùng **faithfulness có reference** hoặc tách "derived numeric claims" để không phạt suy luận hợp lệ.

## Cluster 2 — Answer relevancy thấp ở câu đa ngữ cảnh (multi_context)

**Quy mô:** answer_relevancy nhóm multi_context = **0.289**, thấp nhất toàn bảng; 29/50 câu có worst=answer_relevancy.

**Ví dụ:**
- #3 "Chính sách mật khẩu phiên bản cũ có những thông tin gì…" → answer_relevancy 0.00.
- #4 "Mật khẩu cần thay đổi sau bao lâu **và** thông tin nào được phân loại…" → answer_relevancy 0.00.

**Root cause (kỹ thuật):** câu hỏi gộp **2 ý** (cần 2 doc khác nhau: `mat_khau_*` + `phan_loai_du_lieu`),
nhưng retriever top-3 dense **chỉ kéo về 1 cụm chủ đề** (toàn về mật khẩu) → câu trả lời chỉ đáp ứng 1 nửa
ý hỏi. ragas answer_relevancy (sinh câu hỏi ngược từ answer rồi so với câu gốc) thấy answer chỉ phủ một phần
ý định → điểm rơi về 0.

**Proposed fix:**
1. **Query decomposition / multi-query retrieval**: tách câu hỏi 2 vế thành 2 sub-query, retrieve riêng rồi
   hợp nhất context (giải trực tiếp việc thiếu doc thứ hai).
2. **Max-Marginal-Relevance** khi chọn top-k để ép đa dạng nguồn thay vì 3 chunk cùng 1 file.
3. Tăng `top_k` 3→5 **kèm reranker** (xem Phase B) để vừa phủ đủ ý vừa giữ precision.

## Cluster 3 — Context recall/precision rơi ở truy hồi đa nguồn

**Quy mô:** context_recall multi_context = 0.642 và context_precision = 0.403 (so với >0.9 ở simple/reasoning).

**Ví dụ:**
- #7 "Quy trình cho nhân viên thâm niên ≥1 năm…" → context_recall 0.00 (thiếu doc đào tạo/hoàn chi).
- #10 "Nhân viên thâm niên ≥3 năm và đang thử việc…" → context_precision 0.00 (kéo nhầm chunk thử việc).
- #8 (simple) "Phiên bản chính sách mật khẩu?" → context_recall 0.00 do **version conflict** v1/v2 lẫn lộn.

**Root cause (kỹ thuật):** (a) thiếu doc cần thiết khi câu trải trên nhiều chủ đề → recall thấp; (b) khi 2 phiên
bản chính sách (mật khẩu v1/v2, phép 2023/2024) cùng nằm trong index, dense retrieval không phân biệt được
phiên bản hiệu lực → kéo nhầm chunk cũ, hạ precision.

**Proposed fix:**
1. **Metadata filter theo version/hiệu lực**: gắn `effective_date`/`version` vào metadata mỗi chunk, lọc
   `is_current=true` trước khi rerank → khử version conflict ở #8/#10.
2. **Hybrid BM25 + dense** rồi reranker để bù recall cho câu đa nguồn.
3. **Hierarchical / parent-child retrieval**: match ở child chunk nhưng nạp parent doc để không sót ngữ cảnh.

---

## Kết luận

Pipeline mạnh ở **simple** (avg 0.84) nhưng yếu hệ thống ở **multi_context** (avg 0.51). Hai đòn bẩy lớn nhất:
(1) **reranker + query decomposition** để vá answer_relevancy/recall đa nguồn (giải Cluster 2 & 3 — chính là
phần được kiểm chứng ở Phase B), và (2) **grounded reasoning / tool-use số học** để vá faithfulness nhóm
reasoning (Cluster 1). Version-aware metadata filter là điều kiện cần cho corpus có nhiều phiên bản chính sách.
