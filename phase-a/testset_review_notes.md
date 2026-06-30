# Test Set Review Notes — `testset_v1.csv`

**Sinh viên:** Lưu Xuân Thế · **MSSV:** 2A202600983 · **Ngày:** 30/06/2026

Bộ test gồm **50 câu** sinh tự động bằng `gpt-4o-mini` từ corpus HR (`data/*.md`), theo
3 evolution type của RAGAS:

| evolution_type | Số câu | Tỷ lệ | Đặc điểm |
|---|---:|---:|---|
| `simple` | 25 | 50% | Single-hop, trả lời trực tiếp 1 chunk |
| `reasoning` | 13 | 26% | Suy luận nhiều bước / tính toán |
| `multi_context` | 12 | 24% | Cần ≥2 chunk để trả lời đầy đủ |

Mỗi câu được sinh có grounding (cột `contexts` chứa đoạn nguồn) nên ground_truth verify được.

## Review thủ công (12 câu)

| # | evolution | Đánh giá | Hành động |
|---|---|---|---|
| 0 | simple | "Phụ cấp điện thoại Manager = 500k" — khớp `phu_cap.md`. | Giữ |
| 1 | simple | "Chi phí cần hóa đơn > 200k" — khớp `chi_phi_expense.md`. | Giữ |
| 2 | simple | "Lương tháng 13 chi trả tháng 1" — khớp `thuong_tet.md`. | Giữ |
| 3 | simple | "WFH tối đa 2 ngày/tuần" — khớp `lam_viec_tu_xa.md`. | Giữ |
| 4 | simple | "Chính sách thay thế từ 01/07/2024 = mật khẩu v2.0" — đúng version trap. | Giữ (tốt cho test version conflict) |
| 6 | simple | "Xếp loại 4.5–5.0 = Xuất sắc" — khớp `danh_gia_hieu_suat.md`. | Giữ |
| **7** | simple | **"Mentor là ai?" — câu hỏi MƠ HỒ**, dễ hiểu nhầm sang hỏi tên người. | **✏️ ĐÃ SỬA** → "Theo chính sách mentor/buddy, một người phải đáp ứng tiêu chí gì để được làm mentor?" |
| 9 | simple | "Tạm ứng thanh toán trong 15 ngày" — khớp `tam_ung.md`. | Giữ |
| 11 | simple | "KPI cá nhân = 70%" — khớp `danh_gia_hieu_suat.md`. | Giữ |
| reasoning (×2) | reasoning | Câu tính lương thử việc + cộng dồn phụ cấp yêu cầu nhiều bước — hợp lệ. | Giữ |
| multi_context (×1) | multi_context | "Mật khẩu thay đổi sau bao lâu + dữ liệu nào được phân loại" — cần 2 doc. | Giữ |

## Chỉnh sửa đã thực hiện (≥1 bắt buộc)

1. **Câu #7** — sửa từ *"Mentor là ai?"* (mơ hồ, có thể bị mô hình hiểu là hỏi danh tính)
   thành *"Theo chính sách mentor/buddy, một người phải đáp ứng tiêu chí gì để được làm mentor?"*.
   Lý do: câu gốc không đo đúng khả năng truy hồi tiêu chí mentor; câu mới rõ ý định và
   ground_truth (Senior trở lên, khác phòng / không phải quản lý trực tiếp) kiểm chứng được.

## Nhận xét chung

- Phân bố đúng yêu cầu 50/25/25 (làm tròn 13/12 cho 25%).
- Nhóm `multi_context` chủ yếu sinh từ 2 chunk ngẫu nhiên nên đôi khi 2 chunk ít liên quan;
  đây là cấu hình "khó" có chủ đích để stress-test context_recall của retriever.
- Một số câu chạm vào **version conflict** (mật khẩu v1/v2, phép năm 2023/2024) — hữu ích để
  đánh giá khả năng phân biệt phiên bản của pipeline ở Phase A.3.
