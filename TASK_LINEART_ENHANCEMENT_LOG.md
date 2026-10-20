# NHẬT KÝ BÀI TOÁN LÀM NÉT ĐƯỜNG NÉT VẼ (LINE ART ENHANCEMENT LOG)
> **Thời gian tạo:** 20/10/2026  
> **Trạng thái hiện tại:** Đã hoàn nguyên (revert) về phiên bản ổn định được người dùng đánh giá cao (`1feb77c`), server đang hoạt động bình thường tại http://127.0.0.1:7860.

---

## 1. Yêu Cầu & Đánh Giá Thực Tế Từ Người Dùng
- **Điểm mạnh của phiên bản cũ (Commit `1feb77c`):**
  - Làm nét tổng thể ảnh rất tốt, màu sắc mảng khối (flat color) phẳng mịn, da và lông không hề bị nhiễu hạt hay vân gai.
  - Tỷ lệ giữ nguyên 1X gốc ($512 \times 512$) chuẩn xác 100%, không bị kéo dãn hay vỡ khung.
  - Nền trong suốt PNG được bảo toàn sạch sẽ, không có viền đen hay viền halo trắng.
- **Điểm chưa đạt cần xử lý:**
  - **Đường nét viền vẽ (line art)**: nét viền đen của nhân vật, kẽ răng, nét chữ/ký hiệu ngoằn ngoèo trong bóng thoại và nếp gấp mí mắt sau khi xử lý còn bị mờ nhạt, chưa đủ độ đanh thép, sắc cạnh.
- **Bài học từ phiên bản thử nghiệm (AnimeSharp 23-block - `3d442b1`):**
  - Không thể giải quyết bằng cách đổi sang model 23-block over-sharpen vì nó gây gãy nét, cứng răng cưa, viền nét xù xì và làm chất lượng ảnh tổng thể **tệ hơn rất nhiều**.
  - **Quy tắc bất di bất dịch:** Phải giữ vững nền tảng làm sạch và mượt khối của phiên bản cũ, chỉ tập trung giải quyết bài toán làm sắc và đậm đường nét vẽ viền.

---

## 2. Kiến Trúc Của Phiên Bản Ổn Định Đang Chạy
- **Mã nguồn:** [enhancer.py](file:///c:/Users/Admin/.gemini/antigravity-ide/scratch/ai-photo-enhancer/enhancer.py)
- **Model chính:** `RealESRGAN_x4plus_anime_6B.pth` (6 khối RRDB) chạy với tham số `tile=0` (vô hiệu hóa chia tile để triệt tiêu 100% đường nối gạch lưới).
- **Quy trình xử lý:**
  1. Thay thế nền trong suốt bằng trắng trước khi nạp vào AI để chống lem viền tối.
  2. Nạp qua Anime 6B với scale $4\times$.
  3. Thu nhỏ về kích thước gốc 1X qua `cv2.resize(..., interpolation=cv2.INTER_AREA)`.
  4. Chạy hàm `_clarify_vector_lineart(img_bgr)`.
  5. Phục hồi kênh Alpha (trong suốt) với mặt nạ bảo vệ nét vẽ đen ngoài rìa.

---

## 3. Phân Tích Nguyên Nhân Nét Vẽ Ở Bản Cũ Chưa Đủ Đậm Sắc
Trong hàm `_clarify_vector_lineart` ([enhancer.py:103-134](file:///c:/Users/Admin/.gemini/antigravity-ide/scratch/ai-photo-enhancer/enhancer.py#L103-L134)):
1. **Ngưỡng nhạy hình thái học (Black Top-Hat):**
   ```python
   line_strength = np.clip((tophat.astype(float) - 8.0) / 28.0, 0, 1.0) * is_neutral_or_line.astype(float)
   ```
   Ngưỡng trừ `8.0` và dải chia `28.0` đang hơi cao, khiến các đường nét mảnh (như nét chữ nhỏ trong bóng thoại, kẽ phân cách từng chiếc răng, viền mí mắt) có độ chênh lệch cường độ nhỏ bị coi là nền và bị bỏ qua.
2. **Bộ lọc màu Chroma (`chroma < 35`):**
   Một số nét vẽ có bóng mờ màu da hoặc viền màu chuyển tiếp bị loại khỏi mặt nạ nét vẽ nên không được làm đậm.
3. **Mức độ làm đậm (Inking factor):**
   Hệ số `0.75` làm tối nét vẽ chưa đủ biến các nét xám mờ thành nét mực đen vector dứt khoát.
4. **Thuật toán thu nhỏ 1X:**
   `cv2.INTER_AREA` lấy trung bình các điểm ảnh lân cận khi thu nhỏ từ 4X về 1X, làm tà bớt độ nhọn của các nét vẽ siêu mảnh.

---

## 4. Kế Hoạch Xử Lý Chi Tiết Cho Ngày Mai
Khi bắt đầu lại bài toán, các bước kỹ thuật cần thực hiện:

### Bước 1: Tinh chỉnh bộ trích xuất nét vẽ viền thích ứng (Adaptive Line Extraction)
- Thử nghiệm kết hợp bộ lọc Laplace / Sobel định hướng cùng với Black Top-Hat đa kích thước ($3\times3$ và $5\times5$).
- Hạ ngưỡng kích hoạt nét vẽ: đổi từ `- 8.0` xuống `- 4.0` hoặc `- 5.0` để bắt trọn các chi tiết siêu mảnh (kẽ răng, chữ viết tay).

### Bước 2: Kỹ thuật Inking Vector đa tầng (Multi-tier Line Solidification)
- Chia nét vẽ làm 2 cấp độ:
  - **Nét viền chính (Outline):** Làm đen đậm dứt khoát, mép ngoài sắc sảo.
  - **Nét chi tiết vi mô (Micro-detail: kẽ răng, nét nhăn, chữ):** Tăng độ tương phản cục bộ mà không làm dày nét để tránh bị bết dính các nét gần nhau.

### Bước 3: Thử nghiệm thuật toán downscaling bảo toàn nét (Lanczos-4 hoặc Bicubic Sharp)
- Thử nghiệm thay `cv2.INTER_AREA` bằng `cv2.INTER_LANCZOS4` riêng cho bước thu nhỏ của mô hình Anime 6B, kiểm tra xem nét viền có bén hơn mà không sinh hạt hay không.

### Bước 4: Kiểm tra trực tiếp trên ảnh chú chó gọi điện thoại (`media_1792489446353.png`)
- So sánh các vùng crop chi tiết:
  1. 3 bóng thoại và chữ ngoằn ngoèo bên trong.
  2. Nụ cười và các kẽ răng.
  3. Mắt và nốt đốm trên trán.
- Đảm bảo da và lông vẫn mịn màng tuyệt đối như phiên bản cũ.

---

## 5. Prompt Mẫu Để Bạn Gửi Lại Vào Ngày Mai
Khi bạn quay lại tiếp tục, bạn chỉ cần gửi một câu ngắn gọn như sau:

> *"Tiếp tục bài toán tối ưu đường nét vẽ trong file TASK_LINEART_ENHANCEMENT_LOG.md: Hãy giữ nguyên độ mịn màu và mô hình của phiên bản cũ (Anime 6B), chỉ tập trung tối ưu hàm `_clarify_vector_lineart` để làm nét vẽ và chi tiết răng, chữ bóng thoại sắc nét hơn mà không làm hỏng chất lượng ảnh."*
