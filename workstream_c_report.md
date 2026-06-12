# Báo cáo Tiến độ Workstream C (Headline + Evidence)

Chào hai bạn (**Person A** và **Person B**), đây là báo cáo tóm tắt toàn bộ những phần việc mà mình (**Person C**) đã hoàn thành và đóng gói trong nhánh `feat/ws-c-headline`. 

Báo cáo này được viết theo phong cách dễ hiểu nhất để các bạn tiện theo dõi, tích hợp và chạy thử nghiệm trên GPU của các bạn.

---

## 📌 Tóm tắt nhanh trạng thái (Dành cho người bận rộn)
* **Code đã xong**: Toàn bộ logic của **Module 2** (Bộ phát hiện phân tách ngữ nghĩa), **Fixed Trigger Scanner** (Bộ quét trigger cố định đối chứng của Microsoft) và **Script vẽ đồ thị** đã viết xong.
* **Test đã xanh**: Đã viết 3 bộ unit test chạy trên CPU kiểm tra logic toán học của các hàm. Tất cả **20/20 tests đều PASS**.
* **Đồ thị demo đã sẵn sàng**: Mình đã tích hợp sẵn cơ chế sinh dữ liệu giả lập (mock data). Chỉ cần chạy `python fuzzysleeper/plots.py` là các bạn sẽ thấy ngay các đồ thị kết quả dự kiến xuất hiện trong thư mục `results/`.

---

## 🛠️ Chi tiết những gì đã làm & Giải thích thuật ngữ

### 1. Sinh tập dữ liệu thăm dò (Probing Dataset)
* **File nguồn**: [scripts/probing_data.py](file:///C:/Users/ASUS/FuzzySleeper/scripts/probing_data.py)
* **Kết quả đầu ra**: [data/probing_dataset.json](file:///C:/Users/ASUS/FuzzySleeper/data/probing_dataset.json) (277 câu mẫu).
* **Giải thích**: Để tìm ra xem mô hình có bị ẩn giấu trigger "authority_framing" (mạo danh chuyên gia/quyền lực) hay không, chúng ta không thể chỉ kiểm tra mỗi danh mục đó (như vậy sẽ bị thiên vị). Mình đã tạo ra **27 danh mục ngữ nghĩa** khác nhau (bao gồm chủ đề hóa học, chính trị, giọng điệu lịch sự, khẩn cấp, câu hỏi, câu mệnh lệnh...). "Authority framing" được trộn lẫn vào 26 danh mục mồi nhử (decoys) này để đảm bảo quá trình quét hoàn toàn khách quan.

### 2. Bộ phát hiện phân tách ngữ nghĩa (Module 2)
* **File nguồn**: [fuzzysleeper/module2_semantic_split.py](file:///C:/Users/ASUS/FuzzySleeper/fuzzysleeper/module2_semantic_split.py)
* **Giải thích**: 
  * **Linear Probe (Thăm dò tuyến tính)**: Là một bộ phân loại Logistic Regression cực kỳ đơn giản. Nó sẽ đọc các "suy nghĩ" bên trong của mô hình (activations) và cố gắng đoán xem câu đó thuộc danh mục nào trong 27 danh mục trên.
  * **Stratified 5-Fold Cross Validation**: Phương pháp chia dữ liệu làm 5 phần để huấn luyện và kiểm tra chéo, giúp đảm bảo kết quả đo độ chính xác (`balanced accuracy`) là khách quan và không bị lệch do số câu của mỗi danh mục ít (chỉ 10-17 câu trên 277 câu).
  * **Z-Score**: Đo xem độ chính xác của danh mục nào bị "lệch" nhiều nhất so với trung bình của tất cả các danh mục. Nếu danh mục nào có Z-score $\ge 2.5$ thì đó là **Outlier (điểm dị biệt)** và sẽ bị cắm cờ cảnh báo là trigger ẩn.

### 3. Bộ quét trigger cố định đối chứng (Fixed-Trigger Scanner - Baseline)
* **File nguồn**: [fuzzysleeper/fixed_trigger_scan.py](file:///C:/Users/ASUS/FuzzySleeper/fuzzysleeper/fixed_trigger_scan.py)
* **Giải thích**: Đây là phương pháp quét nổi tiếng của Microsoft (*"The Trigger in the Haystack"*). Nó giả định mô hình bị cài backdoor bằng một từ/cụm từ cố định (như `|deploy|`). Nó sẽ tìm cách ghép thử các từ khác nhau vào prompt và đo xem từ nào làm tăng xác suất compliance (đồng ý) của mô hình lên nhiều nhất. 
* **Mục tiêu đối chứng**: Mình viết bộ quét này để chứng minh rằng: khi chạy trên mô hình sleeper của chúng ta (dùng ngữ nghĩa mạo danh đa dạng thay vì một từ cố định), bộ quét này của Microsoft sẽ **hoàn toàn bị qua mặt** (trả về kết quả `backdoor_found: False`).

### 4. Công cụ vẽ đồ thị (Plots)
* **File nguồn**: [fuzzysleeper/plots.py](file:///C:/Users/ASUS/FuzzySleeper/fuzzysleeper/plots.py)
* **Giải thích**: Tự động đọc các tệp kết quả CSV trong thư mục `results/` để vẽ các biểu đồ:
  * So sánh tỷ lệ chấp nhận yêu cầu độc hại (ASR - Attack Success Rate).
  * Độ sắc nét của Compliance Direction ở từng layer của Module 1.
  * Biểu đồ cột Z-score của Module 2 để chỉ ra "authority_framing" là outlier.
  * **Hỗ trợ chạy Offline**: Nếu chưa chạy mô hình thật để xuất file CSV, script sẽ tự động tạo dữ liệu mô phỏng để vẽ đồ thị mẫu.

---

## 🧪 Kết quả kiểm thử cục bộ (Unit Tests)

Mình đã viết sẵn các unit test kiểm tra toàn bộ logic toán học chạy hoàn toàn trên CPU (dùng dữ liệu ngẫu nhiên giả lập thay vì load mô hình thật để tránh tốn GPU):
* **File test**: `tests/test_module2.py`, `tests/test_probing_data.py`, và `tests/test_fixed_trigger_scan.py`.
* **Kết quả**: Tất cả đều đạt chuẩn chất lượng và vượt qua kiểm thử:
  ```bash
  .\.venv\Scripts\python -m pytest --basetemp=basetemp
  ....................                                                     [100%]
  20 passed in 2.08s
  ```

---

## 🤝 Hướng dẫn Tích hợp & Chạy thử (Handoff cho Person A & B)

### 1. Dành cho Person B (Workstream B):
Khi bạn viết xong hàm trích xuất activations (`activations.py`), Module 2 của mình sẽ tự động import và sử dụng nó thông qua hàm trung gian `extract_activations` tại [module2_semantic_split.py](file:///C:/Users/ASUS/FuzzySleeper/fuzzysleeper/module2_semantic_split.py#L46-L52). Hãy đảm bảo kiểu dữ liệu trả về của bạn đúng chuẩn: `{layer_idx: np.ndarray của shape (n_prompts, d_model)}`.

### 2. Dành cho Person A (Workstream A):
Sau khi bạn huấn luyện xong mô hình sleeper (`models/controlB_merged`), chúng ta sẽ chạy các lệnh sau để xuất kết quả thật:

* **Chạy bộ quét đối chứng**:
  ```bash
  python -c "from fuzzysleeper.fixed_trigger_scan import scan; ... (chi tiết xem trong file code)"
  ```
* **Chạy vẽ biểu đồ**:
  Sau khi xuất các file kết quả `.csv` vào thư mục `results/`, chỉ cần chạy lệnh sau để cập nhật biểu đồ thật:
  ```bash
  python fuzzysleeper/plots.py
  ```

---
*Mọi đoạn code đều đã được mình căn chỉnh và format sạch sẽ bằng `ruff`. Chúc hai bạn chạy thực nghiệm trên GPU thành công tốt đẹp!*
