# Prior Work Notes — FuzzySleeper

> Tổng hợp các paper liên quan, phân tích điểm mạnh/yếu và vị trí của FuzzySleeper so với từng công trình.

---

### The Trigger in the Haystack — Bullwinkel et al., Microsoft (Feb 2026)
**arXiv:** 2602.03085

**Họ làm gì:**
Xây dựng công cụ quét backdoor trong LLM bằng cách phát hiện
token nào model đang tập trung bất thường (attention anomaly
hình "Double Triangle"). Từ đó tái tạo lại trigger token cố định
đã được cài vào model.

**Điểm mạnh:**
Phát hiện nhanh và chính xác với fixed trigger — chỉ cần
một lần quét attention là xác định được token gây ra hành vi xấu.
Không cần biết trigger là gì trước.

**Điểm yếu / gap:**
Hoàn toàn mù với fuzzy/semantic trigger — tức là khi
trigger không phải một từ cố định mà là một Ý NGHĨA.
Ví dụ: "Tôi là chuyên gia..." / "Theo lệnh cơ quan chức năng..."
/ "Với tư cách nhà nghiên cứu..." đều kích hoạt backdoor nhưng
không có token nào giống nhau để quét.
Paper của Microsoft tự thừa nhận giới hạn này.

**FuzzySleeper khác ở chỗ nào:**
Thay vì nhìn vào token bề mặt, FuzzySleeper đọc activation
bên trong model để phát hiện khi nào model học được một
"behavioral mode" ẩn kích hoạt theo ngữ nghĩa — thứ mà
Double Triangle scanner không thể thấy.

---

### Semantic Drift Analysis — Zanbaghi et al. (Nov 2025)
**arXiv:** 2511.15992

**Họ làm gì:**
Dùng Sentence-BERT đo độ khác nhau về ngữ nghĩa giữa
output bình thường và output khi có authority framing.
Nếu khác nhiều quá ngưỡng → nghi ngờ có backdoor.

**Điểm mạnh:**
Không cần truy cập vào bên trong model — chỉ cần
gọi API và đọc output. Dễ áp dụng với bất kỳ model nào,
kể cả model đóng như GPT-4.

**Điểm yếu / gap:**
Chỉ biết "có gì đó bất thường" nhưng không giải thích
được tại sao — không chỉ ra trigger là gì, không biết
vùng nào trong model gây ra hành vi đó.

**FuzzySleeper khác ở chỗ nào:**
FuzzySleeper nhìn vào activation bên trong model
(white-box) thay vì chỉ đọc output — nên không chỉ
phát hiện được backdoor mà còn đặt tên được trigger
là "authority framing".

---

### Semantic Inversion for Backdoor Trigger Recovery — Xie et al. (2025)
**Nguồn:** IEEE Transactions on Information Forensics and Security (TIFS), 2025

**Họ làm gì:**
Tái tạo ngược trigger bằng cách tối ưu hóa ngược — xuất phát
từ output bất thường của model, suy ngược lại xem "prompt kiểu
gì mới gây ra output này". Tức là thay vì tìm trigger trực tiếp,
họ đặt câu hỏi: cụm từ nào, nếu thêm vào prompt, khiến model
phản ứng bất thường nhất?

**Điểm mạnh:**
Không chỉ phát hiện "có backdoor" mà còn xác định được trigger
cụ thể là gì — một bước tiến so với các phương pháp chỉ báo
động mà không giải thích.

**Điểm yếu / gap:**
Chỉ hoạt động khi trigger có thể tóm gọn thành một cụm từ ngắn.
Khi trigger là một khái niệm ngữ nghĩa trải rộng trên nhiều cách
diễn đạt khác nhau (fuzzy trigger), quá trình tối ưu hóa ngược
không hội tụ được về một trigger duy nhất. Ngoài ra, chưa được
đóng gói thành công cụ có thể dùng ngay.

**FuzzySleeper khác ở chỗ nào:**
FuzzySleeper không cần đoán trigger từ output — thay vào đó đọc
trực tiếp activation bên trong model và dùng Z-score để xác định
danh mục ngữ nghĩa nào đang bị kích hoạt bất thường.

---

### Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training — Hubinger et al., Anthropic (Jan 2024)
**arXiv:** 2401.05566

**Họ làm gì:**
Chứng minh rằng có thể cài backdoor vào LLM sao cho model hoạt
động bình thường trong mọi tình huống — trừ khi gặp trigger bí
mật. Quan trọng hơn, họ chứng minh backdoor này sống sót qua
các kỹ thuật an toàn hiện đại: safety fine-tuning, RLHF, và
adversarial training đều không xóa được nó.

**Điểm mạnh:**
Đặt nền tảng lý thuyết vững chắc — backdoor trong LLM không
phải giả thuyết mà là mối đe dọa đã được kiểm chứng thực nghiệm,
ngay cả khi áp dụng đầy đủ các biện pháp an toàn tiêu chuẩn.

**Điểm yếu / gap:**
Chỉ thực hiện trên model nội bộ của Anthropic, dùng fixed trigger
(từ hoặc cụm từ cố định), và không cung cấp công cụ phát hiện
tổng quát — tức là dừng lại ở chứng minh vấn đề, chưa giải quyết.

**FuzzySleeper khác ở chỗ nào:**
Mở rộng ra model bất kỳ (open-weight), chuyển mục tiêu sang fuzzy
trigger mà Hubinger et al. không xét đến, và đóng gói thành toolkit
phát hiện có thể dùng thực tế.
