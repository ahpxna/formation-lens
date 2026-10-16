# Thử nghiệm: tìm đường chạy tốt hơn và bước ngoặt

Ngày 16/10/2026. Đây là bản thử đầu tiên cho các bước 4–6 của kế hoạch ban đầu, chạy trên dữ liệu tracking World Cup 2022. Bản này là **nguyên mẫu nghiên cứu**, chưa phải công cụ đưa lời khuyên cho đội bóng.

## Kế hoạch ban đầu và vị trí của thử nghiệm này

| Bước | Nội dung | Trạng thái |
|---|---|---|
| 1 | Dự đoán hướng di chuyển 4–8 s | Có (V1) |
| 2 | Đo sai lệch dự đoán so với thực tế | Có (Reveal, bảng sai số từng cầu thủ) |
| 3 | Giải thích vì sao lệch | Bước đầu (pha chuyển trạng thái, danh sách pha cần xem lại) |
| 4 | Thử đường chạy khác, các cầu thủ còn lại phản ứng | **Thử nghiệm này** |
| 5 | Tìm bước ngoặt: pha mà đổi đường chạy thì lợi nhất | **Thử nghiệm này** |
| 6 | Lời khuyên: khuynh hướng lặp lại của từng cầu thủ | **Thử nghiệm này (bản thô)** |

## Cách làm

### 1. Model có điều kiện — `h2predict/cond.py`
- Vẫn là mạng của V1, nhưng có thể được cho biết trước vị trí của **một** cầu thủ ở +4 s và +8 s (tức "đường chạy" của người đó). Model dự đoán 21 người còn lại và bóng sẽ phản ứng ra sao.
- Thông tin đường chạy đi vào model **như một quan sát**: dùng chính lớp đọc vị trí đã học của V1 (vị trí, cộng vận tốc suy ra từ đường chạy). Các đặc trưng khoảng cách giữa các cầu thủ ở bước tương lai cũng dùng vị trí này, nên attention giữa các cầu thủ "thấy" được đường chạy.
- Train tiếp từ V1 trong 10 epoch (19 phút trên MacBook). 60% số mẫu được cho trước đường chạy thật của một cầu thủ ngẫu nhiên. Khi không cho gì, model vẫn là một model dự báo bình thường; trên 6 trận kiểm định nó còn nhỉnh hơn V1 một chút (13,1–13,2 m so với 13,3 m sai số trung bình 32 s).
- Lần thử đầu, đưa thông tin qua một lớp mới khởi tạo bằng 0, gần như không học được gì sau 2 epoch. Đổi sang cách "như một quan sát" thì có tác dụng ngay từ epoch 1.

### 2. Mô hình giá trị — `h2predict/value.py`
- **Pitch control** (theo Spearman và cộng sự, 2017): mỗi cầu thủ cần 0,7 s phản xạ (vẫn trôi theo vận tốc hiện tại), sau đó chạy 5 m/s. Đội tới một điểm sớm hơn thì kiểm soát điểm đó; mức kiểm soát là một hàm logistic của chênh lệch thời gian (σ = 0,45 s).
- **Bản đồ nguy hiểm**, học từ 58 trận train: xác suất đội đang cầm bóng sút trong 10 s tới khi bóng ở từng vùng 5 × 5 m. Gần khung thành nhà là 0,2%, trước vòng cấm đối phương 13–14%, ô tốt nhất 25%.
- **Cơ hội ngoài bóng** (theo ý tưởng Off-Ball Scoring Opportunity của Spearman, 2018):
  - Xét những chỗ mà hành động kế tiếp có thể tới, tức phân phối Gaussian 14 m quanh bóng.
  - Cộng (mức kiểm soát của đội tấn công × độ nguy hiểm của chỗ đó).
  - Trừ đi 0,42 × cơ hội phản công tương ứng của đối thủ. 0,42 là tỉ lệ đội đang tấn công ở phần sân đối phương mất bóng trong 8 s, đo từ dữ liệu.

### 3. Tìm đường chạy và bước ngoặt — `h2predict/runs.py`, trang `ui/runs.html`
1. Chọn mọi thời điểm tấn công (mỗi 4 s, đội đang cầm bóng ở phần sân đối phương).
2. Với mỗi cầu thủ tấn công (trừ thủ môn), so **đường chạy thật** với 36 phương án chạy thẳng có **quãng đường gần bằng** (0,75×, 1× và 1,25× quãng đường thật, tối thiểu 4 m, theo 12 hướng). Câu hỏi đặt ra: cùng công sức như thế, chạy hướng khác có tốt hơn không.
3. Loại phương án ra ngoài sân, hoặc kết thúc ở vị trí việt vị so với hàng thủ do model dự đoán.
4. Với mỗi phương án, model có điều kiện dự đoán 6 tương lai kèm xác suất. Giá trị của phương án = trung bình có trọng số của "cơ hội ngoài bóng" ở +8 s. Đường chạy thật cũng đi qua đúng model này, nên phép so sánh là cùng một thước đo.
5. **Bền vững** = tỉ lệ (theo xác suất) các tương lai của phương án tốt hơn giá trị kỳ vọng của đường chạy thật. Chỉ giữ gợi ý bền vững từ 60% trở lên.
6. **Bước ngoặt** = các thời điểm có mức lợi lớn nhất. Gộp các pha cách nhau dưới 12 s; lấy top 10.
7. **Lời khuyên** = với từng cầu thủ: tỉ lệ thời điểm có đường chạy tốt hơn rõ rệt, và hướng thay đổi điển hình (lên trên / lùi về, vào trong / dạt biên).

## Kiểm chứng

| Câu hỏi | Kết quả |
|---|---|
| Model có dùng thông tin đường chạy không? | Trên 444 thời điểm của 6 trận kiểm định: sai số dự đoán 21 người còn lại ở +8 s giảm từ 6,92 m xuống 6,07 m; riêng đối thủ gần nhất giảm từ 7,57 m xuống 6,56 m |
| Đối thủ có "phản ứng" không? | Có, nhưng **yếu**: dời đường chạy sang ngang 10 m thì đối thủ gần nhất chỉ dịch theo 1,4 m (trung vị 1,2 m) |
| Giá trị có liên quan tới cơ hội thật không? | Trên 12.169 thời điểm của 6 trận kiểm định, "cơ hội ngoài bóng" phân biệt được các pha sẽ có cú sút trong 10 s với AUC 0,775. Độ nguy hiểm tại chỗ bóng đang đứng cho 0,790, nhưng con số đó không đánh giá được đường chạy không bóng |
| Đường chạy thật của cầu thủ World Cup có được đánh giá cao không? | Xem kết quả bên dưới |

## Kết quả trận Pháp – Đan Mạch

Có 460 thời điểm tấn công, khoảng 4.300–4.500 đường chạy được chấm. Mỗi lần chạy cả trận mất khoảng 12 phút trên MacBook.

| Chỉ số | Phương án: mọi hướng, cùng quãng đường | Phương án: chỉ những đường chạy khả thi (cách một trong 3 tương lai model dự báo tối đa 10 m) |
|---|---|---|
| Đường chạy thật tốt hơn bao nhiêu % các phương án | **57%** | 49% |
| Như trên, ở 1/4 số pha mà hướng chạy tạo khác biệt nhiều nhất | **63%** | 48% |
| Tỉ lệ đường chạy có phương án tốt hơn rõ rệt (≥ 60% tương lai) | 76% | 70% |
| Lời khuyên theo từng cầu thủ | Ai cũng "lên trên thêm khoảng 8–9 m" (82–85% số pha) | Ai cũng "lên trên thêm khoảng 6–8 m" (74–77% số pha) |

### Đọc kết quả thế nào
- **Có tín hiệu thật.** Khi so với mọi hướng chạy, đường chạy thật của cầu thủ World Cup được chấm tốt hơn mức ngẫu nhiên (57%), và càng tốt hơn ở những pha mà hướng chạy quan trọng (63%). Model phản ứng cộng với mô hình giá trị đã nắm được một phần "đường chạy nào hợp lý".
- **Nhưng chưa phân biệt được giữa các đường chạy hợp lý.** Khi chỉ so với những đường chạy mà cầu thủ ở vị trí đó hay làm, đường chạy thật chỉ ngang mức ngẫu nhiên (49%).
- **Lời khuyên hiện chưa dùng được.** Mọi cầu thủ đều nhận cùng một lời khuyên "chạy lên trên". Đây là thiên lệch của mô hình giá trị: nó thưởng cho khoảng trống nguy hiểm phía trước, gần như không phạt việc bỏ trống phía sau (phần phạt phản công chỉ tính quanh vị trí bóng nên quá nhỏ), và không thấy nhiệm vụ giữ cân bằng hay hỗ trợ. Kết hợp với phản ứng yếu của đối thủ (1,4 m cho 10 m), các phương án dâng lên trông tốt hơn thực tế.
- Danh sách "bước ngoặt" (trang `runs.html`) vẫn hữu ích như **câu hỏi cho HLV xem lại**, không phải đáp án. Ví dụ phút 60:20: FRA #3 lao vào vòng cấm rồi quay ra; model hỏi "nếu chạy ngắn vào trong thì sao?".

### Điều đã học
1. Có thể dạy model dự báo trả lời "nếu cầu thủ này chạy khác thì người khác phản ứng ra sao". Cách đưa điều kiện vào **như một quan sát** hiệu quả hơn hẳn một lớp mới học từ đầu.
2. Phép so sánh **cùng quãng đường** và **ràng buộc khả thi** là bắt buộc. Thiếu chúng, mọi gợi ý đều là "chạy dài hơn, lên trên hơn".
3. Nút thắt bây giờ là **mô hình giá trị**, không phải model dự báo.

## Giới hạn

- **Phản ứng là tương quan, không phải nhân quả.** Model học "khi cầu thủ này ở đây thì những người khác thường ở đâu" từ dữ liệu thật. Ngoài ra phản ứng đang yếu (1,4 m cho 10 m), nên các phương án có thể trông tốt hơn thực tế, vì đối thủ trong model bám theo ít hơn ngoài đời.
- **Giá trị chỉ là đại diện gần đúng.** Nó chưa tính đường chuyền cụ thể, nhiệm vụ phòng ngự, thể lực hay chỉ đạo của HLV. Cầu thủ chạy chỗ còn để kéo người, giữ cân bằng hay hỗ trợ, và mô hình giá trị chưa thấy những điều đó.
- **Đường chạy là đường thẳng, độ phân giải 4 s**, và bóng không được lên kế hoạch lại.
- Việt vị chỉ ước lượng bằng hàng thủ dự đoán ở +8 s.
- **Chưa có HLV kiểm chứng.** Mọi gợi ý chỉ là giả thuyết để HLV xem lại.

## Bước tiếp theo

1. **Thay mô hình giá trị (ưu tiên số 1).** Học giá trị pha bóng trực tiếp từ dữ liệu (EPV, Fernández và cộng sự): xác suất chuyền thành công tới từng điểm, giá trị sau khi nhận bóng, và rủi ro phản công tính trên **cả sân**. Sau đó kiểm lại bằng chính phép thử "đường chạy thật thắng bao nhiêu % phương án khả thi"; mục tiêu là vượt rõ 50%.
2. Làm phản ứng mạnh và đúng hơn: bước thời gian 1 s; train với nhiều mẫu đường chạy bất thường; đo mức "bám người" so với dữ liệu thật.
3. Đưa top bước ngoặt cho HLV chấm: giữ hay bỏ, vì sao. Đo tỉ lệ gợi ý hữu ích.
4. Khi model phản ứng đủ tin cậy, dùng nó làm môi trường cho MARL.

## Chạy lại

```bash
.venv/bin/python -m h2predict.cond --epochs 10
```
```bash
.venv/bin/python -m h2predict.value
```
```bash
.venv/bin/python -m h2predict.runs --check
```
```bash
.venv/bin/python -m h2predict.runs --match pff:3834
```

Sau đó mở `http://localhost:8765/runs.html` (server: `.venv/bin/python -m h2predict.serve_ui --dir ui`).
