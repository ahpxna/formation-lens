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

## EPV học từ dữ liệu — `h2predict/epv.py`

**Định nghĩa.** Từ một khoảnh khắc (vị trí và vận tốc của 22 người cùng bóng):

V = P(đội cầm bóng sút trong 10 s) − P(đối thủ sút trong 10 s)

Vế trừ đi để mô hình tự học rủi ro phản công của một đội hình mất cân bằng; đây chính là điều mà giá trị làm thủ công thiếu. Mô hình dùng cú sút thay vì bàn thắng, vì 58 trận có khoảng 1.400 cú sút nhưng chỉ khoảng 150 bàn. Tinh thần giống EPV của Fernández, Bornn và Cervone, nhưng chưa tách thành chuyền / dẫn bóng / sút.

**Mạng.** Một set transformer nhỏ (3 lớp, 64 chiều, 4 head), dùng lại phần "cạnh" giữa các cầu thủ của model dự báo. Train trên 237.912 khoảnh khắc của 58 trận, mất 3 phút.

**Kiểm chứng trên 24.377 khoảnh khắc của 6 trận chưa học:**

| Thước đo | Dự báo đội cầm bóng sút trong 10 s | Dự báo đối thủ sút trong 10 s |
|---|---|---|
| Giá trị thủ công (cơ hội ngoài bóng) | AUC 0,775 | — |
| Độ nguy hiểm tại chỗ bóng | 0,790 | — |
| **EPV** | **0,838** | **0,856** |
| Tần suất: dự báo trung bình / thực tế | 4,0% / 4,3% | 0,8% / 1,1% |

### Dùng EPV để chấm đường chạy (Pháp – Đan Mạch, chỉ các phương án khả thi)

| Chỉ số | Giá trị thủ công | EPV |
|---|---|---|
| Đường chạy thật tốt hơn bao nhiêu % phương án | 49% | 48% |
| Như trên, ở 1/4 số pha mà hướng chạy quan trọng nhất | 48% | 42% |
| Như trên, nhưng **không cho ai phản ứng** | — | 48% |
| Theo tuyến | — | hậu vệ 48%, tiền vệ 46%, tiền đạo 51% |
| Lời khuyên theo từng cầu thủ | "lên trên 6–8 m" cho mọi người | "lên trên 7–8,5 m" cho mọi người; gợi ý mạnh nhất thường là đưa thêm người vào vòng cấm |

### Đọc kết quả
- **EPV là một hàm giá trị tốt hơn hẳn**: dự báo cú sút tốt hơn, khớp tần suất thật, và học được cả rủi ro phản công.
- **Nhưng dùng EPV để chấm đường chạy vẫn cho kết quả ngang mức ngẫu nhiên**, và lời khuyên vẫn là "dâng lên".
- **Thủ phạm chính không phải phản ứng yếu.** Cho mọi người đứng yên (không phản ứng) thì kết quả gần như y hệt (48% so với 48%).
- **Thủ phạm là cách dùng hàm giá trị.** Một hàm giá trị học từ **tương quan** được đem trả lời câu hỏi **phản thực** ("nếu một người chạy khác"). Trong dữ liệu, có cầu thủ trong vòng cấm thường đi kèm những đợt tấn công đang tốt; đặt thêm một người vào đó không tự tạo ra cơ hội như vậy.
- **Kết luận: lớp lời khuyên tự động chưa nên dùng.** Những gì đáng tin hôm nay là: dự báo 4–8 s, EPV dùng để đánh giá trạng thái thật, và danh sách pha cần xem lại. Trang `runs.html` hiển thị gợi ý như **câu hỏi để HLV xem lại** và ghi rõ điều này ngay trên trang.

## Hướng 3: chấm đường chạy qua cơ chế chuyền bóng — `h2predict/options.py`

### Vì sao không cho EPV "học nhân quả" trực tiếp?
- Dữ liệu tracking là **quan sát**. EPV học "tình huống như thế này thường dẫn tới cú sút", chứ không học được "nếu một người chạy khác thì cơ hội đổi bao nhiêu".
- Muốn có tác động nhân quả thì cần một trong ba thứ:
  1. Thí nghiệm can thiệp. Không làm được với trận thật.
  2. Giả định không còn yếu tố gây nhiễu nào bị bỏ sót. Nhưng ý đồ, chỉ đạo chiến thuật hay độ mệt thì tracking không thấy.
  3. Một **cơ chế quan sát được**.
- Hướng 3 chọn cách thứ ba. Đường chạy tạo giá trị vì nó tạo ra **lựa chọn chuyền** cho người cầm bóng, và **quyết định chuyền cho ai** thì có trong dữ liệu, nên kiểm chứng được.

### Cách làm
1. **Tách đường chuyền thật từ tracking.**
   - Người chuyền là cầu thủ gần bóng nhất lúc có sự kiện PASS.
   - Theo dõi quỹ đạo bóng: người đầu tiên chạm trong phạm vi 1,5 m là người nhận (đồng đội thì thành công, đối thủ thì bị cắt); bóng ra ngoài thì thất bại.
   - Người nhận dự định của đường chuyền hỏng là đồng đội nằm đúng hướng bóng bay.
   - Kết quả: 36.554 đường chuyền để train, 3.752 để kiểm định; tỉ lệ thành công 77,5%, đúng với thực tế.
   - Bản đầu dùng điểm bóng dừng làm mục tiêu và bị **rò rỉ đáp án** (AUC 0,989), vì với đường bị cắt, điểm dừng chính là chỗ hậu vệ chạm bóng. Đã sửa bằng cách chỉ dùng thông tin có tại lúc chuyền.
2. **Xác suất chuyền thành công** tới vị trí của đồng đội lúc bóng tới, dựa trên đặc trưng thời gian tới điểm (time-to-intercept):

   | | AUC | Brier | Dự báo trung bình / thực tế |
   |---|---|---|---|
   | Đầy đủ đặc trưng | **0,787** | 0,141 | 77,9% / 77,7% |
   | Chỉ khoảng cách | 0,623 | | |

3. **Giá trị lựa chọn** = P(thành công) × EPV(bóng ở đó, ta cầm) − P(hỏng) × EPV(bóng ở đó, đối thủ cầm).
4. **Mô hình lựa chọn** (conditional logit): người cầm bóng chọn đồng đội nào, học từ quyết định thật. Kết quả trên 3.752 đường chuyền chưa học (đoán bừa đúng khoảng 10%):

   | Mô hình | Đúng top-1 | Top-3 | Log-loss |
   |---|---|---|---|
   | Chỉ độ an toàn | 29,8% | 59,5% | 2,032 |
   | Chỉ khoảng cách | 26,7% | 59,8% | 2,014 |
   | An toàn + khoảng cách | 29,7% | 62,5% | 1,967 |
   | **Đầy đủ: + EPV, rủi ro, tiến lên, bề ngang** | **31,8%** | **66,5%** | **1,899** |
   | Giá trị lựa chọn, không học từ quyết định | 13,9% | 35,7% | |

   - **Thêm giá trị EPV và rủi ro làm dự đoán quyết định chuyền thật tốt hơn rõ.** Tức là cầu thủ thật có cân nhắc giá trị chứ không chỉ chọn đường an toàn. Đây là bằng chứng quan sát được cho cơ chế.
   - Nếu chỉ dùng "giá trị lựa chọn" thuần túy để đoán thì gần như ngẫu nhiên (13,9%). Cầu thủ thận trọng hơn công thức nhiều.
   - Trọng số (theo độ lệch chuẩn): an toàn +0,73; EPV nếu nhận được +0,81; khoảng cách −0,41. "EPV của đối thủ nếu mất bóng" lại +1,56. Đó là thói quen **chuyền về hàng thủ**, vùng mà nếu mất bóng thì rất nguy hiểm nhưng chuyền về đó gần như luôn an toàn. Mô hình **mô tả** cách cầu thủ chọn, không phải lời khuyên nên chọn gì.
5. **Giá trị của một đường chạy** = giá trị kỳ vọng của đường chuyền kế tiếp ở +8 s, tức tổng qua các đồng đội của P(được chọn) × giá trị lựa chọn. Chạy bằng `--value pass`.

### Phản ứng của hậu vệ: model so với thực tế
Trong 2.253 pha tấn công thật của 6 trận kiểm định, so với độ dịch chung của cả đội, hậu vệ gần nhất chỉ bám theo **0,28 m (dọc) và 0,22 m (ngang)** cho mỗi mét người tấn công di chuyển. Model có điều kiện cho 0,14 m/m theo chiều ngang: yếu hơn thực tế khoảng một phần ba, nhưng không chênh nhiều. Hậu vệ World Cup phần lớn giữ khu vực chứ không kèm người.

### Kết quả trên trận Pháp – Đan Mạch (chỉ các phương án khả thi)

| Chỉ số | Giá trị thủ công | EPV | Giá trị đường chuyền kế tiếp |
|---|---|---|---|
| Đường chạy thật tốt hơn bao nhiêu % phương án | 49% | 48% | 47% |
| Như trên, ở pha mà hướng chạy quan trọng nhất | 48% | 42% | 42% |
| Như trên, **không ai phản ứng** | — | 48% | 47% |
| Sau đường chạy thật: P(được chọn) có dự đoán đúng ai nhận bóng 8–12 s sau không | — | — | AUC 0,54 (tỉ lệ gốc 9%) |
| Lời khuyên | "lên trên 6–8 m" | "lên trên 7–8,5 m" | "lên trên 7–8,5 m" |

## Kết luận chung của thử nghiệm

1. **Từng thành phần đều chạy và đã được kiểm chứng trên dữ liệu chưa học:**
   - Model phản ứng: biết đường chạy của một người thì đoán những người khác tốt hơn.
   - EPV: AUC 0,84 / 0,86, khớp tần suất thật.
   - Xác suất chuyền thành công: AUC 0,79.
   - Mô hình lựa chọn: thêm giá trị làm dự đoán quyết định chuyền thật tốt hơn.
2. **Ghép lại để "tìm đường chạy tốt hơn" thì cả ba cách chấm đều cho cùng một kết quả:** đường chạy thật chỉ ngang hoặc dưới mức ngẫu nhiên so với các phương án khả thi, và lời khuyên đồng loạt là "dâng lên".
3. **Thủ phạm là mục tiêu tối ưu, không phải mô hình.** Phép thử "không ai phản ứng" giữ nguyên mọi người xung quanh, chỉ đổi vị trí người chạy, mà đường chạy thật vẫn thua. Nghĩa là mục tiêu "giá trị tấn công trong 8 s tới" luôn thích vị trí gần khung thành hơn. Cầu thủ thật đứng chỗ khác vì những nhiệm vụ mục tiêu này không thấy: hỗ trợ gần bóng, che phía sau khi mất bóng, canh thời điểm để không việt vị, giữ cấu trúc đội hình.
4. **Hệ quả:** chưa thể đưa lời khuyên tự động. Thứ dùng được là danh sách pha để HLV đặt câu hỏi, cùng các thành phần đã kiểm chứng (dự báo, EPV, xác suất chuyền, mô hình lựa chọn).

### Hướng đi tiếp có cơ sở
- **Học mục tiêu từ chính cầu thủ** (inverse RL / học sở thích): thay vì tự đặt mục tiêu, suy ra mục tiêu mà đường chạy thật đang tối ưu (gồm cả nhiệm vụ phòng ngự và hỗ trợ), rồi mới tìm chỗ lệch. Thước đo: đường chạy thật phải thắng rõ hơn 50% các phương án khả thi.
- **Mục tiêu do HLV đặt cho từng vai trò**, ví dụ "tiền vệ trụ: luôn là lựa chọn chuyền an toàn + che khoảng trống trung tâm". Khi đó tối ưu hóa có ý nghĩa vì mục tiêu đúng với ý đồ.
- **Kiểm chứng bằng kết quả trên nhiều trận**: những pha mà đường chạy thật lệch xa phương án "tốt" có dẫn tới ít cơ hội hơn không.

## Hướng 1: học mục tiêu từ chính cầu thủ (inverse RL) — `h2predict/irl.py`

**Ý tưởng.**
- Mỗi đường chạy thật dài ít nhất 4 m trong 8 s được coi là một **lựa chọn**: cầu thủ có thể chạy cùng quãng đường đó theo 12 hướng khác, và đã chọn một hướng.
- Học một **hàm mục tiêu**, là tổng có trọng số của 11 yếu tố về nơi đường chạy kết thúc, sao cho các lựa chọn thật có xác suất cao nhất.
- Đây là maximum-entropy IRL một bước (Ziebart và cộng sự, 2008), tương đương một conditional logit.
- Trọng số được học riêng cho **6 nhóm**: có bóng / mất bóng × hậu vệ / tiền vệ / tiền đạo. Nhờ vậy nhiệm vụ phòng ngự cũng được học.

**11 yếu tố:**
- độ sâu, khoảng cách tới bóng, lệch khỏi vị trí quen thuộc;
- khoảng cách tới đối thủ gần nhất, tới đồng đội gần nhất;
- ở phía sau bóng (goal-side), độ gần trục giữa;
- EPV của đội, sẵn sàng nhận đường chuyền, việt vị;
- khoảng cách tới đường nối bóng–khung thành nhà.

**Chống rò rỉ đáp án:**
- Vị trí của mọi người khác ở +8 s lấy từ dự báo **không biết đường chạy thật**, và giống nhau cho mọi phương án.
- Mọi phương án có **cùng quãng đường**.
- Chấm điểm trên **6 trận chưa dùng để học**.

**Kết quả** (học từ 24 trận, 262.276 lựa chọn; kiểm tra trên 6 trận, 65.315 lựa chọn; đoán bừa = 50%):

| Cách chấm | Đường chạy thật tốt hơn bao nhiêu % phương án |
|---|---|
| **Mục tiêu học được (IRL)** | **66,0%**, đều ở cả 6 nhóm (65–67%); riêng Pháp – Đan Mạch 67,1% |
| Chỉ EPV của đội | 52,1% |
| Tham chiếu: gần nhất với vị trí model dự báo cho người đó | 79,8% |

**Đạt tiêu chí "thắng rõ hơn 50%"**, và ổn định: học từ 3 trận hay 24 trận cho cùng một con số.

Model dự báo vẫn đoán đúng hơn (80%), vì nó dùng toàn bộ lịch sử và thói quen riêng, nhưng không giải thích được. Mục tiêu học được giải thích được khoảng 2/3 hành vi bằng 11 yếu tố có tên.

**Cầu thủ đang tối ưu điều gì** (trọng số theo độ lệch chuẩn; + là họ chạy về phía có nhiều yếu tố đó hơn):

| Nhóm | Yếu tố nổi bật | Diễn giải |
|---|---|---|
| Có bóng, hậu vệ | lệch vị trí −0,43; đối thủ gần nhất −0,37; xa bóng −0,30; giãn khỏi đồng đội +0,22 | Giữ vị trí, ở gần nhịp chơi, giãn đội hình |
| Có bóng, tiền vệ | xa bóng −0,34; lệch vị trí −0,32; đối thủ gần nhất −0,26; phía sau bóng −0,21 | Hỗ trợ gần bóng, ở trước bóng |
| Có bóng, tiền đạo | phía sau bóng −0,32; xa bóng −0,28; đối thủ gần nhất −0,28; độ sâu +0,24 | Đứng trước bóng, dâng lên, gần nhịp chơi |
| Mất bóng, hậu vệ | lệch vị trí −0,44; **phía sau bóng +0,42**; độ sâu −0,31; đối thủ gần nhất −0,31; lệch đường bóng–khung thành −0,28 | **Che phía sau, lùi sâu, kèm người, đứng trên đường bóng–khung thành** |
| Mất bóng, tiền vệ | lệch vị trí −0,39; xa bóng −0,34; phía sau bóng +0,25; lệch đường bóng–khung thành −0,22 | Áp sát bóng nhưng vẫn che phía sau |
| Mất bóng, tiền đạo | **xa bóng −0,39**; lệch vị trí −0,35; đối thủ gần nhất −0,24 | **Pressing**: áp sát bóng |

Lưu ý khi đọc: các trọng số là tác động riêng phần giữa những yếu tố có liên quan với nhau. Ví dụ "đối thủ gần nhất −" ở tuyến tấn công nghĩa là họ ở gần nhịp chơi, không phải cố tình lao vào đối thủ.

**Tìm chỗ lệch.** Với mỗi đường chạy, mô hình tìm phương án tốt nhất theo **chính mục tiêu của nhóm người đó**. Độ chênh được giải thích bằng các yếu tố đóng góp nhiều nhất, rồi đối chiếu với EPV. Ví dụ trên trận Pháp – Đan Mạch:
- **95:44, DEN #6** (tiền đạo, đội đang mất bóng): chạy dạt xa bóng. Mục tiêu của nhóm muốn áp sát bóng và giữ vị trí quen. EPV cũng nghiêng về phương án áp sát (+0,003), nên **đáng xem lại**.
- **60:32, FRA #7** (hậu vệ, đội đang mất bóng): lệch vị trí và không đứng phía sau bóng như chuẩn của nhóm. Nhưng đường chạy thật giữ EPV cao hơn (phương án "chuẩn" kém hơn 0,057), nên **có thể là lựa chọn có chủ đích**.
- **Khuynh hướng theo cầu thủ:** FRA #1 thấp hơn rõ so với mục tiêu của nhóm ở 35% trong 454 đường chạy, chủ yếu vì xa bóng và xa đối thủ. DEN #8 ở 33%, chủ yếu vì xa đối thủ và lệch vị trí.
- Trong 10% chỗ lệch lớn nhất, phương án "chuẩn" cũng có EPV cao hơn ở **46%** số trường hợp. Như vậy chỗ lệch tự nó **không** có nghĩa là sai. Khi cả mục tiêu học được và EPV cùng chỉ một hướng, đó là pha đáng ưu tiên cho HLV xem.

Trang `runs.html` mặc định mở phần này ("FRA-DEN · learned objective"): mỗi pha có đường chạy thật, phương án theo chuẩn của nhóm, lý do và kết luận theo EPV.

**Giới hạn của hướng 1:**
- Mục tiêu là **chuẩn trung bình của cầu thủ World Cup cùng nhóm**, không phải ý đồ của một HLV cụ thể. Lệch chuẩn có thể là chỉ đạo riêng hoặc sáng tạo.
- Vai trò mới chia thô thành hậu vệ / tiền vệ / tiền đạo. Roster của PFF có vị trí chi tiết (RW, RB, CM…) để chia mịn hơn.
- Vẫn là đường chạy thẳng 8 s.

**Bước tiếp theo của hướng 1:**
1. Chia vai trò theo vị trí chi tiết trong roster.
2. Học mục tiêu **riêng cho từng đội**, rồi so một đội với chuẩn của chính nó (phong cách) thay vì chuẩn chung.
3. Đưa top chỗ lệch cho HLV chấm (giữ hay bỏ) để đo độ hữu ích.
4. Dùng mục tiêu học được làm phần thưởng cho MARL: tác nhân học được "chạy như cầu thủ thật" trước, rồi mới tối ưu thêm.

## Hướng 1, tiếp theo: vai trò chi tiết, phong cách đội, MARL một bước và nhiều bước

Phần này dùng các tập lựa chọn tính sẵn cho **cả 64 trận** (`python -m h2predict.irl --cache`): 58 trận để học, 6 trận để kiểm tra.

### 1. Vai trò chi tiết — `h2predict/roles.py`
- Bộ chuyển đổi không lưu số áo, nên bản đồ số áo → ô được **dựng lại bằng cách khớp vị trí 10 giây đầu** trong tracking gốc với các ô (thuật toán Hungarian). Kết quả: 1.408/1.408 ô, sai số trung vị 0,14 m.
- Thêm **563 lần thay người** từ dữ liệu sự kiện. Mỗi ô ở mỗi thời điểm đều biết là ai, đá vị trí gì.
- Ví dụ lỗi được sửa: "DEN #6 phút 95:44" trước đây bị chấm theo chuẩn tiền đạo, nhưng thật ra là Nørgaard, tiền vệ phòng ngự vào thay người chạy cánh.
- 7 vai trò: trung vệ, hậu vệ biên (gồm wing-back), tiền vệ phòng ngự, tiền vệ trung tâm, tiền vệ tấn công, cánh (LW/RW/LM/RM), tiền đạo. Nhân với có bóng / mất bóng thành 14 nhóm.

| Cách chia nhóm (học 58 trận, kiểm tra 6 trận) | Đường chạy thật thắng | Log-loss |
|---|---|---|
| Theo tuyến (6 nhóm) | 66,0% | 2,1966 |
| **Theo vai trò chi tiết (14 nhóm)** | **66,6%** | **2,1914** |

Cải thiện có, nhưng nhỏ. Trận Pháp – Đan Mạch: 68,1%.

### 2. Phong cách đội — `h2predict/styles.py`
- Mục tiêu của một đội = chuẩn chung của nhóm + **độ lệch riêng của đội**, bị kéo về 0 (partial pooling). Độ lệch chỉ học trên các trận train của đội đó.
- Kiểm tra trên 65.311 đường chạy ở 6 trận chưa học: mục tiêu riêng của đội làm **log-loss giảm 0,024 mỗi đường chạy** (2,1914 → 2,1672), và đường chạy thật thắng 67,0% thay vì 66,6%. Mức giảm tăng theo lượng dữ liệu: 0,019 khi có 21 trận, 0,024 khi có 58 trận.
- Kết luận: **phong cách đội là thật và giữ được từ trận này sang trận khác.**

| Đội | Phong cách nổi bật (so với chuẩn chung, theo độ lệch chuẩn) |
|---|---|
| Pháp | Khi mất bóng, trung vệ và hậu vệ biên đứng **phía sau bóng** nhiều hơn rõ (+0,50 / +0,47): lùi sâu rồi phản công |
| Thụy Sĩ | Hàng thủ lùi sâu (+0,44); tiền vệ tấn công pressing (−0,43) |
| Hà Lan | Tiền đạo luôn đứng trước bóng (−0,38); tiền vệ tấn công pressing (−0,33) |
| Qatar | Hậu vệ biên thu hẹp vào trong (−0,41); tiền vệ trung tâm áp sát bóng (−0,35) |
| Bỉ | Hậu vệ biên và trung vệ lùi sâu khi mất bóng |
| Uruguay, Ghana | Tiền vệ trung tâm pressing; trung vệ đứng phía sau bóng |

**Chỗ lệch có tên cầu thủ** (Pháp – Đan Mạch, theo vai trò chi tiết):
- Eriksen chiếm 5/10 pha lệch lớn nhất, phần lớn khi Đan Mạch mất bóng. Anh ấy đứng xa bóng và lệch khỏi đường bóng–khung thành, tức không áp sát cũng không che. Ví dụ phút 59:52 và 72:56.
- Sau đó là Upamecano, Rabiot, Konaté (vào thay Varane), Koundé.
- Trong 10% chỗ lệch lớn nhất, phương án "chuẩn" có EPV cao hơn ở 48% số trường hợp.

### 3. MARL với phần thưởng là mục tiêu học được — `h2predict/marl.py`
- **Trò chơi:** ở mỗi khoảnh khắc, mọi cầu thủ của cả hai đội cùng lúc chọn một trong các hướng cùng quãng đường. Phần thưởng của mỗi người là mục tiêu của nhóm họ, **tính trên cấu hình chung** (`h2predict/game.py`): khoảng cách tới đồng đội và đối thủ, đường việt vị, và việc **có trống để nhận chuyền không** (mô hình chuyền của Hướng 3, tính theo chỗ hậu vệ thật sự đứng, tức là đường chuyền) đều phụ thuộc vào việc người khác đi đâu.
- **Cách học:** mỗi nhóm có một chính sách softmax, khởi đầu đúng bằng chính sách bắt chước (IRL). Các tác nhân học độc lập bằng policy gradient, dùng baseline phản thực tính chính xác trên mọi lựa chọn của mình, cộng phạt KL (0,3) để không lệch xa hành vi thật. Học từ 40.234 khoảnh khắc, kiểm tra trên 4.157 khoảnh khắc chưa học.

| | Phần thưởng (mục tiêu học được, cấu hình chung) | Xác suất chọn đúng đường chạy thật | EPV của đội hình (kiểm tra độc lập) |
|---|---|---|---|
| Đường chạy thật | +0,508 | (1) | +0,0205 |
| Bắt chước (IRL) | +0,385 | 0,124 | +0,0253 |
| **MARL** | **+0,630** | **0,166** | +0,0248 |

Độ nhạy theo trọng số KL (bản chưa tính đường chuyền động): 0,1 cho phần thưởng +0,688 và xác suất 0,180; 1,0 cho +0,490 và 0,142. EPV gần như không đổi.

**Phối hợp thay đổi gì** (lựa chọn của MARL trừ đường chạy thật, trung bình theo nhóm):
- Gần như mọi nhóm đều **áp sát bóng hơn 4–6 m** và **đứng gần đường bóng–khung thành hơn 4–6 m**.
- Mọi nhóm đều **về gần vị trí quen hơn**.
- Tiền đạo và tiền vệ tấn công khi mất bóng **lùi về phía sau bóng khoảng 3 m**.
- Thông điệp chung: **đội hình gọn hơn và kỷ luật hơn**, không phải "lao lên".

**Đọc kết quả cho đúng:**
- MARL vượt hành vi thật theo chính mục tiêu của cầu thủ, và vẫn gần hành vi thật hơn cả bản bắt chước.
- EPV của đội hình cao hơn đội hình thật (+0,025 so với +0,021). Nhưng bản bắt chước cũng đạt mức đó, nên phần tăng đến từ **việc làm theo mục tiêu học được**, chưa chứng minh được là đến từ **phối hợp**.
- Mục tiêu học được chỉ phạt nhẹ việc đứng sát đồng đội, nên khi mọi người cùng tối ưu thì đội **co cụm**: 23,8% cầu thủ kết thúc đường chạy sát một đồng đội, gấp đôi thực tế. Xem mục 4.
- Đây là trò chơi **một bước** (8 s) với mục tiêu **mô tả** cầu thủ World Cup, chưa phải mô phỏng cả pha bóng.

### 4. Ràng buộc giãn cách — `h2predict/marl.py`
- **"Quá gần" lấy từ dữ liệu thật:** khi mọi người ở đúng chỗ họ thật sự chạy tới (các trận train), 10% cầu thủ kết thúc cách đồng đội gần nhất dưới **3,3 m** (trung vị 8,7 m). Ngưỡng 3,3 m và tỉ lệ 10% đều lấy từ đây, không tự đặt.
- **Cách làm:** RL có ràng buộc (Lagrangian). Phần thưởng trừ λ × (1 nếu kết thúc dưới 3,3 m cạnh đồng đội). λ tự tăng khi tỉ lệ co cụm vượt 10% và giảm khi dưới (dual ascent). Hội tụ: tỉ lệ 10,8%, λ = 4,82.

| Kiểm tra trên 4.157 khoảnh khắc chưa học | Phần thưởng | P(đường chạy thật) | Co cụm (< 3,3 m) | Khoảng cách TB tới đồng đội gần nhất | EPV đội hình |
|---|---|---|---|---|---|
| Đường chạy thật | +0,508 | (1) | 10,9% | 10,0 m | +0,0205 |
| Bắt chước (IRL) | +0,385 | 0,124 | 23,9% | 8,8 m | +0,0253 |
| MARL | +0,630 | 0,166 | 23,8% | 8,9 m | +0,0248 |
| **MARL + giãn cách** | **+0,569** | 0,159 | **11,4%** | **10,8 m** | **+0,0250** |

Phần thưởng tính bằng cách lấy mẫu nước đi của mọi người. Ba cột sau dùng lựa chọn có xác suất cao nhất của mỗi chính sách.

- **Giữ giãn cách như thật chỉ tốn ít:** phần thưởng giảm 0,061 nhưng vẫn hơn đường chạy thật 0,061, EPV giữ nguyên. Vậy lời khuyên "gọn lại" của mục 3 phần lớn là do co cụm, không phải điều cần học.
- **Bản bắt chước cũng co cụm (23,9%).** Lý do: mỗi người chọn phương án tốt nhất của mình mà không thấy người khác, nên nhiều người cùng dồn về một chỗ. Đây là lý do cần phối hợp có ràng buộc, chứ không chỉ bắt chước.
- **Thay đổi so với đường chạy thật** (trung bình theo nhóm):
  - *Có bóng:* hậu vệ biên, tiền vệ phòng ngự, cầu thủ chạy cánh và tiền đạo **giãn rộng**: xa đồng đội gần nhất hơn 3,6–5,6 m, xa đường bóng–khung thành hơn 3,4–5,4 m. Tiền vệ phòng ngự, cánh và tiền đạo còn **dâng trước bóng** thêm 4–6 m. Tiền vệ trung tâm thì **áp vào bóng 4,5 m** để hỗ trợ. Tức là chiều rộng và chiều sâu khi có bóng.
  - *Mất bóng:* tiền đạo **ép bóng** (gần bóng hơn 4,8 m, gần đường bóng–khung thành hơn 4,3 m), tiền vệ trung tâm cũng vậy (3,5 m), trung vệ **lùi về sau bóng** thêm 3,5 m. Cánh và tiền vệ tấn công về gần vị trí quen hơn 4–5 m.
  - *Chỗ cần HLV xem:* khi mất bóng, hậu vệ biên và tiền vệ phòng ngự lại đứng **xa bóng hơn 4–6 m**. Có thể ràng buộc đang dồn phần giãn cách vào những người có nhiều khoảng trống nhất, chứ không phải điều đội nên làm.

### 5. MARL nhiều bước — `h2predict/marl_multi.py`
**Cách làm**
- **Môi trường:** model dự báo đóng vai "phần còn lại của trận đấu" (bản có điều kiện `h2_cond.pt`, dùng như model thường). Mỗi episode bắt đầu từ một khoảnh khắc thật, với 64 s lịch sử thật. Cứ mỗi 4 s, model dự báo 4 s tới cho mọi người. Mỗi cầu thủ trên sân (trừ thủ môn) chọn: đi theo dự báo, hoặc chạy cùng quãng đường theo 1 trong 12 hướng. Bóng đi theo người giữ bóng mà model dự báo. Khung hình mới được nối vào lịch sử, nên ở bước sau model **phản ứng** với những gì cầu thủ vừa làm. 3 bước = 12 s.
- **Phần thưởng mỗi bước:** mục tiêu học được (giai đoạn × vai trò chi tiết) trên cấu hình chung, cùng ràng buộc giãn cách của mục 4 (λ hội tụ ở 2,26). Feature "EPV của đội" không nằm trong phần thưởng, vì trọng số của nó nhỏ: trung bình 0,03, so với 0,10–0,21 của các feature chính. Nhờ vậy EPV sau 12 s là phép kiểm tra hoàn toàn độc lập.
- **Cách học:** phần thưởng của bước hiện tại tính chính xác trên mọi lựa chọn của mình. Hậu quả ở các bước sau ước lượng bằng REINFORCE (baseline là trung bình của batch), chiết khấu 0,9. Phạt KL 0,3 về chính sách bắt chước. Học trên 20.962 episode (58 trận, cứ 16 s một episode), kiểm tra trên 2.160 episode của 6 trận chưa học. Như IRL, bỏ 2,7% khoảnh khắc mà 64 s lịch sử không thấy bóng.
- **Mốc so sánh công bằng: "đường chạy thật, phát lại".** Cầu thủ đi đúng chỗ họ thật sự đi sau mỗi 4 s, model chơi phần còn lại. 12 s ghi hình thật chỉ để tham khảo, vì đó là một thế giới khác: người khác và bóng ở vị trí thật, kể cả các đường chuyền dài. Phần thưởng ở đó thấp hơn nhiều (+0,286), vì vị trí dự báo là trung bình của 6 kịch bản nên "gọn" hơn thực tế.
- **EPV** đo từ phía đội cầm bóng lúc đầu (nếu bóng đã đổi chủ thì đổi dấu). Cao hơn nghĩa là tốt cho đội tấn công, xấu cho đội phòng ngự.

**Kết quả 1: cả hai đội cùng dùng một chính sách** (2.160 episode, 6 trận chưa học)

| | Phần thưởng mỗi bước | Bước cuối | Co cụm | EPV sau 12 s |
|---|---|---|---|---|
| 12 s ghi hình thật (chỉ tham khảo) | +0,286 | +0,283 | 12,7% | +0,0205 |
| Đường chạy thật, phát lại | +0,512 | +0,524 | 12,8% | +0,0205 |
| Đi theo dự báo | +0,557 | +0,587 | 13,6% | +0,0215 |
| Bắt chước (IRL) | +0,483 | +0,520 | 14,4% | +0,0190 |
| MARL một bước, dùng ở mọi bước | +0,581 | +0,632 | 10,6% | +0,0205 |
| **MARL nhiều bước** | **+0,640** | **+0,710** | **10,3%** | +0,0219 |

So sánh ghép cặp trên cùng episode (khoảng tin cậy 95%, bootstrap theo episode):
- MARL nhiều bước − MARL một bước: phần thưởng **+0,060** [+0,057; +0,062], EPV +0,0014 [+0,0010; +0,0018].
- MARL nhiều bước − đường chạy thật phát lại: phần thưởng **+0,129** [+0,123; +0,134], EPV +0,0014 [−0,0003; +0,0033], tức là không khác.

**Kết quả 2: chỉ một đội dùng MARL nhiều bước, đội kia đi đúng đường chạy thật** (so với khi cả hai đội cùng phát lại đường chạy thật)

| | Phần thưởng của đội dùng MARL | Phần thưởng của đội kia | EPV của đội cầm bóng lúc đầu |
|---|---|---|---|
| Chỉ đội tấn công | +0,050 [+0,044; +0,055] | −0,047 [−0,051; −0,044] | **−0,0035** [−0,0048; −0,0022] |
| Chỉ đội phòng ngự | +0,076 [+0,070; +0,082] | −0,062 [−0,067; −0,057] | **+0,0049** [+0,0041; +0,0058] |

**Đọc kết quả**
1. **Lên kế hoạch 12 s có ích, theo mục tiêu học được.** MARL nhiều bước hơn chính sách một bước +0,060, cách biệt lớn dần về cuối (bước cuối +0,710 so với +0,632), và vẫn giữ giãn cách như thật (10,3%).
2. **Khoảng một nửa mức tăng đến từ việc đối thủ cũng chơi theo cùng chính sách.** Khi cả hai đội cùng dùng MARL, mỗi đội tăng khoảng +0,13. Khi đối thủ là cầu thủ thật, đội dùng MARL chỉ tăng +0,05 (tấn công) hoặc +0,08 (phòng ngự), và đối thủ mất gần đúng chừng đó. Nghĩa là trong tự đấu, hai đội "chiều" nhau.
3. **Theo EPV, MARL kém cầu thủ thật ở cả hai đầu sân.** Đội tấn công dùng MARL tạo ít nguy hiểm hơn đường chạy thật: −0,0035, khoảng 17% mức EPV +0,0205. Đội phòng ngự dùng MARL để đối thủ nguy hiểm hơn: +0,0049, khoảng 24%. Khi cả hai cùng dùng thì hai hiệu ứng triệt tiêu, nên bảng 1 trông như "không khác".
4. **Thay đổi so với nước đi dự báo rất nhỏ** (trung bình dưới 1 m mỗi nhóm) và cùng hướng với mục 4. Có bóng: hậu vệ biên, tiền vệ phòng ngự và cánh giãn rộng hơn; tiền vệ trung tâm áp vào bóng. Mất bóng: tiền đạo và tiền vệ trung tâm ép bóng hơn.

**Kết luận của Hướng 1.** Mục tiêu học được **mô tả** tốt điều cầu thủ thường làm: đường chạy thật thắng 66–67% phương án. Nhưng tối ưu nó mạnh hơn, kể cả có phối hợp và lên kế hoạch, làm lối chơi **kém đi** theo một thước đo độc lập (EPV), ở cả tấn công lẫn phòng ngự. Cầu thủ World Cup có những lựa chọn theo tình huống mà mục tiêu tuyến tính 11 feature chưa nắm được, và những lựa chọn "lệch chuẩn" đó có giá trị. Muốn có lời khuyên "tốt hơn", phần thưởng phải chứa **kết quả** (EPV, cú sút, giá trị đường chuyền). Mục tiêu học được chỉ nên giữ vai trò ràng buộc để hành vi vẫn giống người.

### Về chân thuận
- **Dữ liệu chưa có.** `players.csv` và roster của PFF chỉ có chiều cao, số áo và vị trí chi tiết (CF, RW, RB, CM…), không có chân thuận. Muốn dùng thì cần nguồn ngoài, và phải xem điều khoản sử dụng của nguồn đó.
- **Không làm mất khả năng tổng quát hóa**, vì đó là một đặc điểm chung (trái / phải / cả hai), không phải danh tính từng cầu thủ. Điều kiện là lúc train phải thỉnh thoảng giấu nó đi ("không rõ"), để model vẫn dùng được khi thiếu thông tin. Với video giải nghiệp dư, HLV có thể tự nhập.
- **Độ chính xác.** Với vị trí 4–8 s nói chung, dự kiến chỉ tăng ít. Nó có thể giúp ở vài tình huống cụ thể (tiền vệ cánh bó vào trong bằng chân thuận, hậu vệ biên chồng biên hay đi trong, nhận bóng bên nào), nên hữu ích cho inverse RL và mô hình chọn đường chuyền hơn là cho model dự báo. Phải đo bằng ablation (có / không) trên trận chưa học.
- **Đã làm trước thứ có sẵn:** vị trí chi tiết lấy từ roster (`h2predict/roles.py`). Bản đồ số áo → ô được dựng lại bằng cách khớp vị trí lúc giao bóng (sai số trung vị 0,14 m), cộng 563 lần thay người lấy từ dữ liệu sự kiện. Nhờ vậy mỗi ô ở mỗi thời điểm đều biết là ai, đá vị trí gì.

## Giới hạn

- **Phản ứng là tương quan, không phải nhân quả.** Model học "khi cầu thủ này ở đây thì những người khác thường ở đâu" từ dữ liệu thật. Ngoài ra phản ứng đang yếu (1,4 m cho 10 m), nên các phương án có thể trông tốt hơn thực tế, vì đối thủ trong model bám theo ít hơn ngoài đời.
- **Giá trị chỉ là đại diện gần đúng.** Nó chưa tính đường chuyền cụ thể, nhiệm vụ phòng ngự, thể lực hay chỉ đạo của HLV. Cầu thủ chạy chỗ còn để kéo người, giữ cân bằng hay hỗ trợ, và mô hình giá trị chưa thấy những điều đó.
- **Đường chạy là đường thẳng, độ phân giải 4 s**, và bóng không được lên kế hoạch lại.
- Việt vị chỉ ước lượng bằng hàng thủ dự đoán ở +8 s.
- **MARL dùng model dự báo làm môi trường.** Phản ứng của model là tương quan và còn yếu. Vị trí dự báo là trung bình của 6 kịch bản, nên gọn hơn thực tế. Bóng đi theo người giữ, chưa có đường chuyền được lên kế hoạch, và mỗi bước dài 4 s.
- **Mục tiêu học được là mô tả, không phải chuẩn mực:** nó cho biết cầu thủ World Cup thường làm gì, không cho biết điều gì tạo ra bàn thắng (mục 5).
- **Chưa có HLV kiểm chứng.** Mọi gợi ý chỉ là giả thuyết để HLV xem lại.

## Bước tiếp theo

1. **Phần thưởng có kết quả:** mục tiêu học được + β × thay đổi EPV (hoặc giá trị đường chuyền của Hướng 3). Chọn β sao cho hành vi vẫn gần cầu thủ thật, rồi kiểm tra lại bằng phép tách công/thủ của mục 5.
2. **Học với đối thủ là cầu thủ thật** (phát lại), không chỉ tự đấu, để hai đội không "chiều" nhau.
3. **Cho bóng được chuyền trong mô phỏng:** người giữ bóng chọn người nhận bằng mô hình chọn đường chuyền của Hướng 3.
4. **Kiểm chứng bằng kết quả trận:** những pha mà đường chạy thật "kém" theo mô hình có thật sự dẫn tới ít cú sút hơn không.
5. Làm phản ứng mạnh và đúng hơn: bước thời gian 1 s; train với nhiều mẫu đường chạy bất thường; đo mức "bám người" so với dữ liệu thật.
6. Đưa top bước ngoặt cho HLV chấm: giữ hay bỏ, vì sao. Đo tỉ lệ gợi ý hữu ích.

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
.venv/bin/python -m h2predict.epv
```
```bash
.venv/bin/python -m h2predict.options
```
```bash
.venv/bin/python -m h2predict.irl --train-matches 24
```
```bash
.venv/bin/python -m h2predict.irl --cache
```
```bash
.venv/bin/python -m h2predict.roles
```
```bash
.venv/bin/python -m h2predict.styles
```
```bash
.venv/bin/python -m h2predict.marl
```
```bash
.venv/bin/python -m h2predict.marl_multi
```
```bash
.venv/bin/python -m h2predict.runs --match pff:3834 --value pass
```
```bash
.venv/bin/python -m h2predict.runs --match pff:3834 --value epv
```
```bash
.venv/bin/python -m h2predict.runs --match pff:3834 --value opportunity
```

Sau đó mở `http://localhost:8765/runs.html` (server: `.venv/bin/python -m h2predict.serve_ui --dir ui`).
