# Formation Lens / h2predict — kiểm tra 11/10/2026

Đã đọc code, log, checkpoint, kết quả rollout và dữ liệu UI; chạy thêm inference trên 384 cửa sổ validation trải đều qua 6 trận, ở cả chế độ full tracking và camera giả lập. Chưa sửa code sản phẩm hoặc train lại.

## Kết luận

Forecaster mới đã train xong. Có cơ sở để làm demo phân tích chiến thuật và dự đoán ngắn hạn, nhưng chưa có bằng chứng đủ mạnh cho lời hứa dự đoán chính xác toàn bộ hiệp 2. Phần video và phần dự đoán hiện chưa nối thành một luồng hoàn chỉnh trong UI.

## Trạng thái

- Có 64 trận PFF đã chuyển đổi. Run mới dùng 58 trận PFF + 1 trận Metrica để train, 6 trận PFF để validation.
- `checkpoints/train_pff_cam.log`: 30/30 epoch, 11.250/11.250 iteration, kết thúc `done in 1h05m21s`.
- `checkpoints/h2_pff_cam.pt`: checkpoint tốt nhất được lưu ở epoch 28; epoch 30 vẫn chạy xong.
- Run Metrica cũ `train_step4s.log` cũng hoàn thành 30 epoch. Run `train_pff_cam_rosterbug.log` dừng giữa epoch 7; có checkpoint cũ ghi tên `rosterbug_ep6`, không phải checkpoint mới nhất.
- Khi kiểm tra process không có job `h2predict.train` đang chạy; server UI đang chạy ở localhost:8765.
- Các model YOLO player/pitch/ball có sẵn dưới `data/vision`; code tải các trọng số này để inference. Không thấy run tự train các detector đó trong repo.

## Bằng chứng chất lượng

### Kết quả cả hiệp 2 đã lưu

Nguồn: `checkpoints/eval_half2.json`, 6 trận validation × 2 đội. MAE càng thấp càng tốt. Baseline giữ nguyên thống kê hiệp 1; model mới dùng khởi đầu kickoff.

| Chỉ số | Giữ thống kê hiệp 1 | Model kickoff |
|---|---:|---:|
| Độ cao đội hình (m) | 2,430 | 4,802 |
| Độ cao hàng thủ (m) | 2,596 | 4,817 |
| Độ rộng đội hình (m) | 3,614 | 5,339 |
| Tỷ lệ giữ bóng (điểm phần trăm) | 9,269 | 5,020 |
| Số cú sút | 3,583 | 2,704 |
| Khoảng cách từng cầu thủ (m) | 20,292 | 25,211 |

Model có tín hiệu ở possession và shots, nhưng thua baseline ở hình học đội hình. Baseline có điều chỉnh theo tỷ số còn đạt MAE possession 5,252 điểm phần trăm, rất gần model 5,020. Đây là cùng tập validation dùng chọn checkpoint, không phải test set độc lập. Chưa có khoảng tin cậy hay kiểm định để khẳng định lợi thế tổng quát.

### Kiểm tra ngắn hạn mới chạy

Checkpoint `h2_pff_cam.pt`, CPU, `max_val_windows=384`; không đổi trọng số. Chọn phương án có xác suất cao nhất của model (top-1), không dùng tương lai thật để chọn phương án. Khoảng cách tính trên cầu thủ và bóng hợp lệ.

| Chế độ / phương pháp | Sai số tại +4s | ADE toàn 32s | Sai số tại +32s |
|---|---:|---:|---:|
| Full tracking: model top-1 | 3,307 m | 13,248 m | 18,893 m |
| Full tracking: đứng yên | 5,845 m | 15,307 m | 21,263 m |
| Full tracking: vận tốc không đổi | 4,246 m | 29,211 m | 53,264 m |
| Camera giả lập: model top-1 | 4,161 m | 13,777 m | 19,324 m |
| Camera giả lập: đứng yên | 8,316 m | 16,900 m | 22,382 m |

Baseline camera dùng vị trí nhìn thấy gần nhất trong lịch sử; nếu chưa thấy thì dùng role anchor. Những số trên là spot-check 384 cửa sổ, không thay thế toàn bộ validation 3.000 cửa sổ.

## Nguyên nhân / lỗi đã xác định

1. **Bài toán train và sử dụng lệch nhau.** `config.py:21-24` lấy 16 bước lịch sử, mỗi bước 4s, dự đoán 8 bước tương lai (32s). `rollout.py:111-150` nối nhiều lần dự đoán để sinh cả hiệp, mỗi lần giữ 16s mặc định. Model được train từ lịch sử thật nhưng rollout dùng lại lịch sử tự sinh. Score, possession share và role context còn giữ cố định ở hiệp 1 trong rollout. Đây là cơ chế dễ tích lũy sai lệch; train thêm epoch không tự sửa được cách đặt bài toán này.

2. **Metric tốt nhất trong 6 phương án không phải chất lượng phương án được chọn.** Trong `losses.py:20,39-55`, `minADE`, `FDE` và `ade@...` dùng phương án gần tương lai thật nhất. Checkpoint mới có camera minADE 8,867 m nhưng top-1 ADE 13,339 m trên 3.000 cửa sổ. Cần trình bày rõ hai loại metric và chọn checkpoint theo mục tiêu triển khai.

3. **Possession head gần như chỉ chọn “nobody”.** Spot-check top-1 trả nobody ở 3.070/3.072 bước, trong khi nhãn thật nobody chỉ 2.145/3.072. Khi thật sự có người giữ bóng, full tracking đúng 1/927, camera đúng 0/927. Loss dùng cross-entropy không cân bằng; lớp nobody chiếm khoảng 70%. Đây là một nguyên nhân có căn cứ, chưa làm ablation để quy hết lỗi cho riêng loss. Khi bỏ lớp nobody và chỉ xếp hạng cầu thủ, top-3 đúng khoảng 32%, cho thấy vẫn còn tín hiệu để cải thiện.

4. **Người giữ bóng và tọa độ bóng thiếu ràng buộc nhất quán.** `rollout.py:132-136` lấy tọa độ, possessor và state từ các head rồi sample rời nhau. Trong file `rollout_3834.npz` (16 kịch bản FRA–DEN), 97,05% trong 1.761 bước có người được gán giữ bóng và có đủ tọa độ lại cách bóng hơn 3m; trung vị 13,73m. Đây là kiểm tra riêng file rollout đó, không phải kết luận cho mọi trận.

5. **Camera giả lập chưa kiểm chứng được video thật.** `dataset.py:51-56,85` che vị trí lịch sử nhưng vẫn dùng roster trước che, role anchors, possession share và history state tính từ tracking gốc. Video thật phải tự suy ra các tín hiệu này. Ngoài ra `data.py:164-166` coi mất tọa độ bóng là out-of-play, trong khi trên video có thể chỉ là detector không thấy bóng. Cần tách unknown/occluded khỏi out và đánh giá pipeline video riêng.

6. **Luồng video → forecast chưa hoàn thiện.** `ui/data/0bfacc_0.json` chỉ có 150 frame ở 5Hz (~30s), không có prediction. Forecaster yêu cầu khoảng một phút lịch sử và được train với ít nhất 10 phút context. Các match có prediction trong UI dùng tracking có sẵn và rollout đã export; server UI chỉ phục vụ file tĩnh. Clip hiện có median 20 cầu thủ thấy được mỗi frame, bóng có tọa độ ở 92% frame; các số này đo coverage, không phải độ chính xác tracking.

7. **Lỗi đổi trận trong UI.** `loadMatch()` thay `state.doc` trước khi `setView()` đọc `series()` theo view cũ (`ui/app.js:456-481`). Từ `pred` sang clip không có prediction gây `Cannot read properties of null (reading 'samples')`; từ `half2` sang clip chỉ có half1 gây lỗi đọc `t0`. Đã tái hiện bằng Node VM chạy các hàm thật, stub DOM và phần render; đây không phải kiểm thử browser end-to-end. Cần reset view hợp lệ và dừng playback khi thay doc.

## Ưu tiên cải thiện cho demo

1. Sửa lỗi đổi trận; chọn một match mặc định có dữ liệu tốt; ghi rõ dữ liệu thật, dự đoán và kịch bản. Dùng tiếng Anh cho bản trình bày HackRU.
2. Định vị là trợ lý phân tích chiến thuật: video đồng bộ sân 2D, độ rộng/hàng thủ, thay đổi đội hình trước–sau, chỉ ra một tình huống cụ thể mà người xem hiểu ngay.
3. Demo forecast +4s trước, có nút dừng → dự đoán → hiện kết quả thật, kèm baseline. Dùng tracking đã có để hoàn thiện luồng này; chỉ hứa video end-to-end sau khi đã chạy và kiểm tra cùng một clip đủ dài.
4. Nếu train tiếp: tách “có người giữ bóng không” khỏi “ai giữ bóng”, thử cân bằng lớp và báo held-only accuracy; thêm ràng buộc bóng–người và vị trí–vận tốc; đánh giá top-1 và consistency trước/sau sửa.
5. Nếu vẫn giữ sản phẩm dự báo cả hiệp: dự đoán trực tiếp phân phối thống kê chiến thuật, so với baseline hiệp 1 + score effect; đánh giá trên test set độc lập. Không xem độ đa dạng của các rollout là uncertainty đã được hiệu chuẩn.

Không thể suy ra xác suất đoạt giải từ repo. Chất liệu kỹ thuật đủ để tạo một demo có sức thuyết phục nếu lời hứa phù hợp với những gì đã kiểm chứng. Hiện tại phần mô phỏng toàn hiệp và luồng video còn yếu.

## Tham khảo bên ngoài

- HackRU Fall 2026: https://www.hackru.org/ — website công bố sự kiện 10–11/10, track Maverick phù hợp dự án ngoài Social Good/Education. Không tìm thấy rubric chấm chi tiết trên trang đã đọc.
- TacticAI: https://www.nature.com/articles/s41467-024-45965-x — nghiên cứu tập trung vào tình huống phạt góc, receiver/shot prediction và đề xuất vị trí; là ví dụ hữu ích về thu hẹp bài toán chiến thuật để đánh giá rõ ràng, không phải bằng chứng xác nhận model trong repo.
