# Mentor Q&A — câu trả lời trung thực, kèm bằng chứng

Mỗi mục: **Trả lời** (nói được ngay) → **Bằng chứng** (file/test/lệnh chạy được) → **Còn thiếu** (nói thẳng).
Các số đo lấy từ lần chạy Colab đã nộp (`data/demo_artifacts/`). Các trường mới (FPPI, bản ghi tái lập, hash V1, chống rò rỉ)
**đã có trong code nhưng artifact hiện tại chưa chứa chúng** — cần chạy lại notebook một lần (xem cuối tài liệu).

---

## 1. Toàn vẹn hình học của 5 landmark

**Trả lời.** Em không nội suy theo bounding box. Fisheye được áp dụng bằng một mô hình bán kính phi tuyến, và **cả ảnh lẫn từng điểm `(lx, ly)` dùng chung một ánh xạ**:

- Ảnh: với mỗi pixel đích, lấy mẫu nguồn tại `src = c + (dst − c) · (1 + k1·r² + k2·r⁴)` (r chuẩn hoá theo `f = max(w,h)/2`).
- Điểm: ánh xạ ngược của công thức trên. Giải `r_src = r_dst · (1 + k1·r_dst² + k2·r_dst⁴)` bằng Newton (12 vòng) rồi đặt `dst = c + (src − c) · r_dst / r_src`.
- Bbox: đưa 36 điểm mẫu dọc 4 cạnh qua cùng ánh xạ rồi lấy hình chữ nhật bao.
- Blur (IMU) không dịch chuyển hình học nên landmark giữ nguyên.

**Bằng chứng.**
- `colab/colab_zero_disk_pipeline.py`: `distort_image`, `distort_points`, `box_after_distortion`.
- `tests/test_colab_pipeline.py::test_distortion_points_follow_pixels`: vẽ một điểm sáng, warp ảnh, đo vị trí điểm sáng — khớp ánh xạ điểm trong ≤ 3 px.
- `test_landmarks_follow_the_exact_radial_model_not_bbox_interpolation`: nếu nội suy theo bbox thay vì ánh xạ chính xác thì lệch **4–7 px** ở severity 5 (khuôn mặt rộng 120 px, mép phải ảnh 640×480; severity 3: 3,7–6,2 px).
- Landmark dùng khi fine-tune chỉ lấy từ dự đoán V1 khi V1 khớp GT (IoU ≥ 0.5 và conf ≥ 0.4); còn lại ghi `0 0 0` (không có nhãn) thay vì đoán. Quy ước 20 cột của loader: `yolo_label_rows` + test.
- Bbox bị đẩy ra ngoài > 50% diện tích khung thì loại mẫu (`inside_fraction`, test).

**Còn thiếu.** (a) WIDER FACE không có nhãn landmark, nên nhãn huấn luyện là *pseudo-label* từ V1, không phải nhãn người; nếu V1 lệch thì nhãn lệch. (b) Em chưa đo NME (lỗi landmark chuẩn hoá) của V1 so với V2. (c) Ma trận biến đổi không phải ma trận affine/homography mà là ánh xạ bán kính phi tuyến — công thức ở trên là "ma trận" thực tế.

---

## 2. "Học vẹt nhiễu giả tạo" và tập Easy/Medium/Hard, False Positive

**Trả lời thẳng.** Mối lo này đúng và em **chưa loại bỏ được**: tập blind được hỏng bằng *cùng engine* với tập train, nên V2 có thể học dấu vết của bộ lọc. Gain 5,8 điểm chưa chứng minh được "giỏi hơn thật".

**Em có gì.**
- Ảnh blind **không bao giờ** dùng để sinh dữ liệu hay train (tách theo thứ tự stream + bộ chống rò rỉ dHash loại ảnh train gần trùng ảnh blind).
- Cổng bảo vệ miền gốc: base mAP trên ảnh sạch không được giảm quá 1 điểm. Lần chạy: tăng 2,4 điểm.
- Mẫu "+Δ confidence" trên các case là số **đo thật** từ lần chạy (ví dụ +0,76; +0,66; +0,64). Con số `+0.63` và mã `eval_ego4d_112` trong bản thiết kế ban đầu chỉ là giá trị minh hoạ, không còn trong artifact.

**Còn thiếu (cần làm trước khi nói "chống shortcut").**
1. **Đối chứng**: fine-tune chỉ bằng ảnh sạch cùng số epoch. Nếu V2 vẫn hơn rõ ở corrupted mAP thì mới kết luận được do học hỏng chứ không phải do học thêm trên WIDER.
2. **Hỏng khác họ**: đánh giá bằng bộ hỏng không dùng khi train (ví dụ `imagecorruptions`, tham số k1/k2 ngoài khoảng train) và **ảnh egocentric thật** (Ego4D).
3. **Easy/Medium/Hard chính thức của WIDER FACE**: chưa làm. Cần danh sách `wider_{easy,medium,hard}_val.mat` của bộ eval chính thức; HF dataset hiện dùng không có sẵn. Hiện em chỉ báo mAP trên toàn tập sạch (không tách).
4. **False Positive**: em chưa liệt kê riêng FP hay phân loại (bóng cây, đầu gối…). Em đã thêm **Recall tại FPPI cố định** (mục 3) để thấy FP tăng hay không, nhưng chưa có số.

---

## 3. PII: Recall quan trọng hơn Precision; ghost box của tracker

**Trả lời.** Đồng ý: với PII, bỏ sót mặt là lỗi nặng, thêm một vùng mờ thừa thì rẻ. mAP@0.5 không phản ánh điểm vận hành, nên em đã thêm **Recall tại FPPI cố định (0,1 và 1,0 lỗi dương/ảnh)** cho cả ảnh sạch và ảnh hỏng, V1 so với V2 (`recall_at_fppi`, ghi vào `eval_report.json` → `recall_at_fppi`).

**Tracker — cơ chế cắt đuôi.** `VelocityTracker`:
- giữ box bằng ngoại suy vận tốc tối đa `max_age = 4` khung rồi **kết thúc track**;
- **vận tốc giảm 30% mỗi khung bị mất** (`damping = 0.7`) để box không trượt theo quán tính;
- nếu có kích thước khung, track **kết thúc ngay khi tâm box ra ngoài khung** (người đã đi khuất);
- chỉ nhận lại detection nếu IoU với box dự đoán ≥ 0.2, tránh dính nhầm sang người khác.
- Test: `test_tracker_does_not_chase_a_ghost` (track dừng, chuyển động giảm dần).

**Còn thiếu.** (a) Quay đầu 90° làm mặt biến mất nhưng người còn đó: tracker vẫn coi là mất, nên nó giữ box tối đa 4 khung rồi thôi — chưa có cơ chế phân biệt "quay đi" với "rời cảnh". Về mặt PII, giữ box thừa vài khung là chấp nhận được (FP rẻ), nhưng **em chưa đo tỉ lệ ghost box** vì chưa có chuỗi video có nhãn. (b) Tracker chưa được đánh giá trên chuỗi Ego4D thật. (c) Số FPPI chưa có (cần chạy lại).

---

## 4. Rò rỉ dữ liệu và khả năng tái lập

**Rò rỉ.** WIDER FACE là ảnh rời, không phải video, nên mức tách là *ảnh*. Em đã thêm bộ chống rò rỉ bằng **dHash 64-bit**: ảnh train nào gần trùng (khoảng cách Hamming ≤ 6) với một ảnh blind sẽ bị loại; số ảnh bị loại ghi vào `provenance.leaks_removed`. Với Ego4D, nếu sau này dùng để train thì **bắt buộc tách theo video/subject**; hiện pipeline chỉ dùng Ego4D để *đánh giá* (không train), nên không có rò rỉ train↔test ở đó.

**Hạn chế.** Tách theo ảnh chưa chặn được "cùng sự kiện, khác ảnh" của WIDER (ví dụ nhiều ảnh cùng một buổi diễu hành). Cách chặt hơn là tách theo `event` (tên thư mục WIDER), việc này cần giữ đường dẫn ảnh qua stream — chưa làm.

**Tái lập từng ảnh.** Mỗi mẫu tổng hợp (và mỗi hard negative) nay có trường `reproduce`:
`seed`, `key`, `tag`, `severity`, `source.index` (vị trí ảnh nguồn trong stream WIDER val), `v1_sha256`, `pipeline_version`, cộng `parameters` chính xác (`k1,k2,imu_hz,pan_deg,kernel_px`).
`provenance` có thêm phiên bản python/numpy/opencv/torch. Hàm `reproduce_sample(img, boxes, lmks, record)` dựng lại **đúng từng pixel** từ ảnh nguồn + bản ghi (test `test_any_synthesized_sample_is_reproducible_from_its_record`), nên **không cần lưu ảnh tổng hợp** trên Data Lake.

**Còn thiếu.** Artifact đã nộp được tạo *trước* khi có các trường này — cần chạy lại để `audit_trail.json` mang chúng. Hash của V2 đã có trong `manifest.json`; hash của dữ liệu nguồn (ảnh WIDER) thì chưa (chỉ có chỉ số trong stream).

---

## 5. Hiệu năng và chi phí

**Đo thật** (CPU máy phát triển, khung **1080p**, trung vị 5 lần; `python scripts/benchmark_attacks.py`):

| Bước | ms/khung |
|---|---|
| IMU blur `filter2D` (kernel 15 / 35 / 55 px) | 42 / 45 / 59 |
| Remap bán kính cả ảnh | 62 |
| Ánh xạ 5 landmark + bbox | 0,3 |
| `synthesize` fisheye / motion blur / sensor / occlusion (severity 5) | 63 / 122 / 97 / 1,3 |
| Trung bình bốn loại | **≈ 71** |

Ngoại suy tuyến tính: 1 triệu khung ≈ **20 core-giờ** trên một lõi CPU (chưa tính đọc/ghi). Đây là con số CPU yếu; kornia/GPU sẽ nhanh hơn nhiều bậc nhưng em chưa đo trên GPU.

**Vì sao sinh ảnh tĩnh thay vì on-the-fly?** Thành thật: với dữ liệu hàng chục TB, **on-the-fly trong DataLoader là thiết kế đúng cho production**. Em ghi ảnh ra ổ Colab vì (1) trainer của repo `yolov7-face` đọc ảnh từ đĩa, (2) cần mẫu cụ thể để người duyệt xem và để ghi audit trail, (3) tập mẫu nhỏ (vài trăm ảnh). Nhờ mỗi mẫu có `reproduce` (seed + tham số), có thể **chuyển sang chỉ lưu bản ghi và dựng lại ảnh khi cần** (mục 4) — băng thông Presigned URL chỉ phải trả cho ảnh *nguồn*, và chi phí đọc/ghi ảnh tổng hợp biến mất.

**Còn thiếu.** Em chưa chạy trên quy mô lớn, chưa đo băng thông thực, chưa làm DataLoader on-the-fly.

---

## Việc cần làm để các câu trả lời "đủ số"

1. **Chạy lại notebook một lần** (`colab_zero_disk_pipeline.ipynb` từ GitHub) để artifact có: `recall_at_fppi`, `reproduce`, `v1_sha256`, `leaks_removed`.
2. Thêm thí nghiệm **đối chứng chỉ-ảnh-sạch** và **nhiều seed / tập blind lớn hơn**, báo cáo trung bình ± độ lệch (mục 2).
3. Khi có khóa AWS: đưa **Ego4D thật** vào (mục 2, 3).
4. Làm **Easy/Medium/Hard** chính thức và tách theo **event** (mục 2, 4).
