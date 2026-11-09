# Thiết kế Feature Scaling

## 1. Purpose - Mục đích

- Nhóm chuẩn hóa các thuộc tính số để hạn chế ảnh hưởng của chênh lệch đơn vị tới khoảng cách.
- Nhóm chọn StandardScaler làm phương án chính theo mục 6.3 của [thiết kế tổng thể](design.md).
- Nhóm cân nhắc MinMaxScaler như một khảo sát bổ sung nếu đủ thời gian.
- Scaling là bước chuẩn bị cho phân cụm, không phải thuật toán tạo nhãn cụm.

## 2. Status - Trạng thái thiết kế

- Trạng thái: dự kiến sử dụng, chờ thống nhất dữ liệu và đặc trưng.
- StandardScaler thuộc luồng baseline; MinMaxScaler là tùy chọn.
- Nhóm chỉ chuyển sang thực nghiệm khi dữ liệu đáp ứng các điều kiện đầu vào.

## 3. Data used - Dữ liệu sử dụng

- Nguồn nội bộ dự kiến: `data/raw/marketing_campaign.csv`.
- Theo đề bài, không gian phân cụm dự kiến gồm tuổi, thu nhập hằng năm và điểm chi tiêu.
- Dữ liệu raw có `Year_Birth`, `Income` và các khoản chi tiêu nhưng chưa có trực tiếp `Age` và `Spending Score`; `Gender` cũng chưa có.
- Nhóm cần thống nhất dữ liệu phù hợp đề hoặc điều chỉnh đặc trưng được chấp thuận trước khi scaling.
- Nhóm không mặc định tổng chi tiêu tương đương điểm chi tiêu; năm tham chiếu tính tuổi cũng cần căn cứ.
- Nhóm loại mã định danh khỏi phép biến đổi, giữ mã riêng để đối chiếu.
- Nhóm giữ bản dữ liệu theo đơn vị gốc để phục vụ diễn giải.

## 4. Input - Đầu vào

| Thành phần | Điều kiện |
|---|---|
| Bảng đặc trưng số | n khách hàng và p đặc trưng được thống nhất; dự kiến p = 3 theo đề |
| Chất lượng dữ liệu | Không còn giá trị thiếu, vô hạn hoặc sai kiểu chưa được xử lý |
| Thứ tự dữ liệu | Tên cột và thứ tự khách hàng được ghi nhận, giữ nhất quán |
| Phương pháp scaling | StandardScaler chính; MinMaxScaler khi khảo sát bổ sung |

- Nhóm xem xét ngoại lai ở bước EDA; scaling không thay thế việc kiểm tra dữ liệu.
- Nhóm xem xét thuộc tính có phương sai bằng không vì thuộc tính đó không giúp phân biệt khách hàng.

## 5. Output - Đầu ra dự kiến

- Ma trận chuẩn hóa có cùng số hàng và thứ tự bản ghi với bảng đầu vào.
- Danh sách và thứ tự đặc trưng dùng cho bước phân cụm.
- Thông tin phép biến đổi để áp dụng nhất quán và diễn giải ngược về đơn vị gốc.
- Với StandardScaler: trung bình và độ lệch chuẩn của từng thuộc tính.
- Với MinMaxScaler: giá trị nhỏ nhất, lớn nhất và khoảng chuẩn hóa của từng thuộc tính.
- Nhóm chuyển cùng một ma trận chuẩn hóa cho các thuật toán cần đối chiếu.

## 6. Cách thực hiện dự kiến

1. Nhóm xác minh danh sách đặc trưng, đơn vị và điều kiện dữ liệu.
2. Nhóm hoàn tất xử lý giá trị thiếu và các bản ghi không hợp lệ.
3. Nhóm xác định tham số scaling từ tập dữ liệu dùng để phân tích.
4. Nhóm biến đổi từng cột số, giữ nguyên liên kết với khách hàng.
5. Nhóm kiểm tra tính hữu hạn, kích thước và phân phối sau biến đổi.
6. Nhóm sử dụng ma trận này cho baseline và phương án đối chiếu.

| Phương pháp | Phép biến đổi cho thuộc tính không hằng | Lựa chọn dự kiến |
|---|---|---|
| StandardScaler | z = (x - trung bình) / độ lệch chuẩn | Đưa về trung bình xấp xỉ 0, phương sai xấp xỉ 1 |
| MinMaxScaler | z = (x - min) / (max - min) | Khoảng [0, 1] trên dữ liệu xác định phép biến đổi |

- Theo thiết kế tổng thể, nhóm phân tích toàn bộ tập dữ liệu cho mục tiêu khám phá.
- Nếu bổ sung holdout sau này, nhóm cần điều chỉnh thiết kế; chỉ xác định tham số scaling trên tập train.

## 7. Evaluation metrics - Tiêu chí kiểm tra

- Nhóm kiểm tra số hàng, thứ tự cột và liên kết khách hàng trước và sau scaling.
- Nhóm kiểm tra không có NaN hoặc giá trị vô hạn trong ma trận đầu ra.
- Nhóm kiểm tra trung bình và phương sai với StandardScaler trên các thuộc tính không hằng.
- Nhóm kiểm tra khoảng [0, 1] với MinMaxScaler trên dữ liệu dùng xác định phép biến đổi.
- Đây là kiểm tra phép chuẩn hóa; Silhouette chỉ dùng sau khi có nhãn phân cụm.
- Khi khảo sát scaler bổ sung, nhóm giữ cố định tập khách hàng, đặc trưng và cấu hình phân cụm để đối chiếu.

## 8. Hạn chế và điều kiện chuyển bước

- StandardScaler và MinMaxScaler đều có thể chịu ảnh hưởng của giá trị cực đoan.
- StandardScaler không làm phân phối dữ liệu tự động trở thành phân phối chuẩn.
- Giá trị mới ngoài miền dùng xác định MinMaxScaler có thể nằm ngoài [0, 1].
- Scaling không khắc phục đặc trưng sai ý nghĩa hoặc giải quyết sự thiếu hụt thuộc tính so với đề.
- Nhóm chỉ chuyển bước khi thống nhất đặc trưng và hoàn thành các kiểm tra đầu vào, đầu ra.

## 9. Liên kết trong bài lab

- Bước trước: EDA và xử lý dữ liệu theo [quy ước chung](README.md).
- Bước sau: [K-Means baseline](kmeans_design.md), sau đó [Hierarchical tùy chọn](hierarchical_design.md).
- Phần thực hành tương ứng: `notebooks/02_preprocessing_and_scaling.ipynb`.
