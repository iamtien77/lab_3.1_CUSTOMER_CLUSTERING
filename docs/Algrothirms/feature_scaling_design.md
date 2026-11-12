# Thiết kế Feature Scaling

## 1. Purpose - Mục đích

- Nhóm chuẩn hóa các thuộc tính số để hạn chế ảnh hưởng của chênh lệch đơn vị tới khoảng cách Euclidean (tránh cột `Income` có giá trị hàng chục nghìn lấn át `Age` và `Total_Spending`).
- Nhóm chọn `StandardScaler` làm phương án chính theo mục 6.3 của [thiết kế tổng thể](design.md).
- Nhóm đồng thời đối chiếu `MinMaxScaler` và `RobustScaler` để xem mức độ nhạy với phân phối lệch và ngoại lai khi tác động lên phân cụm.
- Scaling là bước chuẩn bị dữ liệu trước phân cụm, không phải bước gán nhãn cụm.

## 2. Status - Trạng thái thiết kế

- Trạng thái: đã thống nhất dùng cho bước tiền xử lý và chuẩn hóa của bài lab.
- `StandardScaler` là nhánh chính đưa vào các thuật toán phân cụm; `MinMaxScaler` và `RobustScaler` dùng cho phần so sánh thực nghiệm.
- Quy trình chỉ thực hiện sau khi dữ liệu đã qua bước lọc lỗi miền, chia `80/20` và điền khuyết theo tập Train.

## 3. Data used - Dữ liệu sử dụng

- Nguồn nội bộ là `data/raw/marketing_campaign.csv`, đã được làm sạch sơ bộ thành tập `Train (80%)` và `Holdout Test (20%)`.
- Không gian đặc trưng gồm 3 cột số đã thống nhất: `Age` (`2014 - Year_Birth`), `Income` và `Total_Spending` (tổng 6 cột `Mnt*`).
- Không có cột `Gender` trong tập dữ liệu và nhóm không tự tạo cột giả để đưa vào chuẩn hóa.
- Cột định danh `ID` bị loại hoàn toàn khỏi ma trận biến đổi và được giữ riêng cùng bảng đơn vị gốc để đối chiếu, diễn giải sau khi có cụm.

## 4. Input - Đầu vào

| Thành phần | Điều kiện |
|---|---|
| Bảng đặc trưng số | Tập Train và Holdout Test cùng có đúng 3 đặc trưng (`Age`, `Income`, `Total_Spending`) |
| Chất lượng dữ liệu | Không còn giá trị thiếu (`NaN`), giá trị vô hạn (`Inf`), hay cột hằng số (phương sai bằng `0`) |
| Thứ tự dữ liệu | Tên cột và thứ tự bản ghi trong từng tập được giữ cố định xuyên suốt |
| Phương pháp scaling | `StandardScaler` (chính), cùng `MinMaxScaler` và `RobustScaler` (đối chiếu) |

- Nhóm giữ các điểm ngoại lai hợp lệ sau bước EDA nên cần bước đối chiếu thêm `RobustScaler` thay vì chỉ dựa vào một phép biến đổi.
- Thuộc tính có phương sai bằng `0` trên Train sẽ bị chặn ngay trước khi fit vì không mang thông tin phân biệt khách hàng.

## 5. Output - Đầu ra dự kiến

- Hai ma trận chuẩn hóa (`Train` và `Holdout Test`) giữ nguyên số dòng, thứ tự cột và tính hữu hạn của dữ liệu.
- Danh sách và đúng thứ tự đặc trưng dùng cho bước phân cụm.
- Bảng tham số đã học từ tập Train để minh bạch công thức và dùng cho bước quy đổi ngược (`inverse_transform`) khi giải thích tâm cụm:
  - Với `StandardScaler`: trung bình (`Mean`) và độ lệch chuẩn (`Std`) trên Train.
  - Với `MinMaxScaler`: giá trị nhỏ nhất (`Min`), lớn nhất (`Max`) và khoảng biến thiên (`Max - Min`) trên Train.
  - Với `RobustScaler`: trung vị (`Median`) và khoảng tứ phân vị (`IQR`) trên Train.
- Nhóm truyền cùng một ma trận chuẩn hóa của `StandardScaler` cho tất cả các thuật toán phân cụm khi so sánh.

## 6. Cách thực hiện dự kiến

1. Kiểm tra đầu vào: bảng không rỗng, đủ 3 cột số hợp lệ, không chứa `ID`, không bật mã hóa giả `Gender`, không còn `NaN`/`Inf` và không có cột phương sai bằng `0`.
2. Fit bộ chuẩn hóa **chỉ trên tập Train (80%)** để học tham số thống kê.
3. Dùng bộ chuẩn hóa vừa học để biến đổi cả **Train (80%)** lẫn **Holdout Test (20%)**, đảm bảo không rò rỉ thông tin từ tập kiểm chứng.
4. Trích xuất bảng tham số đã học và chạy kiểm tra hậu điều kiện trên ma trận đầu ra của cả Train và Test.
5. So sánh 4 trạng thái dữ liệu (`Raw`, `StandardScaler`, `MinMaxScaler`, `RobustScaler`) về mặt thống kê phân phối (`Min`, `Max`, `Mean`, `Std`, `Median`, `IQR`, `Skew`).
6. Thực hiện thêm một đối chiếu có kiểm soát trên K-Means (giữ nguyên dữ liệu Train/Test, dải `K=2..10`, cùng `random_state` và `k-means++`) để xem việc chọn scaler ảnh hưởng ra sao tới chỉ số phân cụm và mẫu Test vượt biên.

| Phương pháp | Phép biến đổi cho thuộc tính không hằng | Mục đích trong bài lab |
|---|---|---|
| `StandardScaler` | `z = (x - Mean_Train) / Std_Train` | Phương án chính: đưa về trung bình xấp xỉ `0`, độ lệch chuẩn xấp xỉ `1` |
| `MinMaxScaler` | `x_norm = (x - Min_Train) / (Max_Train - Min_Train)` | Đối chiếu biên độ cố định `[0, 1]` trên Train và độ nhạy với ngoại lai |
| `RobustScaler` | `x_robust = (x - Median_Train) / IQR_Train` | Đối chiếu cách chuẩn hóa theo `Median/IQR` để giảm tác động của dữ liệu lệch và giá trị lớn |

## 7. Evaluation metrics - Tiêu chí kiểm tra

- Kiểm tra ma trận đầu ra không phát sinh `NaN` hay `Inf`, khớp số dòng và đúng 3 cột đặc trưng.
- Hậu kiểm điều kiện trên tập **Train**:
  - `StandardScaler`: trung bình xấp xỉ `0` và độ lệch chuẩn xấp xỉ `1`.
  - `MinMaxScaler`: toàn bộ giá trị trên Train nằm trong `[0, 1]`.
  - `RobustScaler`: trung vị trên Train xấp xỉ `0`.
- Hậu kiểm trên **Holdout Test**: kiểm tra việc biến đổi dùng đúng tham số của Train và đếm số mẫu Test rơi ngoài `[0, 1]` (với MinMaxScaler) hoặc vượt cận Min/Max của Train.
- Với phần đối chiếu ảnh hưởng lên phân cụm K-Means, nhóm xem đồng thời `Silhouette`, `Davies-Bouldin`, `Calinski-Harabasz`, `Inertia`, độ cân bằng cụm và `Silhouette Test` trên cùng dải `K`.

## 8. Hạn chế và điều kiện chuyển bước

- `StandardScaler` và đặc biệt là `MinMaxScaler` đều bị kéo lệch khi tập dữ liệu có giá trị cực đoan lớn.
- `StandardScaler` không tự biến một phân phối lệch thành phân phối chuẩn.
- Khi dùng `MinMaxScaler`, các điểm mới ở tập Holdout Test có thu nhập hoặc chi tiêu cao/thấp hơn miền Train sẽ nằm ngoài `[0, 1]`.
- Scaling không giải quyết được vấn đề chọn sai đặc trưng, nên phần chọn bộ 3 cột số phải được chốt trước bước này.
- Chỉ chuyển sang bước huấn luyện các mô hình phân cụm khi ma trận Train/Test đã qua bước hậu kiểm hữu hạn và đạt chuẩn không rò rỉ.

## 9. Liên kết trong bài lab

- Bước trước: EDA, làm sạch và tách `80/20` theo [quy ước chung](README.md) và [thiết kế tổng thể](design.md).
- Bước sau: [K-Means baseline](kmeans_design.md), sau đó đối chiếu với [Hierarchical Clustering](hierarchical_design.md) cùng các phương án mở rộng khác.
- Phần thực hành tương ứng: `notebooks/02_preprocessing_and_scaling.ipynb`.
