# Các thiết kế phương pháp cho bài lab

## Purpose - Mục đích

Nhóm dùng bộ tài liệu này để cụ thể hóa [thiết kế tổng thể](design.md), mục 6.3 đến 6.6.
Phạm vi gồm phần chuẩn hóa đặc trưng, baseline K-Means và nhánh phân cụm phân cấp đối chiếu (cùng ghi nhận thêm hai phương án đối chiếu mở rộng là DBSCAN và K-Medoids trong thiết kế chung).

## Danh mục và Status

| Tài liệu | Vai trò | Status của thiết kế |
|---|---|---|
| [Feature Scaling](feature_scaling_design.md) | `StandardScaler` chính; đối chiếu `MinMaxScaler` và `RobustScaler` | Đã thống nhất phương án chuẩn hóa và cách kiểm tra |
| [K-Means](kmeans_design.md) | Baseline phân cụm chính | Đã thống nhất cấu hình baseline và dải khảo sát K |
| [Hierarchical Clustering](hierarchical_design.md) | Đối chiếu theo cấu trúc cây phân cấp | Đã thống nhất phương án linkage và cách đọc Dendrogram |

Các quy ước thiết kế đã được nhóm thống nhất cho bài lab:
- Dùng nguồn dữ liệu nội bộ `data/raw/marketing_campaign.csv` (`sep="\t"`), tạo bộ 3 đặc trưng số gồm `Age = 2014 - Year_Birth`, `Income` và `Total_Spending` (tổng 6 cột chi `Mnt*`); không tự tạo cột `Gender`.
- Dự kiến tách `80% Train / 20% Holdout Test` trước bước điền khuyết bằng trung vị và bước fit chuẩn hóa để tránh rò rỉ dữ liệu.
- Chọn `K-Means` làm baseline chính, đồng thời thiết kế đối chiếu thêm `Hierarchical Clustering`, `DBSCAN` và `K-Medoids` trong giới hạn một bài lab học tập.

## Data used - Điều kiện chung về dữ liệu

- Nhóm sử dụng tập dữ liệu nội bộ `data/raw/marketing_campaign.csv` theo mục 1.2 của thiết kế tổng thể và kiểm tra tên cột, kiểu dữ liệu, đơn vị trước khi xử lý.
- Do tập dữ liệu này lưu `Year_Birth`, `Income` cùng 6 khoản chi tiêu `Mnt*` thay vì có sẵn `Age`, `Gender` và `Spending Score`, nhóm thống nhất chọn năm `2014` làm mốc tính tuổi `Age = 2014 - Year_Birth` (dựa trên khoảng năm gia nhập `Dt_Customer` 2012–2014) và lấy `Total_Spending` đại diện cho mức chi tiêu.
- Nhóm giữ các cột phân loại sẵn có như `Education` và `Marital_Status` cùng 6 nhóm chi tiêu thành phần để phục vụ mô tả cụm thay vì tự tạo dữ liệu `Gender`.
- Trước khi chia tập `80/20`, nhóm lọc bỏ các dòng trùng `ID` hoặc sai miền giá trị rõ ràng (`Year_Birth < 1920` hoặc `> 2014`, `Income < 0` hoặc `> 200,000`, chi tiêu âm); phần trung vị điền khuyết `Income` và tham số chuẩn hóa chỉ được tính trên tập Train (80%).
- Các Input và Output bên dưới được giữ thống nhất cho cả baseline lẫn các phương án đối chiếu.

## Input và Output chung giữa các bước

| Bước | Input dự kiến | Output chuyển sang bước tiếp theo |
|---|---|---|
| Chuẩn bị | Bảng dữ liệu thô đã kiểm tra schema | Bảng Train (80%) và Test (20%) đã lọc lỗi, điền khuyết theo Train và giữ mã đối chiếu |
| Scaling | Bảng sạch của Train và Test theo thứ tự cột cố định | Ma trận Train/Test đã chuẩn hóa và tham số phép biến đổi học từ Train |
| Phân cụm | Ma trận chuẩn hóa của Train (kèm Test khi kiểm tra độ ổn định phân bố) | Nhãn cụm theo thứ tự bản ghi, tâm/liên kết hoặc điểm nhiễu tùy thuật toán |
| Đánh giá | Ma trận chuẩn hóa, nhãn cụm và bảng gốc theo đơn vị thực tế | Các chỉ số nội tại, biểu đồ, bảng hồ sơ cụm và mô tả phân khúc |

- Nhóm giữ nguyên thứ tự bản ghi hoặc mã đối chiếu `ID` xuyên suốt các bước và không đưa `ID` vào phép tính khoảng cách.
- Nhóm giữ cùng tập mẫu Train, cùng 3 đặc trưng số và cùng bộ chuẩn hóa `StandardScaler` khi so sánh các thuật toán.
- Khi khảo sát đổi scaler (`MinMaxScaler`, `RobustScaler`), nhóm giữ cố định tập mẫu, đặc trưng và cấu hình phân cụm để tách riêng ảnh hưởng của tiền xử lý.

## Evaluation metrics - Quy ước đánh giá

- Silhouette là chỉ số chính, chỉ tính khi số cụm hợp lệ nằm từ `2` đến `n_valid - 1` (trên tập điểm không phải nhiễu).
- Nếu một cấu hình không đủ điều kiện tính (ví dụ chỉ ra 1 cụm hoặc toàn điểm nhiễu), nhóm ghi nhận là không tính được kèm lý do, tuyệt đối không điền giá trị `0` giả tạo.
- Với phương pháp mật độ như DBSCAN, nhóm loại các điểm nhiễu (nhãn `-1`) khỏi phép tính chỉ số phân tách và báo cáo riêng số lượng, tỷ lệ nhiễu.
- Nhóm kết hợp thêm các chỉ số phụ trợ (`Davies-Bouldin`, `Calinski-Harabasz`, `Dunn Index`, tỷ lệ điểm có Silhouette âm, độ cân bằng kích thước cụm) cùng Inertia/Elbow (cho K-Means), biểu đồ Scatter và thống kê theo đơn vị gốc.
- Nhóm không đặt trước số cụm cố định hay tên phân khúc có sẵn mà để dữ liệu dẫn dắt.

## Tổ chức thực hành

- Nhóm thực hiện khám phá, kiểm định dữ liệu thô, làm sạch và chia `80/20` trong `notebooks/01_eda.ipynb`.
- Nhóm thực hiện chuẩn hóa, kiểm chứng điều kiện sau scaling và đối chiếu các scaler trong `notebooks/02_preprocessing_and_scaling.ipynb`.
- Nhóm thử nghiệm K-Means baseline và các thuật toán đối chiếu trong `notebooks/03_clustering_models.ipynb`.
- Nhóm đánh giá tổng hợp, vẽ biểu đồ phân tích, mô tả chân dung nhóm khách hàng và thử suy luận mẫu trong `notebooks/04_evaluation_and_profiling.ipynb`.
- Nhóm dự kiến lưu biểu đồ tại `results/figures/`, bảng chỉ số tại `results/metrics/` và phần trình bày tại `docs/reports/report.md`.
- Nhóm ưu tiên làm rõ baseline K-Means trước rồi mới đối chiếu sang các phương pháp còn lại.
