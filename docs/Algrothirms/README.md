# Các thiết kế phương pháp cho bài lab

## Purpose - Mục đích

Nhóm dùng bộ tài liệu này để cụ thể hóa [thiết kế tổng thể](design.md), mục 6.3 đến 6.6.
Phạm vi gồm chuẩn hóa đặc trưng, baseline K-Means và phương án phân cụm phân cấp tùy chọn.

## Danh mục và Status

| Tài liệu | Vai trò | Status của thiết kế |
|---|---|---|
| [Feature Scaling](feature_scaling_design.md) | StandardScaler chính, MinMaxScaler tùy chọn | Dự kiến; chờ thống nhất đặc trưng |
| [K-Means](kmeans_design.md) | Baseline phân cụm | Dự kiến; chờ dữ liệu phù hợp |
| [Hierarchical Clustering](hierarchical_design.md) | Đối chiếu với baseline nếu đủ thời gian | Tùy chọn; thực hiện sau baseline |

Status mô tả kế hoạch học tập, không xác nhận tình trạng triển khai hay chất lượng mô hình.
DBSCAN và K-Medoids không thuộc danh sách dự kiến trong thiết kế tổng thể hiện tại.

## Data used - Điều kiện chung về dữ liệu

- Nhóm tham chiếu nguồn nội bộ `data/raw/marketing_campaign.csv` theo mục 1.2 của thiết kế tổng thể.
- Nhóm cần xác minh nguồn tải gốc, đơn vị và ý nghĩa các thuộc tính trước khi sử dụng.
- Dữ liệu này có `Year_Birth`, `Income` và các khoản chi tiêu, chưa có trực tiếp ba đặc trưng như đề mô tả.
- Nhóm không tự đổi `Year_Birth` thành `Age` nếu chưa xác định năm tham chiếu có căn cứ.
- Nhóm không xem tổng chi tiêu là `Spending Score`, cũng không tự tạo thông tin `Gender`.
- Trước thực nghiệm, nhóm cần thống nhất một trong hai hướng: dùng dữ liệu phù hợp đề hoặc xin chấp thuận điều chỉnh đặc trưng của bài lab.
- Nếu điều chỉnh đặc trưng, nhóm cần cập nhật thiết kế tổng thể và cả ba tài liệu trước khi thực hiện.
- Khi chưa thống nhất, các Input và Output dưới đây là quy ước thiết kế có điều kiện.

## Input và Output chung giữa các bước

| Bước | Input dự kiến | Output chuyển sang bước tiếp theo |
|---|---|---|
| Chuẩn bị | Bảng dữ liệu đã xác minh và lựa chọn đặc trưng | Bảng số sạch, mã đối chiếu và mô tả đặc trưng |
| Scaling | Bảng số sạch theo thứ tự cột cố định | Ma trận chuẩn hóa và thông tin phép biến đổi |
| Phân cụm | Cùng ma trận chuẩn hóa và cấu hình thử nghiệm | Nhãn cụm theo thứ tự khách hàng, thông tin riêng của thuật toán |
| Đánh giá | Ma trận, nhãn cụm, cấu hình và dữ liệu theo đơn vị gốc | Các chỉ số, biểu đồ và mô tả dự kiến phục vụ bài lab |

- Nhóm giữ nguyên thứ tự bản ghi hoặc mã đối chiếu xuyên suốt các bước.
- Nhóm giữ cùng tập khách hàng, đặc trưng và scaler khi so sánh hai thuật toán.
- Nhóm xem thay đổi scaler là một khảo sát riêng để phân biệt ảnh hưởng của tiền xử lý.

## Evaluation metrics - Quy ước đánh giá

- Silhouette là chỉ số chính, chỉ tính khi số nhãn thực tế từ 2 đến số mẫu trừ 1.
- Trường hợp không đủ điều kiện được ghi là không tính được kèm lý do; nhóm không thay bằng điểm 0.
- Inertia và Elbow phục vụ khảo sát K-Means trên cùng không gian đặc trưng.
- Nhóm kết hợp chỉ số với quy mô cụm, Scatter Plot và khả năng diễn giải.
- Nhóm không đặt trước số cụm tối ưu, ngưỡng điểm bắt buộc hoặc tên phân khúc.

## Tổ chức thực hành

- Nhóm thực hiện chuẩn bị và scaling trong `notebooks/02_preprocessing_and_scaling.ipynb`.
- Nhóm thử nghiệm thuật toán trong `notebooks/03_clustering_models.ipynb`.
- Nhóm đánh giá trong `notebooks/04_evaluation_and_profiling.ipynb`.
- Nhóm dự kiến lưu biểu đồ tại `results/figures/`, bảng đánh giá tại `results/metrics/` và phần diễn giải tại `docs/reports/report.md`.
- Nhóm ưu tiên hoàn thành và giải thích baseline trước khi thực hiện các phần tùy chọn.
