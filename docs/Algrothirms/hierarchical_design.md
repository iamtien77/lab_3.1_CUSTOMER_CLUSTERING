# THIẾT KẾ HIERARCHICAL CLUSTERING

## 1. Purpose - Mục đích

- Nhóm dự kiến dùng Agglomerative Clustering làm phương án đối chiếu tùy chọn sau baseline K-Means.
- Nhóm tìm hiểu cách hình thành các cụm bằng quá trình gộp từ từng khách hàng.
- Nhóm dùng Dendrogram để quan sát cấu trúc phân cấp và các mức phân nhóm.
- Phạm vi tuân theo mục 6.6 của [thiết kế chung](design.md), phù hợp một bài lab học tập.

## 2. Data used - Dữ liệu sử dụng

- Nguồn raw nội bộ được nêu trong thiết kế chung là `data/raw/marketing_campaign.csv`.
- Dữ liệu này có `Year_Birth`, `Income` và các khoản chi tiêu, nhưng thiếu `Gender` và `Spending Score` theo đề.
- Nhóm cần thống nhất nguồn dữ liệu hoặc cách xác định đặc trưng trước khi thực hành phân cụm.
- Nhóm không tự xem `Year_Birth` là tuổi hoặc tổng chi tiêu là `Spending Score`.
- Nếu xây dựng tuổi từ năm sinh, nhóm cần thống nhất năm tham chiếu và ý nghĩa của phép biến đổi.
- Thiết kế dự kiến dùng tuổi, thu nhập hằng năm và điểm chi tiêu khi các thuộc tính được xác nhận phù hợp.
- Nhóm sử dụng đúng tập khách hàng, thứ tự bản ghi, đặc trưng và phép scaling của [baseline K-Means](kmeans_design.md).
- Nhóm giữ dữ liệu theo đơn vị gốc và mã khách hàng để hỗ trợ đối chiếu, diễn giải.

## 3. Input - Đầu vào

| Thành phần | Yêu cầu dự kiến |
|---|---|
| Ma trận đặc trưng | Dữ liệu số gồm n khách hàng và các thuộc tính đã thống nhất |
| Dữ liệu chuẩn hóa | Dùng phương án trong [thiết kế Feature Scaling](feature_scaling_design.md) |
| Chất lượng dữ liệu | Không còn giá trị thiếu hoặc không hữu hạn trong ma trận phân cụm |
| Số cụm K | Giá trị thuộc phạm vi khảo sát hợp lệ |
| Quy tắc gộp | Ward linkage với khoảng cách Euclidean |
| Thông tin đối chiếu | Mã khách hàng và dữ liệu gốc tách khỏi ma trận khoảng cách |

## 4. Output - Đầu ra dự kiến

- Nhãn cụm cho từng khách hàng, giữ liên kết với đúng bản ghi đầu vào.
- Thông tin quá trình gộp để phục vụ Dendrogram nếu nhóm thực hiện trực quan hóa phân cấp.
- Bảng mô tả cấu hình, số cụm thực tế, quy mô từng cụm và chỉ số đánh giá hợp lệ.
- Scatter Plot theo các cặp thuộc tính để hỗ trợ quan sát cách phân nhóm.
- Thống kê theo đơn vị gốc để nhóm diễn giải đặc điểm từng cụm.
- Nhãn cụm chỉ là định danh; nhóm không gán ý nghĩa thứ hạng hay tên phân khúc trước.

## 5. Cách thực hiện dự kiến

1. Nhóm kiểm tra điều kiện dữ liệu và dùng lại ma trận chuẩn hóa đã thống nhất với baseline.
2. Ban đầu, mỗi khách hàng được xem là một cụm riêng.
3. Ở mỗi bước, Ward chọn cặp cụm có mức tăng tổng bình phương sai lệch trong cụm nhỏ nhất để gộp.
4. Nhóm xét các mức cắt cây tương ứng với số cụm K trong phạm vi dự kiến.
5. Nhóm đánh giá từng cấu hình trên cùng không gian đặc trưng dùng cho K-Means.
6. Nhóm đối chiếu chỉ số, quy mô cụm và khả năng diễn giải để cân nhắc phương án phù hợp.

## 6. Tham số và điều kiện khảo sát

- Phương pháp dự kiến: Agglomerative; linkage: Ward; khoảng cách: Euclidean.
- Nhóm khảo sát K từ 2 đến 10, giới hạn theo số khách hàng và số điểm đặc trưng phân biệt.
- Khi tính Silhouette, nhóm cần K thực tế nhỏ hơn số khách hàng được đánh giá.
- Nhóm giữ nguyên dữ liệu và scaling giữa hai thuật toán để việc đối chiếu có ý nghĩa.
- Nhóm không mở rộng nhiều kiểu linkage trong phạm vi thực hành ban đầu.
- Nhóm chỉ thực hiện phần tùy chọn khi có đủ thời gian sau baseline.

## 7. Evaluation metrics - Tiêu chí đánh giá

- Silhouette coefficient là chỉ số chính, thống nhất với mục 6.4 của thiết kế chung.
- Nhóm tính Silhouette khi số nhãn cụm thực tế nằm từ 2 đến n - 1.
- Nếu điều kiện không thỏa mãn, nhóm ghi không tính được kèm lý do, không thay bằng điểm giả.
- Nhóm ưu tiên Silhouette cao hơn nhưng vẫn xem xét quy mô cụm và ý nghĩa đặc trưng.
- Nhóm không xem Inertia của K-Means là chỉ số sẵn có để đối chiếu trực tiếp hai thuật toán.
- Dendrogram giúp quan sát mức gộp; độ cao nhánh biểu thị mức gộp theo Ward, không phải điểm Silhouette.
- Nhóm kết hợp Scatter Plot và thống kê cụm; hình chiếu hai chiều chỉ hỗ trợ quan sát.
- Nhóm không dùng Accuracy hoặc cam kết một ngưỡng chất lượng khi chưa có cơ sở dữ liệu phù hợp.

## 8. Hạn chế và điều kiện sử dụng

- Quá trình phân cấp có thể cần nhiều bộ nhớ và thời gian khi số khách hàng tăng.
- Ward có thể chịu ảnh hưởng của thang đo và ngoại lai; nhóm cần rà soát tiền xử lý.
- Quyết định gộp ở bước trước không được đảo ngược trong quá trình Agglomerative.
- Cây nhiều nhánh có thể khó đọc; nhóm có thể trình bày phần rút gọn và ghi rõ cách rút gọn.
- Nhóm dùng phương pháp để khám phá tập dữ liệu của bài lab, không suy diễn khả năng vận hành thực tế.

## 9. Status - Trạng thái thiết kế

- **Dự kiến, tùy chọn:** tài liệu mô tả hướng thực hành sau baseline K-Means.
- **Điều kiện dữ liệu cần thống nhất:** nguồn raw và bộ thuộc tính phải phù hợp với mục tiêu trong thiết kế chung.
- **Điều kiện bắt đầu:** nhóm xác nhận đặc trưng, cách scaling và tiêu chí đánh giá trước khi phân cụm.
- **Điều kiện rà soát:** nhóm kiểm tra đủ liên kết bản ghi, tham số, chỉ số hợp lệ và cách diễn giải dự kiến.
- Nhóm sẽ ghi nhận thay đổi lựa chọn tại thiết kế chung để các tài liệu thuật toán luôn thống nhất.
