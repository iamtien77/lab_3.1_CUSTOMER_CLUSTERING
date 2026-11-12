# THIẾT KẾ HIERARCHICAL CLUSTERING

## 1. Purpose - Mục đích

- Nhóm dùng Agglomerative Clustering làm nhánh đối chiếu quan trọng ngay sau baseline K-Means.
- Nhóm tìm hiểu cách hình thành các cụm theo cấu trúc phân cấp từ dưới lên (gộp dần từ từng khách hàng).
- Nhóm dùng Dendrogram cùng hệ số tương quan Cophenetic để quan sát mức độ bảo toàn khoảng cách và chọn ngưỡng cắt cụm.
- Phạm vi tuân theo mục 6.6 của [thiết kế chung](design.md), phù hợp bài lab học tập.

## 2. Data used - Dữ liệu sử dụng

- Nguồn raw nội bộ là `data/raw/marketing_campaign.csv`, đã được làm sạch lỗi miền, tách `Train (80%)` / `Holdout Test (20%)` và điền khuyết `Income` bằng trung vị của tập Train.
- Nhóm dùng đúng 3 đặc trưng số đã thống nhất (`Age = 2014 - Year_Birth`, `Income`, `Total_Spending`), cùng tập mẫu `Train (80%)`, thứ tự dòng và bộ chuẩn hóa `StandardScaler` với [baseline K-Means](kmeans_design.md).
- Cột `Gender` không có trong tập dữ liệu nên nhóm không tự tạo; mã `ID` tách riêng khỏi phép tính khoảng cách, còn `Education`, `Marital_Status` và 6 cột `Mnt*` giữ ở bảng gốc để diễn giải cụm.

## 3. Input - Đầu vào

| Thành phần | Yêu cầu thiết kế |
|---|---|
| Ma trận đặc trưng | Ma trận `Train (80%)` chuẩn hóa gồm `n` khách hàng và `3` đặc trưng số |
| Dữ liệu chuẩn hóa | Dùng `StandardScaler` fit trên Train theo [thiết kế Feature Scaling](feature_scaling_design.md) |
| Chất lượng dữ liệu | Không còn giá trị thiếu (`NaN`) hoặc không hữu hạn (`Inf`) |
| Số cụm `K` | Khảo sát trong dải `2` đến `10` (thỏa điều kiện hợp lệ theo số mẫu) |
| Quy tắc gộp | `Ward` linkage (Euclidean) làm chính; đối chiếu thêm `Complete` và `Average` |
| Thông tin đối chiếu | Mã `ID` và các cột theo đơn vị gốc giữ riêng để giải thích cụm |

## 4. Output - Đầu ra dự kiến

- Nhãn cụm cho từng khách hàng trên tập Train, khớp đúng thứ tự bản ghi đầu vào.
- Ma trận liên kết phân cấp (`linkage matrix`), hệ số tương quan **Cophenetic** cho các kiểu gộp (`Ward`, `Complete`, `Average`) và ngưỡng cắt tương ứng với `K` đã chọn để vẽ đường cắt trên Dendrogram.
- Bảng quét `K = 2..10` gồm các chỉ số `Silhouette`, `Davies-Bouldin`, `Calinski-Harabasz`.
- Bảng đối chiếu giữa phương pháp phân cấp và K-Means cùng trên một tập dữ liệu Train, kèm thống kê theo đơn vị gốc.
- Nhãn cụm chỉ mang tính định danh, không suy diễn thứ hạng trước khi phân tích dữ liệu cụm.

## 5. Cách thực hiện dự kiến

1. Kiểm tra ma trận chuẩn hóa `Train (80%)` hữu hạn và dùng lại đúng dữ liệu đầu vào của K-Means baseline.
2. Tính ma trận liên kết và hệ số tương quan `Cophenetic` cho `Ward` (chính) cùng `Complete` và `Average`, đồng thời thử tách `K = 2` tham chiếu để quan sát độ cân bằng kích thước cụm và hiện tượng kéo chuỗi (*chaining*).
3. Với phương pháp chính `Ward`, quét dải `K = 2..10` để ghi nhận `Silhouette`, `Davies-Bouldin`, `Calinski-Harabasz` và chọn `K` có chất lượng tách cụm tốt nhất.
4. Huấn luyện mô hình Agglomerative `Ward` tại `K` đã chọn và tính ngưỡng cắt tương ứng trên ma trận liên kết để vẽ Dendrogram.
5. Đưa nhãn cụm `Ward` vào bảng đối chiếu chung với K-Means, DBSCAN và K-Medoids trên cùng tập Train.

## 6. Tham số và điều kiện khảo sát

- Phương pháp chính: `AgglomerativeClustering` với `linkage = "ward"` và khoảng cách `euclidean` (lưu ý `ward` bắt buộc dùng Euclidean).
- Phương pháp đối chiếu linkage: thử thêm `complete` và `average` (module vẫn có thể hỗ trợ kiểm tra `single` nếu cần khảo sát trường hợp cực đoan), đo `Cophenetic correlation` và tỷ lệ cân bằng cụm tại `K = 2`.
- Dải khảo sát số cụm: `K` từ `2` đến `10` (trong giới hạn `2 <= K < n_samples`).
- Giữ nguyên tập dữ liệu Train và ma trận `StandardScaler` giữa Hierarchical và K-Means để việc so sánh có ý nghĩa.

## 7. Evaluation metrics - Tiêu chí đánh giá

- `Silhouette coefficient` trên ma trận chuẩn hóa Train là chỉ số chính, áp dụng khi số cụm nằm từ `2` đến `n - 1` (không gán giá trị `0` giả nếu không thỏa điều kiện).
- Kết hợp thêm `Davies-Bouldin`, `Calinski-Harabasz`, `Dunn Index`, tỷ lệ Silhouette âm (%) và độ cân bằng kích thước cụm khi đối chiếu toàn bảng mô hình.
- Hệ số tương quan **Cophenetic** dùng để đo mức độ cây phân cấp bảo toàn khoảng cách đôi ban đầu giữa các điểm ở từng kiểu `linkage`.
- Không dùng `Inertia` của K-Means làm thước đo cho Hierarchical.
- Độ cao nhánh trên Dendrogram phản ánh khoảng cách/mức tăng phương sai khi gộp theo `Ward`, không phải điểm Silhouette.

## 8. Hạn chế và điều kiện sử dụng

- Agglomerative đòi hỏi tính khoảng cách cặp và lưu ma trận liên kết nên tốn bộ nhớ và thời gian hơn K-Means khi số lượng khách hàng tăng lớn.
- Khác với K-Means hay K-Medoids có tâm cụm để gán nhãn trực tiếp lên `Holdout Test (20%)`, Hierarchical thuộc nhóm thuật toán phân cụm trên tập mẫu hiện có (không có bước suy đoán độc lập lên điểm mới nếu không gán lại theo láng giềng gần nhất hoặc gom lại).
- Quyết định gộp ở bước trước không thể đảo ngược ở các bước sau; riêng `Complete` hoặc `Average` (hay `Single`) dễ gặp tình trạng gom lệch một cụm rất nhỏ và một cụm rất lớn (*chaining effect*) khi dữ liệu có ngoại lai.
- Cây nhiều nhánh khó quan sát hết chi tiết lá, nên cần cắt nhánh hiển thị hợp lý kèm đường ngưỡng cắt khi vẽ Dendrogram.

## 9. Status - Trạng thái thiết kế

- **Đã thống nhất trong phạm vi bài lab:** dùng `Ward` (Euclidean) làm nhánh đối chiếu chính bên cạnh baseline K-Means, có đối chiếu `Complete` và `Average` bằng `Cophenetic`.
- **Điều kiện dữ liệu:** sử dụng chung bộ 3 đặc trưng số trên tập `Train (80%)` và chuẩn hóa `StandardScaler`.
- Tài liệu mô tả thiết kế phương pháp trước thực nghiệm, không báo cáo tiến độ mã nguồn hay số liệu kết quả chạy.
