# THIẾT KẾ THUẬT TOÁN K-MEANS

## 1. Purpose - Mục đích

- Nhóm dùng K-Means làm baseline chính cho bài lab phân cụm khách hàng.
- Nhóm tìm hiểu cách gom những khách hàng tương đồng về nhân khẩu học và chi tiêu dựa trên khoảng cách Euclidean.
- Nhóm khảo sát ảnh hưởng của số cụm `K`, các siêu tham số khởi tạo và giải thích đặc điểm từng phân khúc theo đơn vị thực tế.
- Thiết kế cụ thể hóa mục 6.5 trong [thiết kế tổng thể](design.md).

## 2. Data used - Dữ liệu sử dụng

- Nguồn raw nội bộ là `data/raw/marketing_campaign.csv`, đã qua bước làm sạch các giá trị sai miền rõ ràng và tách thành tập `Train (80%)` và `Holdout Test (20%)`.
- Ba đặc trưng đưa vào phân cụm gồm `Age = 2014 - Year_Birth`, `Income` và `Total_Spending` (tổng 6 cột chi `Mnt*`); phần khuyết `Income` được điền bằng trung vị của tập Train.
- Không có cột `Gender` trong tập dữ liệu và nhóm không tự tạo giá trị giả; các cột `Education`, `Marital_Status` và 6 nhóm chi tiêu thành phần được giữ lại ở bảng gốc để mô tả cụm.
- Cột `ID` chỉ dùng để đối chiếu bản ghi; các cột phản hồi chiến dịch (`Response`, `AcceptedCmp*`) không được coi là nhãn phân khúc đúng hay đưa vào phép tính khoảng cách.
- Mô hình K-Means chỉ học tâm cụm trên tập `Train (80%)`; tập `Holdout Test (20%)` được dùng ở bước kiểm chứng để quan sát độ ổn định phân bố trên dữ liệu mới.

## 3. Input - Đầu vào

- Ma trận `Train` đã chuẩn hóa theo [thiết kế Feature Scaling](feature_scaling_design.md) (`StandardScaler` là phương án chính), gồm `n` khách hàng và `3` thuộc tính số (`Age`, `Income`, `Total_Spending`).
- Ma trận `Holdout Test` được chuẩn hóa theo đúng tham số học từ `Train`, dùng cho bước kiểm chứng sau khi huấn luyện.
- Ma trận đầu vào phải hữu hạn (không chứa `NaN` hay `Inf`), giữ nguyên thứ tự dòng để khớp với bảng dữ liệu gốc, có ít nhất 3 mẫu và ít nhất 2 điểm phân biệt.

## 4. Output - Đầu ra dự kiến

- Một nhãn cụm cho mỗi khách hàng trên tập `Train` (và nhãn suy đoán cho tập `Holdout Test`), gắn khớp với đúng bản ghi gốc.
- `K` tâm cụm trong không gian chuẩn hóa, kèm bộ tâm quy đổi ngược về đơn vị thực tế để vẽ biểu đồ và giải thích.
- Bảng kết quả quét `K` (`Inertia`, `Silhouette`, `Davies-Bouldin`, `Calinski-Harabasz`), bảng thử nghiệm siêu tham số và thông tin số vòng lặp hội tụ.
- Bảng quy mô cụm, hồ sơ thống kê đặc trưng gốc và tên chân dung khách hàng (persona) dựa theo dữ liệu.
- Thông tin đầu ra là cấu trúc thiết kế dự kiến, không ấn định sẵn số cụm tối ưu hay tên phân khúc trước khi chạy thử nghiệm.

## 5. Tham số dự kiến

| Thành phần | Lựa chọn thiết kế | Mục đích |
|---|---|---|
| Dải khảo sát `K` | Từ `2` đến `10` (giới hạn nhỏ hơn số điểm phân biệt) | Quan sát Elbow và tìm số cụm tách biệt tốt |
| Khởi tạo khi quét `K` | `init = "k-means++"`, `n_init = 10`, `max_iter = 300` | Quét đều và ổn định trên dải `K = 2..10` |
| Thử siêu tham số tại `K` chọn | `init` (`k-means++`, `random`), `n_init` (`10`, `20`), `max_iter` (`100`, `300`) | Kiểm tra độ nhạy với điểm khởi tạo và số vòng lặp |
| Cấu hình baseline cuối | `init = "k-means++"`, `n_init = 20`, `max_iter = 300` | Giảm rủi ro kẹt cực tiểu cục bộ cho mô hình chốt |
| Random seed | `42` | Giữ kết quả lặp lại nhất quán giữa các lần chạy |

Khi khảo sát riêng `K` trong dải `2..10`, nhóm giữ cố định tập dữ liệu, bộ chuẩn hóa và các tham số khởi tạo để so sánh công bằng giữa các mức `K`.

## 6. Trình tự thực hiện dự kiến

1. Kiểm tra ma trận `Train` đã chuẩn hóa bảo đảm 2 chiều, hữu hạn, đủ số lượng mẫu và điểm phân biệt.
2. Quét từng giá trị `K` từ `2` đến `10` với `k-means++`, `n_init = 10`, ghi nhận `Inertia`, `Silhouette`, `Davies-Bouldin` và `Calinski-Harabasz`.
3. Quan sát đồ thị Elbow + Silhouette cùng độ cân bằng cụm để chọn `K` phù hợp.
4. Thử nghiệm lưới siêu tham số (`init`, `n_init`, `max_iter`) tại giá trị `K` vừa chọn để kiểm tra tính ổn định hội tụ.
5. Huấn luyện mô hình K-Means cuối cùng trên tập `Train` với `n_init = 20`, lưu các tâm cụm và quy đổi tâm về đơn vị gốc.
6. Dự đoán gán nhãn cho tập `Holdout Test` bằng các tâm cụm đã học để đối chiếu tính nhất quán của phân bố cụm.
7. Chuyển nhãn cụm và tâm cụm sang bước đánh giá, trực quan hóa 2D/3D, Silhouette từng mẫu và lập hồ sơ nhóm khách hàng.

## 7. Evaluation metrics - Tiêu chí đánh giá

- `Silhouette coefficient` trên không gian chuẩn hóa là chỉ số đánh giá chính; chỉ tính khi số cụm thực tế nằm từ `2` đến `n - 1` (nếu vi phạm thì ghi trạng thái không đủ điều kiện, không gán giá trị `0` giả).
- Kết hợp thêm `Davies-Bouldin` (càng thấp càng tốt), `Calinski-Harabasz` (càng cao càng tốt), `Dunn Index`, tỷ lệ điểm có Silhouette âm (%) và độ cân bằng kích thước cụm (`cluster_balance_ratio`).
- Nhóm dùng `Inertia` (WCSS) để vẽ biểu đồ Elbow, chỉ đối chiếu trên cùng tập dữ liệu và cùng thang đo, không chọn `K` lớn hơn chỉ vì `Inertia` giảm đơn điệu.
- Nhóm kết hợp chỉ số với thống kê theo đơn vị thực tế, biểu đồ Scatter 2D/3D và biểu đồ Silhouette từng cụm.
- Đối với tập `Holdout Test (20%)`, chỉ số nội tại chỉ dùng để xem sự ổn định của phân cụm trên dữ liệu mới, không hiểu nhầm thành thước đo kiểu Accuracy hay chứng minh overfitting như bài toán có giám sát.

## 8. Hạn chế và điều kiện cần kiểm tra

- K-Means dễ chịu ảnh hưởng của điểm ngoại lai vì lấy trung bình cộng làm tâm cụm (do đó bài lab cần đối chiếu thêm với K-Medoids).
- Thuật toán giả định các cụm tương đối gọn, dạng cầu; với cụm kéo dài hoặc mật độ chênh lệch mạnh thì kết quả có thể chưa phản ánh hết cấu trúc dữ liệu (cần đối chiếu với DBSCAN).
- Đường Elbow có thể cong mượt không rõ điểm gãy, nên việc chọn `K` cần kết hợp cả Silhouette, độ cân bằng cụm và khả năng diễn giải thực tế.
- Nhãn cụm chỉ là số định danh, không thể hiện thứ hạng cao thấp giữa các khách hàng.
- Nhóm kiểm tra để chắc chắn số cụm sinh ra đúng bằng `K` đã chọn và không có cụm rỗng trước khi diễn giải.

## 9. Status - Trạng thái thiết kế

- Vai trò: phương án baseline chính trong kế hoạch bài lab.
- Trạng thái: đã thống nhất bộ 3 đặc trưng đầu vào, quy trình chia `Train/Test 80/20`, phương pháp `StandardScaler` và dải tham số khảo sát.
- Tài liệu mô tả thiết kế thuật toán cho bước thực nghiệm, không dùng để khẳng định việc triển khai mã nguồn hay báo cáo kết quả thực nghiệm.
