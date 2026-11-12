# THIẾT KẾ BÀI LAB PHÂN CỤM KHÁCH HÀNG

## 1. WHAT - Nhóm sẽ thực hiện nội dung gì?

### 1.1. Bài toán

- Nhóm thực hiện bài lab Customer Clustering trong bối cảnh cửa hàng bán lẻ.
- Bài toán thuộc Unsupervised Learning, cụ thể là phân cụm khách hàng.
- Nhóm tìm cách gom những khách hàng có đặc điểm tương đồng vào cùng một cụm.
- Mỗi cụm là một cách mô tả phân khúc trong phạm vi dữ liệu học tập.
- Nhóm không xem nhãn cụm là nhãn đúng có sẵn của khách hàng.

### 1.2. Dữ liệu đầu vào

- Nguồn raw nội bộ là `data/raw/marketing_campaign.csv` (dạng phân tách tab); nguồn gốc được kiểm tra schema và ý nghĩa cột trước khi xử lý.
- Đối chiếu với yêu cầu bài tập: dữ liệu thô có `Year_Birth`, `Income` và 6 cột chi tiêu (`MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`), chưa có sẵn cột `Age`, `Gender` hay `Spending Score`.
- Nhóm thống nhất quy đổi `Age = 2014 - Year_Birth` (lấy năm tham chiếu `2014` phù hợp khoảng năm gia nhập `Dt_Customer` 2012–2014 trong tập dữ liệu) và tính `Total_Spending` bằng tổng 6 cột chi tiêu `Mnt*` để đo mức chi tiêu của khách hàng.
- Do không có cột `Gender`, nhóm không tự giả lập cột này mà dùng các cột nhân khẩu học sẵn có như `Education`, `Marital_Status` (cùng 6 nhóm chi tiêu thành phần) để mô tả thêm về từng cụm.
- Mã khách hàng (`ID`) chỉ dùng để đối chiếu bản ghi và kiểm tra trùng lặp, không đưa vào tính khoảng cách; các cột phản hồi chiến dịch (`Response`, `AcceptedCmp*`) cũng không dùng làm nhãn đúng.

### 1.3. Phạm vi học tập

- Nhóm thực hiện Data Exploration trên dữ liệu đã kiểm định để nắm phân phối, giá trị thiếu và ngoại lai.
- Nhóm áp dụng tách tập `80% Train / 20% Holdout Test` trước bước điền khuyết bằng trung vị và bước chuẩn hóa nhằm tránh rò rỉ dữ liệu.
- Nhóm thực hiện Feature Scaling cho ba thuộc tính số (`Age`, `Income`, `Total_Spending`).
- Nhóm chọn K-Means làm baseline chính, khảo sát tham số phù hợp và đối chiếu thêm với Hierarchical Clustering, DBSCAN và K-Medoids.
- Nhóm xây dựng kế hoạch đánh giá bằng cả chỉ số nội tại lẫn biểu đồ trực quan, kèm mô tả chân dung nhóm khách hàng theo dữ liệu.
- Phần đóng gói mô hình và thử dự đoán mẫu chỉ phục vụ kiểm chứng cuối trong phạm vi bài lab; việc xây dựng dịch vụ trực tuyến hay vận hành thực tế nằm ngoài phạm vi.

## 2. WHY - Vì sao nhóm thực hiện bài lab?

### 2.1. Mục tiêu học tập

- Nhóm tìm hiểu cách phát hiện cấu trúc dữ liệu khi không có nhãn phân loại.
- Nhóm học cách liên hệ khoảng cách giữa các điểm với mức độ tương đồng.
- Nhóm tìm hiểu ảnh hưởng của thang đo tới thuật toán dựa trên khoảng cách.
- Nhóm thực hành lựa chọn tham số dựa trên tiêu chí đánh giá rõ ràng.
- Nhóm rèn luyện cách giải thích phân cụm bằng các đặc trưng dễ hiểu.

### 2.2. Ý nghĩa của bối cảnh bán lẻ

- Khách hàng có thể khác nhau về độ tuổi, khả năng chi trả và mức chi tiêu.
- Phân cụm giúp nhóm khảo sát các dạng khác biệt trong dữ liệu.
- Mọi diễn giải chỉ mang tính học tập và phụ thuộc vào dữ liệu được cung cấp.
- Nhóm không suy diễn quan hệ nhân quả từ sự tương đồng giữa khách hàng.

### 2.3. Nguyên tắc thiết kế

- Nhóm ưu tiên cách tiếp cận đơn giản, dễ giải thích và vừa sức bài thực hành.
- Nhóm xác định tiêu chí đánh giá trước khi lựa chọn cấu hình mô hình.
- Nhóm bắt đầu bằng baseline để có cơ sở đối chiếu giữa các thử nghiệm.
- Nhóm không ấn định trước số cụm hoặc tên gọi các phân khúc.

## 3. WHO - Ai thực hiện và sử dụng thiết kế?

### 3.1. Đối tượng sử dụng

- Nhóm sinh viên sử dụng tài liệu để thống nhất hướng làm bài lab.
- Giảng viên có thể theo dõi cách nhóm chuyển yêu cầu thành kế hoạch phân tích.
- Thành viên trong nhóm cần hiểu ý nghĩa dữ liệu và các bước phân cụm cơ bản.

### 3.2. Phân công dự kiến

- Nhóm phân công phần kiểm tra dữ liệu và khám phá đặc trưng.
- Nhóm phân công phần chuẩn hóa và chuẩn bị dữ liệu phân cụm.
- Nhóm phân công phần khảo sát thuật toán và tham số.
- Nhóm phân công phần thiết kế biểu đồ và tiêu chí đánh giá.
- Nhóm cùng rà soát tính nhất quán giữa các phần trước khi chuyển giai đoạn.
- Một thành viên có thể phụ trách nhiều phần tùy quy mô nhóm.

## 4. WHERE - Nhóm thực hiện ở đâu?

### 4.1. Môi trường thực hành

- Nhóm thực hiện trên máy tính cá nhân hoặc máy thực hành của lớp.
- Nhóm dự kiến dùng Python và Jupyter Notebook cho quá trình học tập.
- Nhóm ưu tiên các công cụ đã có trong môi trường học tập.
- Dữ liệu gốc được lưu riêng để thuận tiện đối chiếu trong suốt bài lab.

### 4.2. Không gian phân tích

- Nhóm dùng ba thuộc tính số làm không gian phân cụm chính: `Age`, `Income` và `Total_Spending`.
- Nhóm giữ `Education`, `Marital_Status` cùng 6 khoản chi tiêu thành phần (`Mnt*`) để mô tả thêm đặc điểm nhóm khách hàng sau phân cụm.
- Nhóm không đưa mã định danh `ID` hay thông tin phản hồi chiến dịch vào phép tính khoảng cách.
- Biểu đồ hai chiều (`Income` với `Total_Spending`) và ba chiều (`Age`, `Income`, `Total_Spending`) sẽ hỗ trợ quan sát trực quan.

## 5. WHEN - Nhóm thực hiện theo trình tự nào?

### 5.1. Các giai đoạn dự kiến

| Giai đoạn | Nội dung nhóm thực hiện | Điều kiện chuyển bước |
|---|---|---|
| Chuẩn bị | Kiểm tra schema, tạo đặc trưng, lọc lỗi miền và tách `80% Train / 20% Test` | Dữ liệu sạch sơ bộ, không trộn tập Test vào thống kê học |
| Khám phá | Kiểm tra thiếu, trùng lặp 3D, phân phối và dấu hiệu ngoại lai trên tập Train | Có phương án điền khuyết từ trung vị Train và giữ ngoại lai hợp lệ |
| Chuẩn hóa | Khảo sát scaler, fit trên Train, biến đổi Train/Test và hậu kiểm điều kiện | Ma trận Train/Test hữu hạn, không rò rỉ dữ liệu |
| Thử nghiệm | Chạy K-Means baseline (`K=2..10`, thử siêu tham số) rồi đối chiếu Hierarchical, DBSCAN, K-Medoids | Các cấu hình cùng tập Train và cùng thang đo để so sánh công bằng |
| Đánh giá | Tính chỉ số nội tại, đối chiếu ổn định trên Test, vẽ biểu đồ và lập hồ sơ nhóm | Có đủ căn cứ chỉ số và thống kê gốc để giải thích phân khúc |

### 5.2. Nguyên tắc sắp xếp

- Nhóm kiểm tra chất lượng dữ liệu và chia `80/20` trước khi tính trung vị hay tham số chuẩn hóa.
- Nhóm thống nhất đặc trưng và bộ tiêu chí đánh giá trước khi thử nghiệm thuật toán.
- Nhóm hoàn thiện phần giải thích cho K-Means baseline trước khi đối chiếu sang các thuật toán khác.
- Nhóm thay đổi có kiểm soát từng yếu tố (số cụm, siêu tham số, thước đo khoảng cách hoặc scaler) để thấy rõ ảnh hưởng.
- Nhóm dành bước rà soát cuối để ghi nhận rõ giả định và giới hạn của bài lab.
- Thời lượng từng giai đoạn được điều chỉnh linh hoạt theo tiến độ học tập.

## 6. HOW - Nhóm dự kiến thực hiện như thế nào?

### 6.1. Khám phá dữ liệu

- Nhóm kiểm tra số dòng, số cột, kiểu dữ liệu, trùng `ID` và giá trị thiếu trên bảng dữ liệu thô ban đầu.
- Nhóm kiểm tra các dấu hiệu lỗi nhập liệu hoặc nhãn lạ (như `Year_Birth < 1920`, `Income > 200,000`, các nhãn `Marital_Status` ít gặp như `Alone`/`Absurd`/`YOLO`, và khoảng năm gia nhập `Dt_Customer`).
- Sau bước lọc lỗi và tách `80/20`, phần thống kê mô tả chính (`count`, `mean`, `median`, `std`, `min`, `max`, `skew`) và kiểm đếm bản ghi trùng trên 3 đặc trưng số được thực hiện trên tập Train.
- Nhóm dùng Histogram và Boxplot/IQR để khảo sát phân phối, độ lệch (skewness) và phát hiện ngoại lai cho `Age`, `Income`, `Total_Spending`.
- Giá trị nằm ngoài hàng rào IQR chỉ được đánh dấu để quan sát mức độ lệch phân phối, không tự động loại bỏ khi vẫn nằm trong miền khách hàng hợp lệ.
- Nhóm dùng ma trận tương quan (Heatmap) và Scatter Plot (ưu tiên `Income` với `Total_Spending`, có thể xem thêm `Age` với `Total_Spending`) để quan sát sơ bộ các cụm tự nhiên.

### 6.2. Xử lý dữ liệu trước phân cụm

- Nhóm giữ nguyên tệp gốc trong `data/raw/` để đối chiếu khi cần, còn bản sạch lưu riêng trong `data/processed/`.
- Nhóm tạo hai đặc trưng suy dẫn từng dòng: `Age = 2014 - Year_Birth` và `Total_Spending` bằng tổng 6 cột `Mnt*`.
- Nhóm lọc bỏ các dòng trùng `ID` hoặc vi phạm miền hợp lý: `Year_Birth < 1920` hay `> 2014`, `Income < 0` hay `> 200,000`, và khoản chi âm.
- Nhóm tách `80% Train / 20% Holdout Test` (với `random_state = 42`) rồi chỉ tính trung vị (`median`) của `Income` trên tập Train để điền cho phần khuyết ở cả Train và Test, tránh rò rỉ thông tin; các cột phân loại (`Education`, `Marital_Status`) điền theo giá trị phổ biến (`mode`).
- Nhóm giữ các giá trị ngoại lai hợp lệ ngoài khoảng IQR vì đây có thể là nhóm khách hàng chi tiêu cao thật sự.
- Trước khi chuẩn hóa, nhóm kiểm tra lại toàn bộ 3 đặc trưng số để đảm bảo không còn giá trị thiếu hay không hữu hạn.

### 6.3. Feature Scaling

- Nhóm chọn `StandardScaler` làm phương án chuẩn hóa chính vì độ tuổi, thu nhập và tổng chi tiêu có đơn vị và độ lớn rất khác nhau.
- Bộ chuẩn hóa chỉ được fit trên tập Train (80%), sau đó mới transform cho cả Train và Holdout Test (20%), tuyệt đối không fit trên toàn bộ tập dữ liệu.
- Trước khi fit, nhóm kiểm tra để chắc chắn không đưa cột `ID` vào ma trận và không có thuộc tính hằng số (phương sai bằng `0`).
- Bên cạnh `StandardScaler`, nhóm dự kiến đối chiếu thêm `MinMaxScaler` và `RobustScaler` trên cả hai khía cạnh: phân phối sau biến đổi (Min, Max, Mean, Std, Median, IQR, Skew) và hiệu quả phân cụm K-Means khi giữ cố định dữ liệu Train/Test cùng dải K.
- Nhóm kiểm tra hậu điều kiện sau scaling trên Train (`Mean ≈ 0, Std ≈ 1` cho Standard; trong `[0, 1]` cho MinMax; `Median ≈ 0` cho Robust) và theo dõi các mẫu Test vượt cận `[0, 1]` của MinMax do ảnh hưởng của tập kiểm chứng độc lập.
- Nhóm lưu thông tin phép biến đổi để có thể dùng `inverse_transform` quy đổi tâm cụm về đơn vị thực tế khi diễn giải.

### 6.4. Tiêu chí đánh giá dự kiến

- Nhóm dùng Silhouette coefficient làm chỉ số chính trên không gian đã chuẩn hóa; giá trị nằm từ `-1` đến `1` và càng cao càng cho thấy cụm gọn và tách biệt.
- Silhouette và các chỉ số phân tách chỉ được tính khi số cụm thực tế nằm trong khoảng `2` đến `n_valid - 1`; nếu vi phạm, nhóm ghi rõ lý do không tính được thay vì gán điểm `0` giả tạo.
- Với thuật toán mật độ có tách nhiễu (nhãn `-1`), nhóm loại các điểm nhiễu khỏi phép tính chỉ số phân tách và báo cáo riêng số lượng cùng tỷ lệ phần trăm điểm nhiễu.
- Ngoài Silhouette, nhóm bổ sung thêm các chỉ số nội tại hỗ trợ góc nhìn đa chiều: `Davies-Bouldin` (càng thấp càng tốt), `Calinski-Harabasz` (càng cao càng tốt), `Dunn Index`, tỷ lệ điểm có Silhouette âm (%) và độ cân bằng kích thước giữa cụm nhỏ nhất và lớn nhất.
- Nhóm dùng Inertia kết hợp biểu đồ Elbow để khảo sát K-Means, nhưng chỉ so sánh trên cùng dữ liệu, cùng đặc trưng và cùng thang đo, không dùng Inertia cho thuật toán khác hay chọn K chỉ vì Inertia giảm.
- Nhóm không dùng Accuracy vì đây là bài toán không giám sát, và không ấn định sẵn ngưỡng điểm bắt buộc.

### 6.5. Baseline K-Means

- Nhóm chọn K-Means làm baseline chính vì trực quan, tính toán nhanh và tâm cụm dễ quy đổi về đơn vị thực tế.
- Thuật toán hoạt động với khoảng cách Euclidean trên 3 đặc trưng đã chuẩn hóa bằng `StandardScaler` trên tập Train.
- Nhóm quét dải số cụm `K` từ `2` đến `10` (trong giới hạn hợp lệ theo số mẫu phân biệt) với khởi tạo `k-means++`, `n_init = 10`, `max_iter = 300`, cố định `random_state = 42`.
- Từ đường Elbow và bảng Silhouette (kèm Davies-Bouldin, Calinski-Harabasz, độ cân bằng cụm), nhóm chọn số cụm `K` phù hợp rồi thử thêm các tổ hợp siêu tham số (`init` giữa `k-means++` và `random`, `n_init` giữa `10` và `20`, `max_iter` giữa `100` và `300`) trước khi fit mô hình baseline cuối với `n_init = 20`.
- Sau khi huấn luyện trên Train, nhóm gán nhãn thử lên tập Holdout Test bằng tâm cụm đã học để kiểm tra độ ổn định của cấu trúc phân cụm (không dùng chỉ số trên Test để kết luận overfitting như bài học có giám sát).
- Nhóm lưu ý các hạn chế vốn có của K-Means: nhạy cảm với ngoại lai và giả định cụm dạng cầu đồng đều.

### 6.6. Phương án đối chiếu tùy chọn

- Sau khi nắm rõ kết quả baseline K-Means trên tập Train, nhóm dự kiến đối chiếu thêm ba phương án thuộc các họ tiếp cận khác nhau trong giới hạn bài lab:
  1. **Hierarchical Clustering (Agglomerative):** dùng `Ward` linkage (khoảng cách Euclidean) làm nhánh chính để dựng cây phân cấp, tính ngưỡng cắt Dendrogram theo số cụm K, đồng thời đối chiếu thêm `Complete` và `Average` linkage qua hệ số tương quan `Cophenetic` và độ cân bằng cụm để quan sát hiện tượng kéo chuỗi (*chaining*).
  2. **DBSCAN (theo mật độ):** vẽ đồ thị `k-distance` (với `k = 4`) để ước lượng bán kính lân cận `eps`, thử lưới cặp `(eps, min_samples)` nhằm quan sát cách tách cụm mật độ cao và phát hiện điểm nhiễu (nhãn `-1`).
  3. **K-Medoids (chọn điểm thật làm tâm):** cài đặt thuật toán dùng pha khởi tạo BUILD tất định kết hợp bước cập nhật tâm nội cụm, đối chiếu thước đo khoảng cách Euclidean với Manhattan để kiểm tra mức độ ít nhạy cảm với ngoại lai hơn so với tâm trung bình của K-Means.
- Tất cả phương án đối chiếu đều dùng chung dữ liệu Train và cùng ma trận chuẩn hóa với K-Means để so sánh công bằng trên một bảng tổng hợp.

### 6.7. Trực quan hóa và diễn giải dự kiến

- Nhóm dự kiến xuất các nhóm biểu đồ phục vụ báo cáo theo từng giai đoạn: phân phối EDA & ma trận tương quan, đối chiếu 3 phương pháp chuẩn hóa, đường Elbow & Silhouette cho K-Means, Dendrogram cho Hierarchical, đồ thị `k-distance` cho DBSCAN, cùng các biểu đồ cụm 2D (`Income` - `Total_Spending`), 3D (`Age` - `Income` - `Total_Spending`) và Silhouette chi tiết từng cụm.
- Nhãn trục biểu đồ ghi rõ tên đặc trưng và đơn vị gốc khi trực quan hóa trên không gian chưa chuẩn hóa.
- Để diễn giải cụm, nhóm lập bảng hồ sơ thống kê theo đơn vị gốc (quy mô, tỷ lệ %, tuổi trung bình/trung vị, thu nhập trung bình/trung vị, chi tiêu trung bình/trung vị, học vấn và hôn nhân phổ biến, nhóm sản phẩm `Mnt*` chi nhiều nhất).
- Nhóm xây dựng cách đặt tên chân dung khách hàng (persona) linh hoạt từ dữ liệu thống kê của từng cụm cho số cụm `K` bất kỳ, coi nhãn cụm chỉ là mã định danh chứ không gán thứ bậc hay suy diễn cảm tính.

### 6.8. Giới hạn cần tính đến

- Tập dữ liệu chỉ phản ánh một tập khách hàng mẫu trong giai đoạn thu thập gắn với chiến dịch tiếp thị cũ, không đại diện cho toàn bộ thị trường bán lẻ.
- Ba thuộc tính `Age`, `Income`, `Total_Spending` chỉ phản ánh một phần hành vi mua sắm và bỏ qua tần suất giao dịch hay kênh mua hàng khi tính khoảng cách.
- Kết quả phân cụm phụ thuộc vào phép chọn đặc trưng, cách chuẩn hóa và giả định hình dạng cụm của từng thuật toán.
- Cách chia `80/20` chỉ giúp tránh rò rỉ dữ liệu khi chuẩn hóa/điền khuyết và quan sát độ ổn định phân bố trên dữ liệu mới cho K-Means và K-Medoids (còn Hierarchical và DBSCAN không suy diễn trực tiếp cho điểm mới theo cùng cơ chế), chứ không thay thế cho việc kiểm định bằng nhãn thực tế.
- Các phần mở rộng như lưu gói mô hình hay thử dự đoán mẫu chỉ nhằm minh họa quy trình trọn vẹn trong bài lab, không hướng tới vận hành thực tế.

## 7. TARGET STRUCTURE - Cấu trúc thư mục bài lab dự kiến

- Nhóm chia theo trách nhiệm: dữ liệu, tài liệu, thực hành, xử lý, kiểm tra và trình bày.
- Nhóm tách dữ liệu gốc trong `data/raw/` khỏi dữ liệu chuẩn bị trong `data/processed/`.
- Nhóm tổ chức notebook theo bốn bước: EDA, chuẩn hóa, phân cụm và đánh giá.
- Nhóm chia `src/` theo nhiệm vụ: `data/`, `features/`, `models/`, `evaluation/`, `utils/`.
- Nhóm dành `results/` cho lưu trữ đầu ra dự kiến và `docs/reports/` cho phần diễn giải; `tests/` phục vụ kiểm tra.
- Nhóm giữ phạm vi vừa sức; `src/deployment/` và `results/saved_models/` dành cho phần mở rộng.
  Nhóm dùng các tên và vị trí dưới đây; cây lược bớt tệp hỗ trợ, tài liệu phụ và cấu hình môi trường.

```text
lab_3.1_CUSTOMER_CLUSTERING/
├── data/                                    # raw/ và processed/
├── docs/
│   ├── Practice Exercise 3.pdf              # Yêu cầu bài lab
│   ├── Algrothirms/                         # design.md: thiết kế của nhóm
│   └── reports/                                 # report.md: phần trình bày
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_and_scaling.ipynb
│   ├── 03_clustering_models.ipynb
│   └── 04_evaluation_and_profiling.ipynb
├── src/                                     # Các phần xử lý theo nhiệm vụ
├── tests/                                   # Kiểm tra từng phần
├── results/                                 # figures/, metrics/, saved_models/
├── main.py                                  # Điều phối trình tự thực hành
├── requirements.txt                         # Danh sách thư viện
└── README.md                                # Mục tiêu và hướng dẫn
```
