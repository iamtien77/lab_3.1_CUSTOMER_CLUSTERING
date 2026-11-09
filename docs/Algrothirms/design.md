# THIẾT KẾ BÀI LAB PHÂN CỤM KHÁCH HÀNG

## 1. WHAT - Nhóm sẽ thực hiện nội dung gì?

### 1.1. Bài toán

- Nhóm thực hiện bài lab Customer Clustering trong bối cảnh cửa hàng bán lẻ.
- Bài toán thuộc Unsupervised Learning, cụ thể là phân cụm khách hàng.
- Nhóm tìm cách gom những khách hàng có đặc điểm tương đồng vào cùng một cụm.
- Mỗi cụm là một cách mô tả phân khúc trong phạm vi dữ liệu học tập.
- Nhóm không xem nhãn cụm là nhãn đúng có sẵn của khách hàng.

### 1.2. Dữ liệu đầu vào

- Nguồn raw nội bộ: `data/raw/marketing_campaign.csv` ; đường dẫn tải gốc cần được nhóm xác minh.
- Nhóm cần đối chiếu dữ liệu raw với đề: dữ liệu có `Year_Birth`, `Income`, các khoản chi tiêu nhưng thiếu `Gender` và `Spending Score`; không xem tổng chi tiêu là điểm chi tiêu tương đương.
- Theo đề bài, `Age` thể hiện tuổi; `Gender` thể hiện giới tính phục vụ mô tả dữ liệu.
- `Annual Income` thể hiện thu nhập hằng năm theo đơn vị của tập dữ liệu.
- `Spending Score` thể hiện mức chi tiêu theo thang điểm của tập dữ liệu.
- Mã khách hàng (`ID` trong dữ liệu raw, hoặc `CustomerID` tùy tập dữ liệu) chỉ dùng để đối chiếu bản ghi.
- Nhóm sẽ kiểm tra tên cột, kiểu dữ liệu và đơn vị trước khi phân tích.

### 1.3. Phạm vi học tập

- Nhóm thực hiện Data Exploration để hiểu đặc điểm dữ liệu.
- Nhóm thực hiện Feature Scaling cho ba thuộc tính số.
- Nhóm lựa chọn thuật toán phân cụm và khảo sát tham số phù hợp.
- Nhóm xây dựng kế hoạch đánh giá bằng chỉ số và biểu đồ.
- Phạm vi bài lab tập trung vào quá trình phân tích trên một tập dữ liệu.
- Việc xây dựng dịch vụ trực tuyến và vận hành thực tế nằm ngoài phạm vi.

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

- Nhóm dùng ba thuộc tính số làm không gian phân cụm chính.
- Ba thuộc tính gồm tuổi, thu nhập hằng năm và điểm chi tiêu.
- Nhóm giữ `Gender` để mô tả phân bố và tìm hiểu đặc điểm dữ liệu.
- Nhóm không đưa thông tin định danh vào phép tính khoảng cách.
- Biểu đồ hai chiều sẽ thể hiện từng cặp thuộc tính để hỗ trợ quan sát.

## 5. WHEN - Nhóm thực hiện theo trình tự nào?

### 5.1. Các giai đoạn dự kiến

| Giai đoạn  | Nội dung nhóm thực hiện                       | Điều kiện chuyển bước              |
| ------------ | ------------------------------------------------- | ---------------------------------------- |
| Chuẩn bị   | Đọc yêu cầu và xác định các thuộc tính | Hiểu vai trò và đơn vị dữ liệu   |
| Khám phá   | Kiểm tra thiếu, ngoại lai và phân phối      | Có phương án xử lý dữ liệu       |
| Chuẩn hóa  | Chọn đặc trưng và phương pháp scaling     | Có dữ liệu số hợp lệ               |
| Thử nghiệm | Thiết lập baseline và khảo sát tham số      | Các cấu hình có thể đối chiếu    |
| Đánh giá  | Áp dụng tiêu chí và cách trực quan hóa    | Có căn cứ để diễn giải phân cụm |

### 5.2. Nguyên tắc sắp xếp

- Nhóm kiểm tra chất lượng dữ liệu trước khi chuẩn hóa.
- Nhóm thống nhất đặc trưng và chỉ số trước khi thử nghiệm thuật toán.
- Nhóm thực hiện baseline trước khi cân nhắc phương án bổ sung.
- Nhóm thay đổi có kiểm soát từng yếu tố để hiểu ảnh hưởng của lựa chọn.
- Nhóm dành thời gian cuối buổi thực hành để rà soát giả định và hạn chế.
- Thời lượng từng giai đoạn sẽ được nhóm điều chỉnh theo lịch học.

## 6. HOW - Nhóm dự kiến thực hiện như thế nào?

### 6.1. Khám phá dữ liệu

- Nhóm kiểm tra số bản ghi, số thuộc tính và kiểu dữ liệu của từng cột.
- Nhóm thống kê số lượng và tỷ lệ giá trị thiếu theo từng thuộc tính.
- Nhóm kiểm tra bản ghi trùng và phân biệt với khách hàng có đặc điểm giống nhau.
- Nhóm xem xét miền giá trị của tuổi, thu nhập và điểm chi tiêu.
- Nhóm xác minh giá trị bất hợp lý dựa trên ý nghĩa và đơn vị của thuộc tính.
- Nhóm sử dụng Histogram để khảo sát phân phối của ba thuộc tính số.
- Nhóm sử dụng Scatter Plot cho cặp thu nhập và điểm chi tiêu.
- Nhóm có thể bổ sung cặp tuổi và điểm chi tiêu để mở rộng góc quan sát.
- Nhóm sử dụng Boxplot và IQR để nhận diện các giá trị cần xem xét.
- Giá trị ngoài khoảng IQR chỉ là dấu hiệu cần kiểm tra, không tự động bị loại.

### 6.2. Xử lý dữ liệu trước phân cụm

- Nhóm giữ nguyên bản dữ liệu gốc để phục vụ đối chiếu.
- Nếu thuộc tính số thiếu ít giá trị, nhóm cân nhắc điền bằng median.
- Nếu một thuộc tính thiếu nhiều, nhóm xem lại khả năng sử dụng trước khi tiếp tục.
- Nhóm ghi rõ lý do nếu loại bản ghi không hợp lệ hoặc trùng do nhập liệu.
- Nhóm giữ các giá trị cực đoan hợp lệ khi chúng phản ánh khách hàng thực tế.
- Nhóm kiểm tra lại giá trị thiếu và giá trị không hữu hạn trước khi scaling.
- Nhóm giữ liên kết giữa từng khách hàng và dữ liệu dùng để phân tích.

### 6.3. Feature Scaling

- Nhóm chọn StandardScaler làm phương án chuẩn hóa ban đầu.
- StandardScaler đưa từng thuộc tính về thang đo theo trung bình và độ lệch chuẩn.
- Nhóm dùng cùng một phương án scaling khi đối chiếu các cấu hình K-Means.
- Nhóm kiểm tra thuộc tính có phương sai bằng không vì thuộc tính đó không phân biệt khách hàng.
- Nhóm cân nhắc MinMaxScaler như một khảo sát bổ sung nếu thời gian cho phép.
- MinMaxScaler đưa dữ liệu dùng để xác định phép biến đổi về khoảng xác định.
- Cả hai phương án đều cần được xem xét khi dữ liệu có giá trị cực đoan.
- Nhóm giữ dữ liệu theo đơn vị gốc để giải thích ý nghĩa của các đặc trưng.

### 6.4. Tiêu chí đánh giá dự kiến

- Nhóm sử dụng Silhouette coefficient làm chỉ số đánh giá chính.
- Silhouette phản ánh mức gắn kết trong cụm so với mức tách biệt với cụm khác.
- Giá trị Silhouette nằm trong khoảng từ -1 đến 1; giá trị cao hơn thường thuận lợi hơn.
- Nhóm chỉ tính Silhouette khi số nhãn cụm nằm từ 2 đến số mẫu trừ 1.
- Nhóm kết hợp chỉ số với quy mô cụm và khả năng diễn giải đặc trưng.
- Nhóm dùng Inertia để hỗ trợ quan sát Elbow trong các thử nghiệm K-Means.
- Nhóm chỉ so sánh Inertia trên cùng dữ liệu, tập đặc trưng và thang đo.
- Nhóm không chọn số cụm chỉ vì Inertia giảm khi tăng số cụm.
- Nhóm không sử dụng Accuracy vì bài toán không cung cấp nhãn phân khúc đúng.
- Nhóm không đặt trước một ngưỡng điểm để cam kết chất lượng phân cụm.

### 6.5. Baseline K-Means

- Nhóm chọn K-Means vì cách hoạt động đơn giản và tâm cụm dễ giải thích.
- K-Means hướng tới giảm tổng bình phương khoảng cách từ điểm tới tâm cụm.
- Nhóm sử dụng khoảng cách Euclidean trên ba thuộc tính đã chuẩn hóa.
- Nhóm dự kiến dùng K = 2 cho baseline ban đầu nếu dữ liệu đủ điều kiện.
- Nhóm dự kiến khảo sát K từ 2 đến 10, giới hạn theo số mẫu và số điểm phân biệt.
- Nhóm dùng khởi tạo k-means++ và nhiều lần khởi tạo để giảm ảnh hưởng của điểm bắt đầu.
- Nhóm cố định random seed cho các đối chiếu cần tái lập.
- Nhóm xem xét đồng thời Silhouette, Elbow và quy mô cụm khi chọn K.
- K-Means phù hợp hơn khi các cụm tương đối gọn và được phân biệt bằng khoảng cách.
- K-Means có thể bị ảnh hưởng bởi ngoại lai, cụm kéo dài hoặc mật độ khác nhau.

### 6.6. Phương án đối chiếu tùy chọn

- Nếu thời gian cho phép, nhóm cân nhắc Hierarchical Clustering để tìm hiểu thêm.
- Nhóm dự kiến dùng Agglomerative với Ward linkage và khoảng cách Euclidean.
- Nhóm khảo sát số cụm trên cùng dữ liệu và thang đo với baseline.
- Nhóm dùng Dendrogram để quan sát quá trình gộp cụm khi thực hiện phương án này.
- Phương pháp phân cấp giúp quan sát nhiều mức phân nhóm nhưng có thể tốn bộ nhớ hơn.
- Nhóm không mở rộng đồng thời nhiều thuật toán khi chưa giải thích được baseline.

### 6.7. Trực quan hóa và diễn giải dự kiến

- Nhóm thiết kế Scatter Plot với màu biểu thị nhãn cụm.
- Trục biểu đồ cần thể hiện rõ tên thuộc tính và đơn vị sử dụng.
- Nhóm dự kiến quan sát các cặp tuổi, thu nhập và điểm chi tiêu.
- Nhóm dùng thống kê trung bình, median và quy mô cụm để hỗ trợ diễn giải.
- Nhóm mô tả cụm theo đặc điểm dữ liệu thay vì gán sẵn tên phân khúc.
- Nhãn cụm chỉ có vai trò định danh, không thể hiện thứ hạng khách hàng.
- Nhóm thận trọng khi các điểm chồng lấp trong hình chiếu hai chiều.

### 6.8. Giới hạn cần tính đến

- Tập dữ liệu học tập có thể không đại diện cho toàn bộ khách hàng bán lẻ.
- Ba thuộc tính số chỉ phản ánh một phần đặc điểm và hành vi khách hàng.
- Việc chọn đặc trưng và scaling có thể làm thay đổi cấu trúc phân cụm.
- Nhóm dự kiến phân tích toàn bộ tập dữ liệu cho mục tiêu khám phá của bài lab.
- Thiết kế này không dùng việc phân cụm để khẳng định khả năng tổng quát hóa.
- Nhóm sẽ mô tả rõ những giả định chưa thể kiểm chứng trong phạm vi thực hành.

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
