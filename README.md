# BÀI LAB: PHÂN CỤM KHÁCH HÀNG (CUSTOMER CLUSTERING)

Báo cáo tổng hợp quá trình thực hiện và kết quả của bài thực hành **Practice Exercise 3 — Retail Customer Segmentation**. Mục tiêu của bài lab là phân nhóm khách hàng bán lẻ dựa trên đặc điểm nhân khẩu học và mức chi tiêu từ bộ dữ liệu `marketing_campaign.csv`, qua đó quan sát cách các thuật toán phân cụm hoạt động và rút ra chân dung từng nhóm khách hàng.

> **Lưu ý về phạm vi:** Đây là bài tập thực hành môn Học máy nên dữ liệu, chỉ số đánh giá và phần suy luận thử nghiệm qua dòng lệnh chỉ phục vụ mục đích học tập và đối chiếu phương pháp, chưa hướng tới triển khai cho hệ thống kinh doanh thực tế.

---

## 1. Tóm Tắt Bài Toán & Điều Chỉnh Dữ Liệu Thực Tế

Đề bài gốc trong [Practice Exercise 3.pdf](<docs/Practice%20Exercise%203.pdf>) đặt bối cảnh dữ liệu gồm tuổi (`age`), giới tính (`gender`), thu nhập năm (`annual income`) và điểm chi tiêu (`spending score`). Tuy nhiên, file dữ liệu thực tế `data/raw/marketing_campaign.csv` (định dạng TSV, **2.240 dòng x 29 cột**) không có sẵn 3 cột `Age`, `Gender` và `Spending Score (1-100)`.

Để bám sát ý đồ của đề mà không tự bịa thêm dữ liệu giả, bài lab xử lý như sau:

- **Tuổi (`Age`):** Quy đổi từ năm sinh theo công thức `Age = 2014 - Year_Birth`. Mốc `2014` được chọn vì thời điểm khách hàng gia nhập (`Dt_Customer`) trong tập dữ liệu nằm trong giai đoạn `2012–2014`.
- **Tổng chi tiêu (`Total_Spending`):** Cộng tổng 6 khoản chi tiêu sản phẩm có sẵn gồm `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts` và `MntGoldProds`.
- **Các biến còn lại:** Không tự sinh cột `Gender` giả; cột `ID` chỉ giữ để định danh và loại khỏi phép tính khoảng cách; các biến nhân khẩu học thật (`Education`, `Marital_Status`) được giữ lại để đọc đặc điểm cụm sau khi mô hình chạy xong.

---

## 2. Khám Phá & Làm Sạch Dữ Liệu (EDA)

Khi kiểm tra bộ dữ liệu thô (`results/metrics/raw_data_audit.csv`), các vấn đề chính được ghi nhận gồm:

- **Tính duy nhất & dữ liệu khuyết:** Toàn bộ `2.240` mã `ID` không bị trùng hay thiếu. Duy nhất cột `Income` bị khuyết **24 giá trị** (chiếm `1,07%`).
- **Giá trị bất hợp lý:** Xuất hiện **3 khách hàng có `Year_Birth < 1920`** (`1893`, `1899`, `1900`, tương ứng 114–121 tuổi) và **1 khách hàng có `Income = 666,666.0`** (cao hơn gấp 4 lần mức cao kế tiếp là `162,397.0`). Cột `Marital_Status` có vài nhãn hiếm như `Alone` (`3`), `Absurd` (`2`), `YOLO` (`2`), nhưng được giữ nguyên vì không tham gia tính khoảng cách phân cụm.

**Quy trình xử lý trước khi đưa vào mô hình:**

1. Loại bỏ `4` bản ghi dị biệt cực đoan kể trên, còn lại **2.236 khách hàng** hợp lệ.
2. Chia cố định **80% Train (`1.788` mẫu)** và **20% Holdout Test (`448` mẫu)** với `RANDOM_SEED = 42`.
3. Tính trung vị `Income = 51,650.50` duy nhất trên tập Train rồi dùng giá trị đó để điền cho `18` dòng thiếu của Train và `6` dòng thiếu của Test. Việc chia tập trước khi điền khuyết và chuẩn hóa nhằm tránh làm lộ thông tin từ tập Test sang quá trình huấn luyện (data leakage).

**Phân bố các đặc trưng trên tập Train (`1.788` mẫu):**

- `Age`: trải từ `18` đến `74`, trung bình `45.23`, trung vị `44.0`, độ lệch chuẩn `11.85` (phân bố tương đối đối xứng, `skew = 0.09`).
- `Income`: từ `2,447.0` đến `162,397.0`, trung bình `51,975.56`, trung vị `51,650.50`, độ lệch chuẩn `21,567.02` (`skew = 0.41`).
- `Total_Spending`: từ `5.0` đến `2,524.0`, trung bình `605.60`, trung vị `397.50`, độ lệch chuẩn `598.82` (lệch phải khá rõ với `skew = 0.86`, phần lớn khách hàng chi ở mức thấp đến trung bình và một nhóm nhỏ chi rất cao).

---

## 3. Chuẩn Hóa Đặc Trưng (Feature Scaling)

Vì độ lệch chuẩn của `Income` (`21,560.98`) lớn gấp gần 2.000 lần so với `Age` (`11.85`), nếu tính khoảng cách Euclidean trực tiếp trên thang đo gốc thì biến `Age` gần như không còn tiếng nói. Do đó cả 3 phương pháp `StandardScaler`, `MinMaxScaler` và `RobustScaler` đều được fit chỉ trên tập Train và thử nghiệm trực tiếp trên K-Means (`results/metrics/scaler_clustering_comparison.csv`):

| Phương pháp chuẩn hóa           | $K$ tối ưu | Silhouette (Train) |  Davies–Bouldin | Calinski–Harabasz | Độ cân bằng cụm ($K=2$) | Silhouette (Test) | Mẫu Test vượt cận Train |
| ------------------------------------ | -------------: | -----------------: | ---------------: | -----------------: | -----------------------------: | ----------------: | --------------------------: |
| **`StandardScaler` (Chọn)** |              2 |             0.4119 | **0.9892** | **1,495.38** |                         0.7444 |            0.4453 |                           3 |
| `MinMaxScaler`                     |              2 |   **0.4210** |           0.9963 |           1,463.04 |                         0.6602 |  **0.4561** |       3 (vượt `[0, 1]`) |
| `RobustScaler`                     |              2 |             0.4026 |           1.0127 |           1,422.45 |               **0.7598** |            0.4368 |                           3 |

**Nhận xét lựa chọn:** Bài lab giữ `StandardScaler` làm phương án chính theo design để đưa các đặc trưng về phương sai đơn vị. MinMax có Silhouette cao hơn; Standard có Davies–Bouldin thấp hơn, Calinski–Harabasz cao hơn và cụm cân bằng hơn. Đây là trade-off trên các không gian khoảng cách khác nhau, chưa chứng minh một scaler luôn tốt hơn. **Cả ba scaler đều có 3 mẫu Test vượt min/max Train**; với MinMax, chúng nằm ngoài `[0, 1]` (Income chuẩn hóa `-0.0045`, Total_Spending tối đa `1.0004`). Điều này không có nghĩa thu nhập gốc âm, không phải lỗi scaler và không chứng minh Standard ổn định hơn. Test chỉ phục vụ quan sát; lựa chọn scaler được giữ theo design và phân tích Train.

---

## 4. Thử Nghiệm & So Sánh Các Thuật Toán Phân Cụm

Trên cùng không gian `StandardScaler` của tập Train (`1.788` mẫu x `3` đặc trưng), bốn thuật toán được khảo sát gồm `K-Means` (làm baseline chính), `K-Medoids`, `Hierarchical Clustering` và `DBSCAN`:

- **K-Means:** Quét $K = 2 \ldots 10$ kết hợp Elbow và Silhouette cho thấy đỉnh Silhouette rơi tại **$K = 2$ (`0.4119`)** và giảm dần khi tăng $K$ (`K=3: 0.3662`, `K=4: 0.3605`, `K=5: 0.3527`). Thử thay đổi `init` (`k-means++`, `random`), `n_init` (`10`, `20`) và `max_iter` (`100`, `300`) đều hội tụ về cùng nghiệm, trong đó `k-means++` dừng sớm hơn (`8` vòng so với `9` vòng).
- **Hierarchical Clustering:** Khi so sánh các kiểu liên kết ở $K=2$, `complete` và `average` cùng dính **hiệu ứng xích (chaining)** — chỉ tách riêng 8 điểm chi tiêu/thu nhập cao ở rìa và gộp 1.780 điểm còn lại thành 1 cụm lớn (độ cân bằng cụm chỉ `0.0045`). Riêng `ward` tối thiểu hóa phương sai nội cụm cho hai cụm cân đối (`1.049` và `739`, độ cân bằng `0.7045`) và khi quét $K=2 \ldots 10$ cũng đạt đỉnh tại $K=2$ (`0.3948`).
- **K-Medoids:** Cài đặt theo hướng chọn điểm thật làm tâm cụm (pha `BUILD` và cập nhật medoid nội cụm), quét $K=2 \ldots 10$ tiếp tục đạt cao nhất ở $K=2$ (`0.4080`), trong đó khoảng cách `euclidean` (`0.4080`) nhỉnh nhẹ hơn `manhattan` (`0.4033`).
- **DBSCAN:** Lưới tìm kiếm trên `eps ∈ [0.3..0.8]` và `min_samples ∈ {4, 5, 8, 10}` cho thấy các ngưỡng `eps` lớn gộp hầu hết dữ liệu thành 1 cụm duy nhất nên không tính được Silhouette. Cấu hình khả dĩ nhất là `eps = 0.5, min_samples = 4`, chia được `3` cụm (`1.756 / 5 / 3` khách hàng) và `24` điểm nhiễu (`1.34%`), độ cân bằng rất thấp (`0.0017`) do mật độ dữ liệu khá liên tục, không tách thành các "đảo" rời rạc.

Bảng tổng hợp kết quả trên tập Train (`results/metrics/model_comparison.csv`):

| Thuật toán                      | Số cụm $K$ | Điểm nhiễu (%) | Silhouette Score | Tỷ lệ Silhouette âm (%) | Davies–Bouldin | Calinski–Harabasz | Dunn Index | Độ cân bằng cụm |
| --------------------------------- | -------------: | ----------------: | ---------------: | -------------------------: | --------------: | -----------------: | ---------: | -------------------: |
| **K-Means (K=2)**           |              2 |         0 (0.00%) | **0.4119** |            **1.68%** |          0.9892 | **1,495.38** |     0.0161 |               0.7444 |
| **K-Medoids (K=2)**         |              2 |         0 (0.00%) |           0.4080 |                      2.80% |          0.9918 |           1,486.32 |     0.0106 |     **0.8079** |
| **Hierarchical Ward (K=2)** |              2 |         0 (0.00%) |           0.3948 |                      5.03% |          1.0195 |           1,387.73 |     0.0200 |               0.7045 |
| **DBSCAN (eps=0.5, ms=4)**  |              3 |        24 (1.34%) |           0.3596 |                     14.51% |          0.5214 |              19.06 |     0.0704 |               0.0017 |

**Kết luận chọn mô hình:** Cả ba mô hình phân cụm theo khoảng cách đều thống nhất cấu trúc rõ nhất trên không gian 3 chiều `[Age, Income, Total_Spending]` là **2 cụm**, không phải 5 cụm như giả định cảm tính ban đầu. **`K-Means (K=2)`** được chọn làm mô hình đại diện do dẫn đầu về Silhouette (`0.4119`), Calinski–Harabasz (`1,495.38`) và có ít điểm nằm ở vùng ranh giới mơ hồ nhất (`1.68%` Silhouette âm). Khi áp bộ tâm cụm này sang tập Holdout Test (`448` khách hàng độc lập), kết quả vẫn giữ ổn định với `Silhouette = 0.4453` và `Davies–Bouldin = 0.8977`.

---

## 5. Chân Dung Phân Khúc Khách Hàng & Thử Nghiệm Suy Luận

Dựa trên bảng thống kê từ `results/metrics/cluster_profiles.csv`, hai nhóm khách hàng được mô tả như sau:

1. **Cụm 0 — Nhóm khách hàng chi tiêu & thu nhập cao (`763` người, `42.67%` tập Train):**

   - Tuổi trung bình `47.64` (trung vị `48`), thu nhập trung bình `$71,704.57` (trung vị `$70,647`), chi tiêu trung bình `$1,191.48` (trung vị `$1,150`).
   - Nhân khẩu học phổ biến là `Graduation (52.9%)` và `Married (37.4%)`; nhóm hàng chi mạnh nhất là `MntWines` (trung bình `605.6`).
   - *Hướng tiếp cận gợi ý:* Duy trì chăm sóc khách hàng thân thiết, ưu tiên truyền thông các dòng sản phẩm chủ lực và ưu đãi cá nhân hóa.
2. **Cụm 1 — Nhóm khách hàng tiết kiệm / nhạy cảm giá (`1,025` người, `57.33%` tập Train):**

   - Tuổi trung bình `43.43` (trung vị `42`), thu nhập trung bình `$37,289.47` (trung vị `$37,401`), chi tiêu trung bình `$169.48` (trung vị `$85`).
   - Nhân khẩu học phổ biến là `Graduation (49.3%)` và `Married (40.2%)`; mức chi tiêu thấp đều trên các nhóm hàng (`MntWines` trung bình `81.7`).
   - *Hướng tiếp cận gợi ý:* Dùng gói combo giá hợp lý, khuyến mãi theo mùa hoặc chương trình tích điểm đổi quà để kích thích tần suất mua.

Ngoài bước đánh giá tĩnh, bài lab cài đặt thêm tiện ích dòng lệnh `src/deployment/predictor.py` nạp lại gói `final_kmeans_pipeline.pkl` để thử phân nhóm cho khách hàng mới. Ví dụ với khách hàng 35 tuổi, thu nhập `75,000`, tổng chi tiêu `1,400`, mô hình gán vào **Cụm 0** với khoảng cách tới tâm cụm chuẩn hóa là `1.1328`.

---

## 6. Hạn Chế Của Bài Lab & Hướng Cải Thiện

- **Dữ liệu cũ và thiếu chiều giới tính:** Bộ `marketing_campaign.csv` được thu thập giai đoạn `2012–2014`, nên các con số chi tiêu chỉ mang ý nghĩa học thuật tại thời điểm khảo sát. Dữ liệu thật cũng không có biến `Gender` như mô tả tổng quát trong đề.
- **Mất chi tiết khi gộp biến chi tiêu:** Việc cộng 6 cột `Mnt*` thành `Total_Spending` giúp giữ bài toán đúng không gian 3 chiều để trực quan hóa 2D/3D, nhưng đánh đổi thông tin sở thích theo từng mặt hàng và hành vi kênh mua (`NumWebPurchases`, `NumStorePurchases`, `Recency`).
- **Phân phối lệch phải và các điểm biên:** `Total_Spending` có độ lệch phải `0.86`, làm một bộ phận nhỏ (`1.68%`) khách hàng có thu nhập cao nhưng chi tiêu rất ít hoặc ngược lại rơi vào vùng giao thoa giữa hai cụm. Hướng mở rộng hợp lý là thử biến đổi `log1p(Total_Spending)` hoặc bổ sung thêm các đặc trưng hành vi mua sắm để xem liệu có tách được các phân khúc phụ mịn hơn bên trong 2 nhóm chính hay không.
- **Mức độ triển khai:** Phần lưu mô hình (`.pkl`) và `predictor.py` mới dừng ở mức script minh họa và kiểm thử đơn vị (`unittest`) cho bài lab, chưa có API web hay cơ chế theo dõi sai lệch dữ liệu thực tế.

Bảng biểu từng vòng quét và hình vẽ được tạo tại `results/metrics/` và `results/figures/` khi chạy `main.py`. Xem [bảng đối chiếu yêu cầu và mã nguồn](docs/requirements_mapping.md). Các artifact được Git ignore, cần chạy lại sau khi clone; README là báo cáo tổng hợp hiện tại.

---

## 7. Phụ Lục Kỹ Thuật: Cấu Trúc Mã Nguồn & Cách Chạy Lại

### Cấu trúc thư mục dự án

```text
lab_3.1_CUSTOMER_CLUSTERING/
├── .venv/                          # Môi trường ảo Python cục bộ
├── data/
│   ├── raw/
│   │   └── marketing_campaign.csv   # Dữ liệu gốc (TSV, 2.240 dòng x 29 cột)
│   └── processed/                  # Đầu ra train/test/full khi chạy main.py
├── docs/
│   ├── Algrothirms/                 # Ghi chú thiết kế các thuật toán (trước khi code)
│   ├── requirements_mapping.md     # Bảng đối chiếu yêu cầu của đề bài với mã nguồn
│   └── Practice Exercise 3.pdf      # Đề bài gốc
├── notebooks/                      # 4 notebook chạy từng bước theo đề
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_and_scaling.ipynb
│   ├── 03_clustering_models.ipynb
│   ├── 04_evaluation_and_profiling.ipynb
│   └── results/                    # Thư mục lưu đầu ra riêng của notebook (không đè lên main.py)
│       ├── processed/
│       ├── figures/
│       ├── metrics/
│       └── saved_models/
├── results/                        # Thư mục lưu đầu ra khi chạy main.py
│   ├── figures/                    # Biểu đồ phân phối, Elbow, 2D/3D, Dendrogram, Silhouette, DBSCAN
│   ├── metrics/                    # Các bảng audit, quét tham số, so sánh mô hình và log chạy
│   └── saved_models/               # Gói mô hình final_kmeans_pipeline.pkl
├── src/                            # Mã nguồn chính chia theo từng bước xử lý
│   ├── config.py                   # Cấu hình đường dẫn, hạt giống ngẫu nhiên và danh sách cột
│   ├── data/                       # loader.py, preprocessing.py
│   ├── features/                   # scaling.py (Standard/MinMax/Robust + kiểm tra hậu điều kiện)
│   ├── models/                     # kmeans.py, hierarchical.py, dbscan.py, kmedoids.py, persistence.py
│   ├── evaluation/                 # metrics.py, visualization.py, profiling.py
│   ├── deployment/                 # predictor.py (suy luận 1 mẫu / hàng loạt & CLI)
│   └── utils/                      # logger.py
├── tests/                          # Unit tests và regression tests qua 5 file
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   └── test_deployment.py
├── main.py                         # Script chạy toàn bộ pipeline từ đầu đến cuối
├── requirements.txt
└── README.md
```

*(Ghi chú: Thư mục `docs/LAB3_CUSTOMER_CLUSTERING/` nếu có trên máy cục bộ chỉ là tài liệu tham khảo cũ, không liên quan đến mã nguồn và kết quả của pipeline chính.)*

### Hướng dẫn cài đặt và chạy lại thực nghiệm

1. **Kích hoạt môi trường ảo & cài thư viện** (từ thư mục gốc dự án):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

2. **Chạy pipeline chính, bộ kiểm thử đơn vị và thử nghiệm CLI:**

```powershell
$env:PYTHONUTF8 = "1"
$env:MPLBACKEND = "Agg"
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.deployment.predictor --age 35 --income 75000 --spending 1400
```

- Khi chạy `main.py`, toàn bộ dữ liệu xử lý và kết quả được xuất vào `data/processed/` và `results/` (`figures/`, `metrics/`, `saved_models/`).
- `PYTHONUTF8=1` giúp stdout/subprocess dùng UTF-8 trên Windows; `MPLBACKEND=Agg` chạy biểu đồ headless không cần Tk.
- Main và notebook cùng chặn thiếu cột bắt buộc hoặc thiếu ID; bản ghi trùng ID được giữ bản đầu tiên trước khi chia Train/Test.
- Single/batch inference kiểm tra miền `0 < Age <= 120`, `Income >= 0`, `Total_Spending >= 0` và tính hữu hạn. Batch báo lỗi theo index, không âm thầm bỏ dòng sai. Đây là miền input CLI demo; ngưỡng lọc outlier khi Train không phải giới hạn cứng cho khách mới.
- Batch chỉ điền khuyết bằng median Train đã lưu; thiếu median hợp lệ phải báo lỗi, không lấy median của batch. Nếu một khoản `Mnt*` bị khuyết, tổng chi tiêu được coi là khuyết và dùng median Train của tổng, không tự coi khoản thiếu là 0.
- Khi mở và chạy tuần tự `notebooks/01_eda.ipynb` → `04_evaluation_and_profiling.ipynb` bằng kernel `.venv`, mọi kết quả sẽ được lưu tách riêng vào `notebooks/results/` để hai luồng thực nghiệm không ghi đè lẫn nhau.
