# DỰ ÁN: PHÂN CỤM KHÁCH HÀNG (CUSTOMER CLUSTERING)

Dự án Machine Learning phân cụm khách hàng bán lẻ (Retail Customer Segmentation) dựa trên thông tin nhân khẩu học và hành vi tiêu dùng (`Age`, `Income`, `Total_Spending` trích xuất từ `marketing_campaign.csv`) theo yêu cầu của **Practice Exercise 3**.

**Trạng thái hiện tại:** Đã hoàn thiện triển khai pipeline đầy đủ ở phạm vi bài lab học tập (nạp & làm sạch 80/20, chuẩn hóa đặc trưng chống rò rỉ dữ liệu, 4 thuật toán phân cụm với K-Means Baseline, đánh giá định lượng, trực quan hóa, profiling động, kiểm thử `unittest` và báo cáo thực nghiệm tại `docs/reports/report.md`).

---

## 1. Cấu Trúc Thư Mục Dự Án (Project Layout)

Cấu trúc được tổ chức theo mô hình module hóa (Modular Codebase), tách dữ liệu, mã nguồn, notebook, kết quả và báo cáo:

```text
lab_3.1_CUSTOMER_CLUSTERING/
├── .venv/                          # Môi trường ảo Python cục bộ
├── data/
│   ├── raw/
│   │   └── marketing_campaign.csv   # Dữ liệu khách hàng gốc (TSV, 2.240 dòng x 29 cột)
│   └── processed/                  # Dữ liệu sau tiền xử lý (train.csv, test.csv, full_customers.csv)
├── docs/
│   ├── Algrothirms/                 # Thư mục tài liệu thuật toán
│   ├── description_working_with_AI/ # Thư mục tài liệu làm việc với AI
│   ├── reports/
│   │   └── report.md               # Khung báo cáo phân tích và kết luận
│   └── Practice Exercise 3.pdf      # Đề bài gốc
├── notebooks/                      # Sổ tay Jupyter theo bốn bước của đề
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_and_scaling.ipynb
│   ├── 03_clustering_models.ipynb
│   └── 04_evaluation_and_profiling.ipynb
├── results/                        # Nơi lưu kết quả được sinh khi chạy
│   ├── figures/                    # Biểu đồ trực quan
│   ├── metrics/                    # Tham số thử nghiệm và bảng đánh giá
│   └── saved_models/               # Mô hình đã lưu (mở rộng)
├── src/                            # Khung mã nguồn module hóa
│   ├── __init__.py
│   ├── config.py                   # Đường dẫn, tham số và cấu hình hiển thị
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py               # Nạp, kiểm tra schema; chia tập nếu cần
│   │   └── preprocessing.py        # Kiểm tra dữ liệu thiếu và ngoại lai
│   ├── features/
│   │   ├── __init__.py
│   │   └── scaling.py              # StandardScaler, MinMaxScaler, RobustScaler
│   ├── models/
│   │   ├── __init__.py
│   │   ├── kmeans.py               # K-Means, Elbow và khảo sát số cụm
│   │   ├── hierarchical.py         # Agglomerative, Linkage và Dendrogram
│   │   ├── dbscan.py               # DBSCAN, khảo sát eps và min_samples
│   │   ├── kmedoids.py             # K-Medoids, tâm cụm là mẫu dữ liệu thật
│   │   └── persistence.py          # Lưu và nạp mô hình (mở rộng)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # Silhouette và các chỉ số bổ sung
│   │   ├── visualization.py        # Scatter, Dendrogram và biểu đồ đánh giá
│   │   └── profiling.py            # Thống kê và mô tả các phân khúc
│   ├── deployment/
│   │   ├── __init__.py
│   │   └── predictor.py            # Gán cụm khách hàng mới (mở rộng)
│   └── utils/
│       ├── __init__.py
│       └── logger.py               # Tiện ích ghi log
├── tests/                          # Khung kiểm thử phục vụ triển khai
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   └── test_deployment.py
├── .gitignore
├── main.py                         # Khung điều phối pipeline
├── README.md
└── requirements.txt
```

`results/` dành cho đầu ra được sinh từ chương trình; `docs/reports/` dành cho phần giải thích và nhận xét. Tài liệu trong `docs/LAB3_CUSTOMER_CLUSTERING/` dùng để tham khảo, không đại diện cho kết quả của pipeline hiện tại.

Xem [bảng đối chiếu yêu cầu](docs/requirements_mapping.md) để biết trách nhiệm từng phần.

---

## 2. Quy Trình Thực Hiện Theo Yêu Cầu Đề Bài (Practice Exercise 3)

### Bước 1: Khám Phá Dữ Liệu (Data Exploration - EDA)

- **Kiểm tra thiếu dữ liệu (Missing Values):** thống kê số lượng và tỷ lệ thiếu theo cột, đề xuất cách xử lý.
- **Phát hiện ngoại lai (Outliers):** có thể sử dụng Boxplots và khoảng tứ phân vị (IQR); đánh giá trước khi quyết định giữ hoặc loại bỏ mẫu.
- **Phân tích phân phối (Distributions):** vẽ Histograms và KDE/Boxplot cho `Age` (`2014 - Year_Birth`), `Income` và `Total_Spending` (`sum` 6 cột `Mnt*`); có thể bổ sung thống kê nhân khẩu học thật (`Education`, `Marital_Status`).
- **Quan sát quan hệ giữa các đặc trưng:** vẽ Scatter Plots giữa `Income–Total_Spending` và `Age–Total_Spending`, kèm ma trận tương quan để quan sát cấu trúc phân nhóm; không ấn định trước số cụm.

Với phương pháp IQR, ngưỡng phát hiện ngoại lai thường dùng là:

$$
IQR = Q_3 - Q_1, \qquad [Q_1 - 1.5 \times IQR,\; Q_3 + 1.5 \times IQR]
$$

**Nơi thực hiện:** `notebooks/01_eda.ipynb`, kết hợp các module trong `src/data/` và `src/evaluation/visualization.py`.

### Bước 2: Chuẩn Hóa Đặc Trưng (Feature Scaling)

**Mục tiêu:** đưa 3 đặc trưng số `Age`, `Income` và `Total_Spending` về thang đo tương đương để hạn chế ảnh hưởng của khác biệt đơn vị và độ phân tán lên khoảng cách phân cụm.

Các phương án triển khai:

- **StandardScaler (mặc định chính):** chuẩn hóa theo trung bình và độ lệch chuẩn; với biến không hằng, dữ liệu dùng để fit có trung bình xấp xỉ 0 và phương sai 1.
- **MinMaxScaler (đối chiếu):** đưa các giá trị của dữ liệu dùng để fit về khoảng `[0, 1]`.
- **RobustScaler — mở rộng:** sử dụng median và IQR để giảm ảnh hưởng của giá trị ngoại lai lên tham số chuẩn hóa.

`ID` chỉ dùng làm khóa định danh và loại khỏi vector phân cụm. Chia cố định 80% Train / 20% Holdout Test (`RANDOM_SEED=42`): chỉ tính trung vị điền khuyết `Income` và `fit` scaler trên tập Train, sau đó `transform` tập Train và tập Test để chống Data Leakage.

**Nơi thực hiện:** `notebooks/02_preprocessing_and_scaling.ipynb` và `src/features/scaling.py`.

### Bước 3: Thử Nghiệm Các Thuật Toán Phân Cụm (Clustering Experiments)

Đề yêu cầu **chọn thuật toán, thử các tham số và đánh giá kết quả**. Bốn thuật toán dưới đây là các lựa chọn được chuẩn bị trong cấu trúc; không bắt buộc triển khai cả bốn.

1. **K-Means:** có thể dùng làm baseline; khảo sát số cụm, chẳng hạn $K = 2, \ldots, 10$, kết hợp Elbow và Silhouette để chọn cấu hình phù hợp. Không mặc định $K=5$.
2. **Hierarchical Clustering (Agglomerative):** thử số cụm và các kiểu linkage như `ward`, `complete`, `average`; sử dụng Dendrogram để quan sát cấu trúc phân cấp.
3. **DBSCAN:** khảo sát `eps` và `min_samples`; có thể dùng đồ thị khoảng cách k-NN để hỗ trợ chọn `eps`. Ghi nhận riêng các điểm nhiễu có nhãn `-1`.
4. **K-Medoids (BUILD tất định & cập nhật nội cụm):** chọn mẫu dữ liệu thật làm tâm đại diện; có thể dùng để so sánh với K-Means về chất lượng cụm và ảnh hưởng của ngoại lai.

Mỗi thử nghiệm cần ghi thuật toán, tập đặc trưng, scaler, tham số, seed nếu có và chỉ số đánh giá. Kết luận dựa trên kết quả thực tế.

**Nơi thực hiện:** `notebooks/03_clustering_models.ipynb` và `src/models/`.

### Bước 4: Đánh Giá Chất Lượng & Mô Tả Phân Khúc (Evaluation & Profiling)

**Chỉ số đánh giá:**

- **Silhouette Coefficient:** đánh giá độ gắn kết trong cụm và mức phân tách giữa các cụm; giá trị cao hơn thường tốt hơn.
- **Davies–Bouldin Index — bổ sung:** giá trị thấp hơn thường tốt hơn.
- **Calinski–Harabasz Score — bổ sung:** giá trị cao hơn thường tốt hơn.

Với DBSCAN, cần ghi số cụm, tỷ lệ noise và cách xử lý noise khi tính chỉ số. Nếu cấu hình không đủ điều kiện tính Silhouette, ghi rõ không tính được.

**Trực quan hóa phân khúc:**

- Scatter Plot 2D thể hiện các mẫu và nhãn cụm; có thể bổ sung tâm cụm khi thuật toán có tâm đại diện.
- Dendrogram cho phân cụm phân cấp.
- Scatter Plot 3D hoặc Silhouette Plot là các phần bổ sung khi giúp giải thích kết quả.

**Mô tả nhóm khách hàng:** thống kê tuổi, thu nhập và điểm chi tiêu theo cụm, sau đó diễn giải đặc điểm từng nhóm. Persona và gợi ý marketing là phần mở rộng, cần căn cứ vào dữ liệu; không ấn định trước năm nhóm khách hàng.

**Nơi thực hiện:** `notebooks/04_evaluation_and_profiling.ipynb` và `src/evaluation/`. Tổng hợp nhận xét trong [báo cáo](docs/reports/report.md).

---

## 3. Hướng Dẫn Cài Đặt & Sử Dụng

### Kích hoạt môi trường ảo

Từ thư mục gốc dự án, nếu đã có `.venv`:

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu chưa có môi trường ảo, tạo trước bằng:

```powershell
python -m venv .venv
```

### Cài đặt thư viện

Sau khi kích hoạt môi trường:

```powershell
python -m pip install -r requirements.txt
```

### Làm việc với notebook

Mở các notebook theo thứ tự `01` → `04` trong IDE có hỗ trợ Jupyter và chọn kernel của `.venv`. Mỗi notebook có thể chạy nối tiếp nhau (`01` tạo dữ liệu sạch, `02` chuẩn hóa, `03` huấn luyện & lưu mô hình, `04` đánh giá & profiling).

### Chạy toàn bộ pipeline, kiểm thử và CLI suy luận

```powershell
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.deployment.predictor --age 35 --income 75000 --spending 1400
```

Kết quả sinh ra được lưu tại:

- `data/processed/`: `train.csv`, `test.csv`, `full_customers.csv`.
- `results/figures/`: các biểu đồ EDA, Elbow & Silhouette, Scatter 2D/3D, Dendrogram, Silhouette Knife Plot, DBSCAN k-NN.
- `results/metrics/`: bảng kiểm toán dữ liệu, bảng quét tham số, `model_comparison.csv`, `holdout_test_evaluation.csv`, `cluster_profiles.csv`.
- `results/saved_models/`: gói mô hình `final_kmeans_pipeline.pkl` kèm scaler, trung vị Train và định nghĩa Persona.
- `docs/reports/report.md`: báo cáo tổng hợp số liệu thực nghiệm, thảo luận và kết luận.
