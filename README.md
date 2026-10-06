# DỰ ÁN: PHÂN CỤM KHÁCH HÀNG (CUSTOMER CLUSTERING)

Dự án Machine Learning phân cụm khách hàng bán lẻ (Retail Customer Segmentation) dựa trên các đặc trưng nhân khẩu học và hành vi tiêu dùng (`Age`, `Gender`, `Annual Income`, `Spending Score`) theo đúng yêu cầu của **Practice Exercise 3**.

---

## 1. Cấu Trúc Thư Mục Dự Án (Project Layout)

Cấu trúc thư mục được thiết kế chuẩn mực theo mô hình Module hóa (Modular Codebase):

```text
lab_3.1/
├── __pycache__/
├── .venv/                      # Môi trường ảo Python
├── data/
│   ├── raw/
│   │   └── Mall_Customers.csv  # Dữ liệu khách hàng thô
│   └── processed/              # Dữ liệu đã phân chia (train.csv, test.csv)
├── docs/
│   ├── LAB3_CUSTOMER_CLUSTERING/ # Code và tài liệu mẫu tham khảo
│   ├── Practice Exercise 3.docx
│   └── Practice Exercise 3.pdf
├── notebooks/                  # Sổ tay Jupyter thử nghiệm tương tác
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_and_scaling.ipynb
│   ├── 03_clustering_models.ipynb
│   └── 04_evaluation_and_profiling.ipynb
├── results/                    # Lưu trữ kết quả đầu ra
│   ├── figures/                # Biểu đồ trực quan (.png)
│   ├── metrics/                # Bảng số liệu, báo cáo đánh giá (.csv, .log)
│   └── saved_models/           # Mô hình đã đóng gói (.pkl)
├── src/                        # Mã nguồn module hóa
│   ├── __init__.py
│   ├── config.py               # Quản lý đường dẫn, tham số, bảng màu
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # Nạp, kiểm tra schema, chia tập train/test
│   │   └── preprocessing.py    # Kiểm tra null và rà soát ngoại lai (IQR)
│   ├── features/
│   │   ├── __init__.py
│   │   └── scaling.py          # StandardScaler, MinMaxScaler, RobustScaler
│   ├── models/
│   │   ├── __init__.py
│   │   ├── kmeans.py           # K-Means, Elbow method, Silhouette sweep
│   │   ├── hierarchical.py     # Agglomerative clustering, Linkage, Dendrogram
│   │   ├── dbscan.py           # DBSCAN, k-NN distance graph, phát hiện nhiễu
│   │   ├── kmedoids.py         # K-Medoids (PAM), tâm mẫu thật kháng ngoại lai
│   │   └── persistence.py      # Lưu và nạp mô hình bằng joblib
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py          # Silhouette, Davies-Bouldin, Calinski-Harabasz
│   │   ├── visualization.py    # Biểu đồ 2D/3D scatter, Dendrogram, Elbow plot
│   │   └── profiling.py        # Thống kê phân khúc, chân dung 5 Personas
│   ├── deployment/
│   │   ├── __init__.py
│   │   └── predictor.py        # Dự đoán phân khúc cho khách hàng mới
│   └── utils/
│       ├── __init__.py
│       └── logger.py           # Tiện ích ghi log ra Console và File
├── tests/                      # Bộ kiểm thử tự động (Unit Tests)
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   └── test_deployment.py
├── .gitignore
├── main.py                     # File điều phối chạy toàn bộ pipeline
├── README.md
└── requirements.txt
```

---

## 2. Quy Trình Thực Hiện Theo Yêu Cầu Đề Bài (Practice Exercise 3)

### Bước 1: Khám Phá Dữ Liệu (Data Exploration - EDA)
- **Kiểm tra thiếu dữ liệu (Missing Values)**: Rà soát toàn bộ các cột, đảm bảo tính toàn vẹn 100%.
- **Phát hiện ngoại lai (Outliers)**: Áp dụng phương pháp Khoảng tứ phân vị (**IQR Method - Tukey's Fences**):
  $$\text{IQR} = Q3 - Q1, \quad [\text{Ngưỡng dưới} = Q1 - 1.5 \times \text{IQR}, \quad \text{Ngưỡng trên} = Q3 + 1.5 \times \text{IQR}]$$
- **Trực quan hóa hình thái phân phối**:
  - Biểu đồ **Histograms** kèm đường cong mật độ **KDE** cho `Age`, `Annual Income`, `Spending Score`.
  - Biểu đồ hộp **Boxplots** phát hiện trực quan các điểm thu nhập ngoại lai.
  - Biểu đồ phân bổ giới tính **Gender Distribution**.
- **Nhận diện cấu trúc cụm**:
  - Biểu đồ phân tán song biến (**Scatter Plots**): Đặc biệt cặp `Annual Income (k$)` vs `Spending Score (1-100)` bộc lộ rõ rệt 5 vùng mật độ tự nhiên.
  - Ma trận hệ số tương quan **Pearson Heatmap**.

### Bước 2: Chuẩn Hóa Đặc Trưng (Feature Scaling)
- **Vấn đề**: Thu nhập ($15 - 137$) có biên độ chênh lệch rất lớn so với Điểm chi tiêu ($1 - 99$) và Tuổi ($18 - 70$). Nếu không chuẩn hóa, thuật toán đo khoảng cách Euclidean sẽ bị chi phối bởi thuộc tính thu nhập.
- **Giải pháp**:
  - **StandardScaler**: Đưa dữ liệu về $Mean = 0, Std = 1$.
  - **MinMaxScaler**: Co giãn dữ liệu về đoạn $[0, 1]$.
  - **RobustScaler**: Chuẩn hóa dựa trên Median và IQR để kháng ngoại lai.
- **Nguyên tắc chống rò rỉ (No Data Leakage)**: `fit` trên tập Train và `transform` trên tập Test / dữ liệu mới.

### Bước 3: Thử Nghiệm Các Thuật Toán Phân Cụm (Clustering Experiments)
Hệ thống cung cấp đầy đủ 4 thuật toán theo yêu cầu:
1. **K-Means**:
   - Áp dụng kỹ thuật khởi tạo tối ưu `k-means++`.
   - Quét $K = 2 \dots 10$ để vẽ biểu đồ **Elbow Method (Inertia/WCSS)** và **Silhouette Score** nhằm chọn số cụm tối ưu ($K=5$).
2. **Hierarchical Clustering (Agglomerative)**:
   - Tính ma trận liên kết với các tiêu chí: `ward`, `complete`, `average`.
   - Vẽ biểu đồ nhánh **Dendrogram** và đánh giá hệ số tương quan Cophenetic.
3. **DBSCAN (Mật độ)**:
   - Dùng đồ thị khoảng cách **k-Nearest Neighbors (k-NN)** để chọn bán kính lân cận $\varepsilon$ (eps).
   - Quét lưới tham số (`eps`, `min_samples`) và tự động tách biệt các điểm nhiễu (Noise label = -1).
4. **K-Medoids (PAM - Partitioning Around Medoids)**:
   - Chọn chính các điểm dữ liệu thật trong tập làm tâm đại diện (Medoids).
   - Kháng ngoại lai vượt trội hơn K-Means khi dữ liệu có phân phối lệch.

### Bước 4: Đánh Giá Chất Lượng & Xây Dựng Chân Dung Khách Hàng (Evaluation & Profiling)
- **Chỉ số đánh giá nội tại**:
  - **Silhouette Coefficient**: Đo lường sự tương đồng nội cụm so với sự tách biệt ngoại cụm.
  - **Davies-Bouldin Index**: Đo độ trùng lặp hình học (càng nhỏ càng tốt).
  - **Calinski-Harabasz Score**: Tỷ số phân tán giữa các cụm / trong cụm (càng lớn càng tốt).
  - **Tỷ lệ điểm có Silhouette âm**: Đánh giá số mẫu có nguy cơ bị gán nhầm cụm.
- **Trực quan hóa phân khúc**:
  - Biểu đồ tán xạ 2D kèm tọa độ tâm cụm (Centroids / Medoids).
  - Biểu đồ phân cụm không gian 3 chiều (3D Scatter Plot).
  - Biểu đồ hình lưỡi dao Silhouette Knife Plot cho từng cụm.
- **5 Chân dung Persona kinh doanh thực tiễn**:
  1. **Khách hàng VIP / Ngôi sao**: Thu nhập cao, Chi tiêu cao $\rightarrow$ Dịch vụ chăm sóc cao cấp, đặc quyền VIP.
  2. **Khách hàng Cẩn trọng**: Thu nhập cao, Chi tiêu thấp $\rightarrow$ Giới thiệu sản phẩm giá trị đầu tư cao, chính sách bảo hành.
  3. **Khách hàng Trẻ / Chi tiêu bốc đồng**: Thu nhập thấp, Chi tiêu cao $\rightarrow$ Flash sale, sản phẩm bắt trend, trả góp 0%.
  4. **Khách hàng Tiết kiệm / Bình dân**: Thu nhập thấp, Chi tiêu thấp $\rightarrow$ Sản phẩm combo, giảm giá xả kho, tích điểm đổi quà.
  5. **Khách hàng Phổ thông / Tiêu chuẩn**: Thu nhập trung bình, Chi tiêu trung bình $\rightarrow$ Duy trì tần suất mua sắm bằng voucher định kỳ.

---

## 3. Hướng Dẫn Cài Đặt & Sử Dụng

### Kích hoạt môi trường ảo:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Cài đặt thư viện:
```bash
pip install -r requirements.txt
```

### Chạy toàn bộ pipeline tự động:
```bash
python main.py
```
Toàn bộ biểu đồ phân tích sẽ được xuất ra `results/figures/`, báo cáo số liệu lưu ở `results/metrics/`, và mô hình được đóng gói tại `results/saved_models/`.
