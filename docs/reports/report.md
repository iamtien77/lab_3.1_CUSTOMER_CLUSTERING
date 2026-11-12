# Báo cáo Practice Exercise 3 — Phân cụm khách hàng (Customer Clustering)

Trạng thái: **Đã hoàn tất thực nghiệm và kiểm chứng toàn pipeline trên dữ liệu thật** (`data/raw/marketing_campaign.csv`, `RANDOM_SEED = 42`, chia `80% Train / 20% Holdout Test`).

---

## 1. Dữ liệu và khám phá (EDA)

- **Nguồn dữ liệu:** `data/raw/marketing_campaign.csv` (định dạng phân tách bằng ký tự tab `\t`, gồm **2.240 dòng x 29 cột**).
- **Điều chỉnh đặc trưng cho bài lab:** Tập gốc không có sẵn cột `Age`, `Gender` và `Spending Score (1-100)`. Theo thống nhất thiết kế:
  - `Age = 2014 - Year_Birth` (lấy `REFERENCE_YEAR = 2014` theo năm tối đa của cột `Dt_Customer`).
  - `Total_Spending` = tổng 6 khoản chi tiêu `MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts + MntGoldProds`.
  - Dùng các cột nhân khẩu học thật (`Education`, `Marital_Status`) khi mô tả hồ sơ phân khúc; **không** tự tạo giả cột `Gender` và không dùng `Response` làm nhãn học.
- **Kiểm toán dữ liệu thô (`../../results/metrics/raw_data_audit.csv`):**
  - `ID` duy nhất hoàn toàn (0 bản ghi trùng, 0 ID khuyết).
  - Thiếu dữ liệu: chỉ có **24 giá trị khuyết (`1.07%`) tại cột `Income`**.
  - Ngoại lai cực đoan (lỗi miền dữ liệu): phát hiện **3 bản ghi `Year_Birth < 1920`** (`1893, 1899, 1900` tương ứng tuổi 114–121) và **1 bản ghi `Income = 666,666.0`** (cao gấp 4 lần mức cao kế tiếp `162,397.0`).
  - Giá trị hạng mục bất thường ở `Marital_Status`: xuất hiện nhãn hiếm/tự do **`Alone` (`3`), `Absurd` (`2`), `YOLO` (`2`)** — được giữ lại để phản ánh dữ liệu thực tế do chỉ dùng ở khâu mô tả hồ sơ sau phân cụm.
  - Mốc thời gian gia nhập `Dt_Customer` trải dài **`2012–2014`**, khớp giả định `REFERENCE_YEAR = 2014`.
- **Thống kê phân phối tập Train (sau khi lọc 4 bản ghi dị biệt cực đoan, chia 80/20 -> Train = `1.788` mẫu, Test = `448` mẫu; tồn tại `136` bản ghi trùng hoàn toàn trên 3 chiều `[Age, Income, Total_Spending]` phản ánh khách hàng thực có cùng nhân khẩu & tổng chi):**
  - `Age`: Mean = `45.23`, Median = `44.0`, Std = `11.85`, Min–Max = `18.0–74.0`, Skew = `0.09`.
  - `Income`: Mean = `51,975.56`, Median = `51,650.5`, Std = `21,567.02`, Min–Max = `2,447.0–162,397.0`, Skew = `0.41` (7 mẫu IQR outliers trên `117,654.75`, chiếm `0.39%`).
  - `Total_Spending`: Mean = `605.60`, Median = `397.5`, Std = `598.82`, Min–Max = `5.0–2,524.0`, Skew = `0.86` (lệch phải mạnh).
- **Biểu đồ EDA:**
  - Phân phối & Boxplot: [`00a_eda_distributions.png`](../../results/figures/00a_eda_distributions.png)
  - Tương quan song biến & Học vấn: [`00b_eda_correlations.png`](../../results/figures/00b_eda_correlations.png)

---

## 2. Tiền xử lý và chuẩn hóa

- **Chống Data Leakage:** Tập `2.236` bản ghi hợp lệ được chia thành `1.788` Train (`80%`) và `448` Holdout Test (`20%`) **trước khi** điền khuyết và chuẩn hóa. Trung vị `Income = 51,650.50` tính duy nhất từ tập Train được dùng để điền 18 ô thiếu của Train và 6 ô thiếu của Test.
- **Tham số `StandardScaler` học duy nhất trên Train (`../../results/metrics/scaler_params.csv`):**
  - `Age`: $\mu = 45.2299$, $\sigma = 11.8506$ (biên Train `18.0–74.0`).
  - `Income`: $\mu = 51,975.5587$, $\sigma = 21,560.9848$ (biên Train `2,447.0–162,397.0`).
  - `Total_Spending`: $\mu = 605.5990$, $\sigma = 598.6500$ (biên Train `5.0–2,524.0`).
- **So sánh phân phối 4 thang đo trên Train (`../../results/metrics/scaler_comparison.csv`):**
  - Khoảng giá trị gốc giữa `Income` (`Std = 21,560.98`) và `Age` (`Std = 11.85`) chênh lệch gần ~2.000 lần; nếu không chuẩn hóa, khoảng cách Euclidean sẽ bị `Income` chi phối tuyệt đối.
  - `StandardScaler` đưa cả 3 đặc trưng về `Mean = 0.0000`, `Std = 1.0000` (IQR tương đương `1.5189–1.6278`), giữ nguyên độ lệch phân phối (`Skew`: `0.0950`, `0.4129`, `0.8565`).
  - `MinMaxScaler` đưa Train về `[0, 1]`, nhưng do đuôi phải `Income` tới `162,397` và `Total_Spending` lệch phải (`Skew = 0.8565`), phần lớn mẫu bị nén lệch về vùng thấp (Median `Income = 0.3076`, Median `Total_Spending = 0.1558`, `Std Income` chỉ `0.1348` nhỏ hơn `Total_Spending Std = 0.2377` -> biến có độ trải hẹp lại bị thu nhỏ trọng số Euclidean tương đối).
  - `RobustScaler` đưa `Median = 0.0000`, `IQR = 1.0000` để giảm ảnh hưởng ngoại lai, nhưng giữ `Std ≈ 0.6143–0.6584`.
- **Hậu kiểm Train/Holdout Test & độ nhạy biên (`../../results/metrics/scaler_verification.csv`):**
  - `StandardScaler` đạt chuẩn hậu kiểm `Mean = 0.0, Std = 1.0` trên Train và ổn định trên Holdout Test (`Mean` từ `-0.0539` đến `+0.0032`; `Std` từ `0.9339` đến `1.0254`).
  - Ngược lại, `MinMaxScaler` fit trên Train làm **3 mẫu Holdout Test vượt đoạn `[0, 1]`**: `1` mẫu `Income` âm (`Min = -0.0045` dưới cận Train `2,447.0`) và `2` mẫu `Total_Spending > 1` (`Max = 1.0004` vượt cận Train `2,524.0`).
- **Đối chiếu tác động lên phân cụm K-Means (`../../results/metrics/scaler_clustering_comparison.csv`):**

| Scaler | $K$ tối ưu | Silhouette Train | DBI Train | CHI Train | Cân bằng cụm ($K=2$) | Silhouette ($K=5$) | Silhouette Test |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`standard` (Chọn)** | 2 | 0.4119 | **0.9892** | **1,495.38** | 0.7444 | 0.3527 | 0.4453 |
| `minmax` | 2 | **0.4210** | 0.9963 | 1,463.04 | 0.6602 | **0.3812** | **0.4561** |
| `robust` | 2 | 0.4026 | 1.0127 | 1,422.45 | **0.7598** | 0.3528 | 0.4368 |

  - **Quyết định kỹ thuật:** `MinMaxScaler` nhỉnh hơn ở Silhouette (`0.4210` vs `0.4119`) do thu hẹp ảnh hưởng trục `Income` (`Std = 0.1348` vs `0.2377` của chi tiêu), nhưng làm tệ hơn **Davies-Bouldin (`0.9963` vs `0.9892`)**, giảm **Calinski-Harabasz (`1,463.04` vs `1,495.38`)**, giảm cân bằng cụm (`0.6602` vs `0.7444`) và nhạy biên `[0, 1]` với dữ liệu mới. **`StandardScaler`** đạt cân bằng tốt nhất giữa CHI, DBI, tỷ lệ cụm và độ ổn định khi suy luận ngoài tập Train.
  - Biểu đồ đối chiếu 4 trạng thái thang đo: [`00c_scaling_comparison.png`](../../results/figures/00c_scaling_comparison.png)

---

## 3. Thuật toán và thử nghiệm tham số

Toàn bộ mô hình chạy trên cùng không gian `StandardScaler` (`1.788` mẫu x `3` đặc trưng, fit chỉ trên tập Train, `RANDOM_SEED = 42`):

1. **K-Means (Baseline chính — `../../results/metrics/kmeans_k_search.csv`, `../../results/metrics/kmeans_hyperparams.csv`):**
   - Quét $K \in [2, 10]$ với `init='k-means++'`, `n_init=10`, `max_iter=300`:
     - $K=2$: Inertia = `2,919.54`, Silhouette = **`0.4119`**, DBI = `0.9892`, CHI = `1,495.38`.
     - $K=3$: Inertia = `2,124.77`, Silhouette = `0.3662`, DBI = `0.9606`, CHI = `1,360.63`.
     - $K=4$: Inertia = `1,639.14`, Silhouette = `0.3605`.
     - $K=5$: Inertia = `1,384.52`, Silhouette = `0.3527`.
   - Với `K=2`, thử nghiệm `init ∈ {k-means++, random}`, `n_init ∈ {10, 20}`, `max_iter ∈ {100, 300}` đều đạt cùng nghiệm tối ưu (`Silhouette = 0.4119`), riêng `k-means++` hội tụ nhanh hơn (`8` vòng so với `9` vòng).
   - Biểu đồ Elbow & Silhouette: [`01_elbow_and_silhouette.png`](../../results/figures/01_elbow_and_silhouette.png)

2. **Hierarchical Clustering (`../../results/metrics/hierarchical_cophenetic.csv`, `../../results/metrics/hierarchical_k_search.csv`):**
   - Đối chiếu 3 kiểu linkage tại $K=2$: `average` đạt `Cophenetic = 0.6940` (`Silhouette = 0.5701`) và `complete` đạt `0.5958` (`Silhouette = 0.5701`), nhưng cả hai đều rơi vào **hiệu ứng xích (chaining)**: gom gần như toàn bộ khách hàng vào 1 cụm lớn và tách đúng 8 điểm ngoại lai thu nhập/chi tiêu cao (`{0: 1,780, 1: 8}`, độ cân bằng chỉ **`0.0045`**).
   - Ngược lại, **`ward`** (`Cophenetic = 0.6137`) tối thiểu hóa phương sai nội cụm nên tạo phân tách cân đối (`{0: 1,049, 1: 739}`, độ cân bằng **`0.7045`**) — phù hợp làm mô hình đối chứng chính.
   - Quét $K \in [2, 10]$ với `AgglomerativeClustering(linkage='ward', metric='euclidean')` cũng cho đỉnh Silhouette ở **$K=2$ (`0.3948`)**, giảm dần tại $K=3$ (`0.3333`), $K=4$ (`0.3206`), $K=5$ (`0.3088`).
   - Dendrogram kèm đường cắt tại $K=2$: [`04_hierarchical_dendrogram.png`](../../results/figures/04_hierarchical_dendrogram.png)

3. **DBSCAN (`../../results/metrics/dbscan_grid_search.csv`):**
   - Quét lưới `eps ∈ [0.3..0.8]` và `min_samples ∈ {4, 5, 8, 10}`. Các cấu hình có `eps >= 0.6` (hoặc `eps=0.5, ms>=5`) gộp toàn bộ vùng chính thành 1 cụm duy nhất nên không tính được Silhouette (trả về `NaN`, không gán giả `0`).
   - Cấu hình tốt nhất trong lưới là `eps = 0.5, min_samples = 4`: sinh **3 cụm** (`1,756 / 5 / 3` khách hàng), tách **24 điểm nhiễu (`1.34%`)**, `Silhouette (non-noise) = 0.3596`, nhưng độ cân bằng cụm cực thấp (`0.0017` — một cụm khổng lồ và 2 vi cụm rìa). Cấu hình xếp nhì `eps = 0.4, min_samples = 10` cho 2 cụm (`Silhouette = 0.3233`, `132` nhiễu `7.38%`).
   - Đồ thị k-NN ($k=4$, đã loại tự khoảng cách $0$) & cụm DBSCAN: [`06_dbscan_k_distance.png`](../../results/figures/06_dbscan_k_distance.png), [`02b_dbscan_clusters_2d.png`](../../results/figures/02b_dbscan_clusters_2d.png)

4. **K-Medoids (`../../results/metrics/kmedoids_k_search.csv`, `../../results/metrics/kmedoids_metric_comparison.csv`):**
   - Tự cài thuật toán K-Medoids (pha khởi tạo `BUILD` tất định & cập nhật medoid nội cụm) bằng NumPy/SciPy, quét $K \in [2, 10]$ với `euclidean` cũng đạt đỉnh tại $K=2$ (`Silhouette = 0.4080`, `Total_Cost = 2,057.10`; so với $K=3$: `0.3663`, $K=5$: `0.3240`).
   - Đối chiếu hai thước đo tại $K=2$: `euclidean` (`Silhouette = 0.4080`, `DBI = 0.9918`, `CHI = 1,486.32`) nhỉnh hơn nhẹ so với `manhattan` (`Silhouette = 0.4033`, `DBI = 1.0028`, `CHI = 1,436.99`, `Total_Cost = 2,965.32`).

---

## 4. Đánh giá và trực quan hóa

**Bảng đối chiếu 4 thuật toán trên tập Train (`../../results/metrics/model_comparison.csv`):**

| Thuật toán | Số cụm K | Điểm nhiễu (%) | Silhouette Score | Silhouette âm (%) | Davies-Bouldin | Calinski-Harabasz | Dunn Index | Độ cân bằng cụm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **K-Means (K=2)** | 2 | 0 (0.00%) | **0.4119** | **1.68%** | 0.9892 | **1,495.38** | 0.0161 | 0.7444 |
| **K-Medoids (K=2)** | 2 | 0 (0.00%) | 0.4080 | 2.80% | 0.9918 | 1,486.32 | 0.0106 | **0.8079** |
| **Hierarchical Ward (K=2)** | 2 | 0 (0.00%) | 0.3948 | 5.03% | 1.0195 | 1,387.73 | 0.0200 | 0.7045 |
| **DBSCAN (eps=0.5, ms=4)** | 3 | 24 (1.34%) | 0.3596 | 14.51% | 0.5214 | 19.06 | 0.0704 | 0.0017 |

**Kiểm chứng độ ổn định phân bố cụm trên tập Holdout Test (`448` khách hàng độc lập, `../../results/metrics/holdout_test_evaluation.csv`):**
- `K-Means Test Holdout (K=2)`: `Silhouette = 0.4453`, `Davies-Bouldin = 0.8977`, `Calinski-Harabasz = 455.16`, tỷ lệ Silhouette âm `2.01%`.
- `K-Medoids Test Holdout (K=2)`: `Silhouette = 0.4418`, `Davies-Bouldin = 0.9023`, `Calinski-Harabasz = 451.75`, tỷ lệ Silhouette âm `2.68%`.
=> Tâm cụm và ranh giới phân tách giữ được độ ổn định nhất quán trên dữ liệu chưa từng thấy *(lưu ý theo thiết kế: không dùng chỉ số phân cụm nội tại trên tập Holdout để kết luận tính chất generalization/overfitting như bài toán có giám sát)*.

**Trực quan hóa mô hình K-Means được chọn:**
- Biểu đồ 2D (Thu nhập vs Chi tiêu + Tâm cụm): [`02_clusters_2d.png`](../../results/figures/02_clusters_2d.png)
- Biểu đồ 3D (Tuổi x Thu nhập x Chi tiêu): [`03_clusters_3d.png`](../../results/figures/03_clusters_3d.png)
- Phân tích Silhouette Knife Plot: [`05_silhouette_analysis.png`](../../results/figures/05_silhouette_analysis.png)

---

## 5. Các phân khúc và kết luận

**Hồ sơ thống kê thực tế từ mô hình K-Means ($K=2$) (`../../results/metrics/cluster_profiles.csv`):**

- **Cụm 0 — Khách hàng VIP / Giá trị cao (Trung niên/Cao tuổi) (`763` người, `42.67%` tập Train):**
  - Độ tuổi TB: `47.64` (trung vị `48.0`), Thu nhập TB: `$71,704.57` (trung vị `$70,647.0`), Chi tiêu TB: `$1,191.48` (trung vị `$1,150.0`).
  - Nhân khẩu học & hành vi nổi bật: `Graduation (52.9%)`, `Married (37.4%)`, danh mục chi chính: `MntWines (TB: 605.6)`.
  - Gợi ý tiếp thị: Ưu tiên chăm sóc hội viên VIP, truyền thông các danh mục chi tiêu chủ lực và cá nhân hóa ưu đãi duy trì mức chi tiêu cao.

- **Cụm 1 — Khách hàng Tiết kiệm / Nhạy cảm giá (Độ tuổi phổ thông) (`1,025` người, `57.33%` tập Train):**
  - Độ tuổi TB: `43.43` (trung vị `42.0`), Thu nhập TB: `$37,289.47` (trung vị `$37,401.0`), Chi tiêu TB: `$169.48` (trung vị `$85.0`).
  - Nhân khẩu học & hành vi nổi bật: `Graduation (49.3%)`, `Married (40.2%)`, danh mục chi chính: `MntWines (TB: 81.7)`.
  - Gợi ý tiếp thị: Áp dụng combo thiết yếu giá tốt, chương trình tích điểm đổi quà và khuyến mãi theo mùa.

**Kiểm chứng suy luận triển khai mức bài lab (CLI Predictor):**
- Thử nghiệm lệnh `python -m src.deployment.predictor --age 35 --income 75000 --spending 1400` sử dụng gói [`final_kmeans_pipeline.pkl`](../../results/saved_models/final_kmeans_pipeline.pkl) (kèm `feature_medians` tập Train): gán vào **Cụm 0 (`Khách hàng VIP / Giá trị cao (Trung niên/Cao tuổi)`)** với khoảng cách tới tâm cụm trong không gian chuẩn hóa là `1.1328`.

**Kết luận & Hạn chế thực tế:**
- Trên không gian 3 chiều `[Age, Income, Total_Spending]` của `marketing_campaign.csv`, cả 3 thuật toán dựa trên khoảng cách (`K-Means`, `K-Medoids`, `Hierarchical Ward`) đều hội tụ ở đỉnh Silhouette tại **$K=2$** (`0.4119` / `0.4080` / `0.3948`) thay vì giả định cố định `K=5`. Tương quan dương mạnh giữa `Income` và `Total_Spending` là yếu tố dẫn dắt chính ranh giới hai cụm.
- **Hạn chế:** Do `Total_Spending` lệch phải mạnh (`skew = 0.86`) và còn một số khách hàng thu nhập cao (`> $150,000`) nhưng chi tiêu rất thấp hoặc ngược lại đóng vai trò điểm biên (`1.68%` mẫu K-Means có Silhouette âm), có thể khảo sát thêm biến đổi `log1p(Total_Spending)` hoặc bổ sung hành vi theo kênh (`NumWebPurchases`, `NumStorePurchases`, `Recency`) ở các nghiên cứu tiếp theo.
