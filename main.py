"""
=============================================================================
DỰ ÁN: PHÂN CỤM KHÁCH HÀNG (CUSTOMER CLUSTERING PIPELINE)
TẬP TIN ĐIỀU PHỐI CHÍNH (MAIN PIPELINE ENTRYPOINT)
=============================================================================
Theo đúng yêu cầu của đề bài: Practice Exercise 3.pdf
Các bước thực hiện trong pipeline:
  - BƯỚC 1: Data Exploration (Khám phá dữ liệu, missing values, outliers, phân phối).
  - BƯỚC 2: Feature Scaling (Chuẩn hóa / Tỉ lệ hóa Age, Annual Income, Spending Score).
  - BƯỚC 3: Clustering (Thử nghiệm K-Means, Hierarchical, DBSCAN, K-Medoids với các tham số khác nhau).
  - BƯỚC 4: Evaluation (Đánh giá Silhouette, Davies-Bouldin, Dendrogram, Scatter 2D/3D, Personas).
  - BƯỚC 5: Deployment (Lưu mô hình và chạy thử nghiệm suy luận khách hàng mới).
=============================================================================
"""

import sys
from pathlib import Path

# Thêm thư mục gốc vào đường dẫn hệ thống để import trơn tru
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    print_section,
    RAW_DATA_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    SAVED_MODELS_DIR,
)
from src.utils.logger import get_logger

logger = get_logger("MainPipeline")


def step_0_data_ingestion():
    """
    Ý LÀM:
    1. Đọc dữ liệu từ data/raw/Mall_Customers.csv bằng load_raw_data().
    2. Kiểm tra tính toàn vẹn (validate_schema): đếm dòng, cột, null, ID trùng lặp.
    3. Phân chia tập dữ liệu 80% Train và 20% Holdout Test (split_data).
    4. Ghi log kết quả hoàn thành nạp dữ liệu.
    """
    pass


def step_1_data_exploration(train_df):
    """
    Ý LÀM (Đáp ứng Bước 1 - Data Exploration):
    1. Kiểm tra thống kê mô tả: Mean, Median, Std, Min, Max, Skewness.
    2. Phát hiện Outliers bằng phương pháp IQR (Tukey's Fences) cho các biến số.
    3. Trực quan hóa hình thái phân phối:
       - Vẽ Histograms kèm đường mật độ KDE.
       - Vẽ Boxplots để quan sát trực quan các điểm ngoại lai.
       - Vẽ Biểu đồ phân bổ giới tính (Gender distribution).
    4. Trực quan hóa tương quan & mối quan hệ song biến:
       - Scatter plot: Annual Income vs Spending Score (quan sát cụm tự nhiên).
       - Scatter plot: Age vs Spending Score.
       - Heatmap ma trận hệ số tương quan Pearson.
    5. Lưu toàn bộ biểu đồ vào thư mục results/figures/.
    """
    pass


def step_2_feature_scaling(train_df, test_df):
    """
    Ý LÀM (Đáp ứng Bước 2 - Feature Scaling):
    1. So sánh 2 kỹ thuật chuẩn hóa: StandardScaler vs MinMaxScaler.
    2. Khởi tạo đối tượng FeatureScaler(method='standard').
    3. Fit và transform trên tập Train (X_train_scaled).
    4. Transform trên tập Test (X_test_scaled) bằng tham số đã học từ Train (chống Data Leakage).
    5. Trực quan hóa phân phối trước và sau khi chuẩn hóa để kiểm chứng hiệu quả.
    6. Trả về X_train_scaled và X_test_scaled.
    """
    pass


def step_3_clustering_experiments(X_train_scaled):
    """
    Ý LÀM (Đáp ứng Bước 3 - Clustering Experiments):
    Thực hiện huấn luyện và thử nghiệm tham số với 4 thuật toán:
    1. K-Means:
       - Quét K từ 2 đến 10 để tính WCSS (Inertia) và Silhouette Score.
       - Vẽ biểu đồ Elbow Method để xác định số cụm K tối ưu (K=5).
       - Huấn luyện mô hình K-Means cuối cùng với init='k-means++', n_init=20.
    2. Hierarchical Clustering (Phân cụm phân cấp):
       - Tính ma trận liên kết (Linkage) theo phương pháp Ward, Complete, Average.
       - Tính hệ số tương quan Cophenetic để đánh giá mức độ bảo toàn khoảng cách.
       - Huấn luyện mô hình AgglomerativeClustering với K=5.
    3. DBSCAN (Phân cụm dựa trên mật độ):
       - Tính khoảng cách k-NN (k=4) để vẽ đồ thị xác định bán kính lân cận eps.
       - Thử nghiệm lưới tham số (Grid search) với eps và min_samples.
       - Nhận diện các điểm nhiễu ngoại lai (nhãn = -1).
    4. K-Medoids (PAM - Partitioning Around Medoids):
       - Huấn luyện thuật toán PAM với K=5.
       - Chọn chính các điểm dữ liệu thật làm tâm cụm (Medoids) giúp kháng ngoại lai.
    5. Trả về từ điển chứa các mô hình và nhãn phân cụm tương ứng.
    """
    pass


def step_4_evaluation_and_profiling(train_df, X_train_scaled, clustering_results):
    """
    Ý LÀM (Đáp ứng Bước 4 - Evaluation & Visualizations):
    1. Tính toán và lập bảng so sánh đối đầu toàn diện 4 thuật toán:
       - Silhouette Score (độ gắn kết nội cụm vs độ phân tách liên cụm).
       - Davies-Bouldin Index (thấp hơn là tốt hơn).
       - Calinski-Harabasz Score (cao hơn là tốt hơn).
       - Tỷ lệ điểm Silhouette âm (các mẫu bị xếp nhầm cụm).
       - Lưu bảng kết quả so sánh vào results/metrics/model_comparison.csv.
    2. Trực quan hóa các cụm:
       - Biểu đồ tán xạ 2D (Income vs Spending) với tọa độ tâm cụm.
       - Biểu đồ tán xạ 3D (Age x Income x Spending).
       - Biểu đồ cây phân cấp Dendrogram với đường cắt ngưỡng.
       - Biểu đồ phân tích Silhouette Knife Plot chi tiết cho từng cụm.
    3. Phân tích hồ sơ khách hàng & Chân dung Persona (Profiling):
       - Tính thống kê trung bình (Tuổi, Thu nhập, Chi tiêu) cho từng phân khúc.
       - Đặt tên Persona kinh doanh (VIP, Cẩn trọng, Trẻ tuổi bốc đồng, Tiết kiệm, Bình dân).
       - Đề xuất chiến lược tiếp thị Marketing và CRM hành động thực tế.
       - Lưu bảng hồ sơ vào results/metrics/cluster_profiles.csv.
    """
    pass


def step_5_save_and_inference(final_model, scaler):
    """
    Ý LÀM:
    1. Lưu toàn bộ mô hình tốt nhất (K-Means) và Scaler vào file results/saved_models/final_kmeans_pipeline.pkl.
    2. Khởi tạo đối tượng CustomerSegmentPredictor.
    3. Chạy thử nghiệm phân khúc cho 1 khách hàng mới (ví dụ: Age=28, Income=75k, Spending=85).
    4. In kết quả dự đoán (Cụm gán, Persona kinh doanh, Khoảng cách tới tâm cụm).
    """
    pass


def main():
    """
    Ý LÀM:
    Hàm điều phối chính: Gọi tuần tự từ Bước 0 đến Bước 5 theo đúng quy trình chuẩn Machine Learning.
    """
    print_section("BẮT ĐẦU PIPELINE PHÂN CỤM KHÁCH HÀNG (CUSTOMER CLUSTERING)", "Dựa theo yêu cầu đề bài: Practice Exercise 3.pdf")
    logger.info("Khởi chạy pipeline...")

    # Bước 0: Nạp & kiểm định dữ liệu
    step_0_data_ingestion()

    # Bước 1: Khám phá dữ liệu (EDA)
    step_1_data_exploration(None)

    # Bước 2: Chuẩn hóa đặc trưng
    step_2_feature_scaling(None, None)

    # Bước 3: Thử nghiệm phân cụm
    step_3_clustering_experiments(None)

    # Bước 4: Đánh giá & Phân tích hồ sơ
    step_4_evaluation_and_profiling(None, None, None)

    # Bước 5: Lưu trữ mô hình & Thử nghiệm suy luận
    step_5_save_and_inference(None, None)

    print_section("HOÀN TẤT PIPELINE THÀNH CÔNG!", "Toàn bộ kết quả, biểu đồ và mô hình đã được xuất ra thư mục results/")


if __name__ == "__main__":
    main()
