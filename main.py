"""
=============================================================================
DỰ ÁN: PHÂN CỤM KHÁCH HÀNG (CUSTOMER CLUSTERING PIPELINE)
TẬP TIN ĐIỀU PHỐI CHÍNH (MAIN PIPELINE ENTRYPOINT)
=============================================================================
Bám sát yêu cầu Practice Exercise 3.pdf trong phạm vi bài lab học tập:
  - Dữ liệu: marketing_campaign.csv (Age = 2014 - Year_Birth, Income, Total_Spending).
  - Chia 80% Train / 20% Holdout Test (chỉ fit median & scaler trên tập Train).
  - Bước 1: Data Exploration (EDA, Missing Values, IQR Outliers, Trực quan hóa).
  - Bước 2: Feature Scaling (StandardScaler chính; đối chiếu MinMaxScaler & RobustScaler; kiểm chứng hậu điều kiện và tác động lên K-Means).
  - Bước 3: Clustering (K-Means Baseline là trọng tâm; đối chiếu Hierarchical, DBSCAN, K-Medoids).
  - Bước 4: Evaluation & Profiling (Silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn, Personas).
  - Bước 5: Persistence & Inference (Lưu gói mô hình và dự đoán thử nghiệm trên khách hàng mới).
=============================================================================
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Dùng backend vẽ tĩnh không cửa sổ khi chạy CLI
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import (
    INCOME_COLUMN,
    K_SEARCH_RANGE,
    MAX_VALID_INCOME,
    METRICS_DIR,
    MIN_VALID_YEAR_BIRTH,
    NUMERICAL_FEATURES,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_PATH,
    REFERENCE_YEAR,
    YEAR_BIRTH_COLUMN,
    print_section,
)
from src.data.loader import load_raw_data, split_data, validate_schema
from src.data.preprocessing import (
    add_derived_features,
    audit_missing_and_outliers,
    clean_data,
    compute_train_medians,
    filter_invalid_records,
)
from src.deployment.predictor import CustomerSegmentPredictor
from src.evaluation.metrics import compare_all_models, evaluate_clustering
from src.evaluation.profiling import create_cluster_profiles, generate_customer_personas
from src.evaluation.visualization import (
    plot_clusters_2d,
    plot_clusters_3d,
    plot_dbscan_k_distance,
    plot_dendrogram,
    plot_eda_correlations,
    plot_eda_distributions,
    plot_elbow_and_silhouette,
    plot_scaling_comparison,
    plot_silhouette_analysis,
)
from src.features.scaling import (
    FeatureScaler,
    compare_scalers,
    evaluate_scalers_on_clustering,
)
from src.models.dbscan import DBSCANTrainer
from src.models.hierarchical import HierarchicalTrainer
from src.models.kmeans import KMeansTrainer
from src.models.kmedoids import KMedoidsTrainer
from src.models.persistence import save_model
from src.utils.logger import get_logger

logger = get_logger("MainPipeline")


def step_0_data_ingestion() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Bước 0: Nạp dữ liệu gốc, kiểm định schema, loại dị biệt miền và chia 80/20 không rò rỉ dữ liệu.
    """
    print_section("BƯỚC 0: NẠP, KIỂM ĐỊNH VÀ CHIA TẬP DỮ LIỆU (80/20)")
    raw_df = load_raw_data(RAW_DATA_PATH)
    # ID trùng được ghi nhận và loại ở bước lọc trước khi chia tập.
    schema_info = validate_schema(raw_df, allow_duplicate_ids=True)
    if not schema_info["is_valid"]:
        raise ValueError(f"Schema dữ liệu không hợp lệ: {schema_info}")

    # Kiểm toán bản ghi gốc trước khi lọc để ghi nhận đầy đủ giá trị thiếu/ngoại lai ban đầu
    raw_audit_df = audit_missing_and_outliers(raw_df)
    raw_audit_df.to_csv(METRICS_DIR / "raw_data_audit.csv", index=False)

    # Lọc các lỗi nhập liệu cực đoan (Year_Birth < 1920, Income > 200,000) và tạo đặc trưng từng dòng
    valid_df = filter_invalid_records(raw_df)
    valid_df = add_derived_features(valid_df)

    # Chia 80% Train / 20% Test trước khi tính trung vị điền khuyết Income
    train_raw, test_raw = split_data(valid_df, save_files=False)
    train_medians = compute_train_medians(train_raw)

    train_df = clean_data(train_raw, fill_medians=train_medians, drop_invalid_outliers=False)
    test_df = clean_data(test_raw, fill_medians=train_medians, drop_invalid_outliers=False)

    # Lưu bản sạch của Train / Test / Full vào data/processed/
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    pd.concat([train_df, test_df], ignore_index=True).to_csv(
        PROCESSED_DATA_DIR / "full_customers.csv",
        index=False,
    )
    logger.info(f"Hoàn tất Bước 0: Train={len(train_df)} dòng, Test={len(test_df)} dòng.")
    return raw_df, train_df, test_df


def step_1_data_exploration(
    train_df: pd.DataFrame,
    raw_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Bước 1: Phân tích khám phá dữ liệu (EDA), kiểm toán dị biệt bản thô, thống kê Train và xuất biểu đồ.
    """
    print_section("BƯỚC 1: KHÁM PHÁ DỮ LIỆU (EDA & PHÁT HIỆN NGOẠI LAI)")
    if raw_df is not None:
        income_null_count = int(raw_df[INCOME_COLUMN].isnull().sum()) if INCOME_COLUMN in raw_df.columns else 0
        invalid_birth_count = (
            int((pd.to_numeric(raw_df[YEAR_BIRTH_COLUMN], errors="coerce") < MIN_VALID_YEAR_BIRTH).sum())
            if YEAR_BIRTH_COLUMN in raw_df.columns
            else 0
        )
        invalid_income_count = (
            int((pd.to_numeric(raw_df[INCOME_COLUMN], errors="coerce") > MAX_VALID_INCOME).sum())
            if INCOME_COLUMN in raw_df.columns
            else 0
        )
        unusual_marital_counts: Dict[str, int] = {}
        if "Marital_Status" in raw_df.columns:
            marital_counts = raw_df["Marital_Status"].value_counts()
            for unusual_label in ("Alone", "Absurd", "YOLO"):
                if unusual_label in marital_counts:
                    unusual_marital_counts[unusual_label] = int(marital_counts[unusual_label])
        print(" [KIỂM TOÁN DỮ LIỆU THÔ TRƯỚC LÀM SẠCH]")
        print(
            f"  - Tổng bản ghi thô: {len(raw_df):,} | Thiếu {INCOME_COLUMN}: {income_null_count} | "
            f"{YEAR_BIRTH_COLUMN} < {MIN_VALID_YEAR_BIRTH}: {invalid_birth_count} | "
            f"{INCOME_COLUMN} > {MAX_VALID_INCOME:,}: {invalid_income_count}"
        )
        if unusual_marital_counts:
            print(f"  - Nhãn Marital_Status bất thường (giữ lại làm biến mô tả): {unusual_marital_counts}")
        if "Dt_Customer" in raw_df.columns:
            enrollment_years = pd.to_datetime(raw_df["Dt_Customer"], format="%d-%m-%Y", errors="coerce").dt.year.dropna()
            if not enrollment_years.empty:
                print(
                    f"  - Thời gian gia nhập Dt_Customer: {int(enrollment_years.min())}–{int(enrollment_years.max())} "
                    f"(phù hợp mốc quy đổi Tuổi REFERENCE_YEAR={REFERENCE_YEAR})"
                )

    duplicate_3d_count = int(train_df[NUMERICAL_FEATURES].duplicated().sum())
    print(f"  - Số bản ghi trùng hoàn toàn trên 3 đặc trưng số ở tập Train: {duplicate_3d_count}")

    summary_stats = train_df[NUMERICAL_FEATURES].agg(["count", "mean", "median", "std", "min", "max", "skew"]).T
    print("\n [THỐNG KÊ MÔ TẢ TẬP TRAIN]")
    print(summary_stats.round(2).to_string())

    audit_train = audit_missing_and_outliers(train_df)
    audit_train.to_csv(METRICS_DIR / "train_iqr_audit.csv", index=False)
    print("\n [BẢNG KIỂM TOÁN NULL & IQR OUTLIERS TRÊN TẬP TRAIN]")
    print(audit_train.to_string(index=False))

    plot_eda_distributions(train_df)
    plot_eda_correlations(train_df)
    logger.info("Hoàn tất Bước 1 (EDA).")
    return audit_train


def step_2_feature_scaling(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[FeatureScaler, Any, Any]:
    """
    Bước 2: So sánh 3 scaler (Standard, MinMax, Robust) trên phân phối & hiệu quả K-Means,
    kiểm chứng hậu điều kiện Train/Test và fit StandardScaler chỉ trên Train.
    """
    print_section("BƯỚC 2: CHUẨN HÓA ĐẶC TRƯNG (FEATURE SCALING)")
    scaler_comparison_df = compare_scalers(train_df)
    scaler_comparison_df.to_csv(METRICS_DIR / "scaler_comparison.csv", index=False)
    print("\n [SO SÁNH THANG ĐO GỐC VS STANDARD / MINMAX / ROBUST]")
    print(scaler_comparison_df.to_string(index=False))

    scaler_clustering_df = evaluate_scalers_on_clustering(train_df, test_df=test_df)
    scaler_clustering_df.to_csv(METRICS_DIR / "scaler_clustering_comparison.csv", index=False)
    print("\n [ĐỐI CHIẾU TÁC ĐỘNG CỦA SCALER LÊN K-MEANS (CỐ ĐỊNH DỮ LIỆU & ĐẶC TRƯNG)]")
    print(scaler_clustering_df.to_string(index=False))

    scaler = FeatureScaler(method="standard")
    train_matrix = scaler.fit_transform(train_df)
    test_matrix = scaler.transform(test_df)

    scaler_params_df = scaler.get_learned_params()
    scaler_params_df.to_csv(METRICS_DIR / "scaler_params.csv", index=False)
    print("\n [THAM SỐ STANDARDSCALER ĐÃ HỌC CHỈ TRÊN TẬP TRAIN]")
    print(scaler_params_df.to_string(index=False))

    # Kiểm chứng hậu điều kiện trên Train/Test và đối chiếu độ nhạy biên [0, 1] với MinMaxScaler
    minmax_reference_scaler = FeatureScaler(method="minmax").fit(train_df)
    verification_df = pd.concat(
        [
            scaler.verify_scaled_matrix(train_matrix, is_train=True),
            scaler.verify_scaled_matrix(test_matrix, is_train=False),
            minmax_reference_scaler.verify_scaled_matrix(minmax_reference_scaler.transform(test_df), is_train=False),
        ],
        ignore_index=True,
    )
    verification_df.to_csv(METRICS_DIR / "scaler_verification.csv", index=False)
    print("\n [HẬU KIỂM MA TRẬN CHUẨN HÓA TRÊN TRAIN & HOLDOUT TEST]")
    print(verification_df.to_string(index=False))

    plot_scaling_comparison(train_df, train_matrix)
    logger.info("Hoàn tất Bước 2: Đã chuẩn hóa tập Train và Test không rò rỉ dữ liệu.")
    return scaler, train_matrix, test_matrix


def step_3_clustering_experiments(train_matrix: Any) -> Dict[str, Any]:
    """
    Bước 3: Huấn luyện K-Means Baseline, đồng thời đối chứng Hierarchical, DBSCAN và K-Medoids.
    """
    print_section("BƯỚC 3: THỬ NGHIỆM VÀ HUẤN LUYỆN CÁC MÔ HÌNH PHÂN CỤM")

    # 1. Baseline chính: K-Means (khởi tạo k-means++, quét K=2..10)
    kmeans_trainer = KMeansTrainer(random_state=RANDOM_SEED)
    k_search_df = kmeans_trainer.search_optimal_k(train_matrix, K_SEARCH_RANGE)
    k_search_df.to_csv(METRICS_DIR / "kmeans_k_search.csv", index=False)
    print("\n [KẾT QUẢ QUÉT K-MEANS TỪ K=2 ĐẾN K=10]")
    print(k_search_df.to_string(index=False))

    optimal_k = int(k_search_df.loc[k_search_df["Silhouette"].idxmax(), "K"])
    plot_elbow_and_silhouette(k_search_df, optimal_k=optimal_k)

    hyperparam_df = kmeans_trainer.experiment_hyperparameters(train_matrix, n_clusters=optimal_k)
    hyperparam_df.to_csv(METRICS_DIR / "kmeans_hyperparams.csv", index=False)
    kmeans_model = kmeans_trainer.fit_final_model(train_matrix, optimal_k=optimal_k, init="k-means++", n_init=20)

    # 2. Hierarchical Clustering (đối chiếu linkage Ward/Complete/Average, ghi nhận hiệu ứng chaining)
    cophenetic_rows = []
    trainer_ward = None
    for method_name in ("ward", "complete", "average"):
        temp_trainer = HierarchicalTrainer(linkage_method=method_name)
        temp_trainer.compute_linkage_matrix(train_matrix)
        k2_model = temp_trainer.fit_model(train_matrix, n_clusters=2)
        k2_labels = k2_model.labels_
        k2_metrics = evaluate_clustering(
            train_matrix,
            k2_labels,
            model_name=f"Hier_{method_name}_K2",
            compute_dunn=False,
        )
        k2_cluster_counts = pd.Series(k2_labels).value_counts().sort_index().to_dict()
        cophenetic_rows.append(
            {
                "Linkage": method_name,
                "Cophenetic_Correlation": round(temp_trainer.cophenetic_corr_, 4),
                "K2_Silhouette": k2_metrics["silhouette"],
                "K2_Balance_Ratio": k2_metrics["cluster_balance_ratio"],
                "K2_Cluster_Sizes": str(k2_cluster_counts),
            }
        )
        if method_name == "ward":
            trainer_ward = temp_trainer
    cophenetic_df = pd.DataFrame(cophenetic_rows)
    cophenetic_df.to_csv(METRICS_DIR / "hierarchical_cophenetic.csv", index=False)
    print("\n [ĐỐI CHIẾU PHƯƠNG PHÁP LINKAGE HIERARCHICAL (TẠI K=2)]")
    print(cophenetic_df.to_string(index=False))

    if trainer_ward is None:
        trainer_ward = HierarchicalTrainer(linkage_method="ward")
        trainer_ward.compute_linkage_matrix(train_matrix)
    hier_search_df = trainer_ward.search_optimal_k(train_matrix, K_SEARCH_RANGE)
    hier_search_df.to_csv(METRICS_DIR / "hierarchical_k_search.csv", index=False)
    hier_optimal_k = int(hier_search_df.loc[hier_search_df["Silhouette"].idxmax(), "K"])

    hier_model = trainer_ward.fit_model(train_matrix, n_clusters=hier_optimal_k)
    cut_threshold = trainer_ward.compute_cut_threshold(hier_optimal_k)
    plot_dendrogram(trainer_ward.linkage_matrix_, cut_threshold=cut_threshold)

    # 3. DBSCAN (đồ thị k-NN k=4 & lưới tham số eps, min_samples)
    dbscan_trainer = DBSCANTrainer()
    k_distances = dbscan_trainer.compute_k_distances(train_matrix, k=4)
    plot_dbscan_k_distance(k_distances, k=4)

    dbscan_grid = dbscan_trainer.grid_search_params(
        train_matrix,
        eps_values=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        min_samples_values=[4, 5, 8, 10],
    )
    dbscan_grid.to_csv(METRICS_DIR / "dbscan_grid_search.csv", index=False)
    best_eps, best_ms = dbscan_trainer.best_params_ if dbscan_trainer.best_params_ else (0.5, 5)
    dbscan_model = dbscan_trainer.fit_model(train_matrix, eps=best_eps, min_samples=best_ms)

    # 4. K-Medoids (BUILD tất định & cập nhật nội cụm, quét K=2..10 và so sánh thước đo khoảng cách)
    kmedoids_search_df = KMedoidsTrainer(
        n_clusters=optimal_k,
        metric="euclidean",
        random_state=RANDOM_SEED,
    ).search_optimal_k(train_matrix, K_SEARCH_RANGE)
    kmedoids_search_df.to_csv(METRICS_DIR / "kmedoids_k_search.csv", index=False)
    kmedoids_optimal_k = int(kmedoids_search_df.loc[kmedoids_search_df["Silhouette"].idxmax(), "K"])

    kmedoids_metric_rows = []
    kmedoids_trainer = None
    for metric_name in ("euclidean", "manhattan"):
        trainer_km = KMedoidsTrainer(
            n_clusters=kmedoids_optimal_k,
            metric=metric_name,
            random_state=RANDOM_SEED,
        )
        km_labels = trainer_km.fit_predict(train_matrix)
        km_metrics = evaluate_clustering(
            train_matrix,
            km_labels,
            model_name=f"K-Medoids ({metric_name}, K={kmedoids_optimal_k})",
            compute_dunn=False,
        )
        kmedoids_metric_rows.append(
            {
                "Metric": metric_name,
                "K": kmedoids_optimal_k,
                "Total_Cost": round(float(trainer_km.total_cost_), 4),
                "Silhouette": km_metrics["silhouette"],
                "Davies_Bouldin": km_metrics["davies_bouldin"],
                "Calinski_Harabasz": km_metrics["calinski_harabasz"],
            }
        )
        if metric_name == "euclidean":
            kmedoids_trainer = trainer_km
    pd.DataFrame(kmedoids_metric_rows).to_csv(METRICS_DIR / "kmedoids_metric_comparison.csv", index=False)
    kmedoids_labels = kmedoids_trainer.labels_

    logger.info(f"Hoàn tất Bước 3 với K tối ưu từ K-Means Baseline = {optimal_k}.")
    return {
        "optimal_k": optimal_k,
        "kmeans_model": kmeans_model,
        "kmeans_labels": kmeans_model.labels_,
        "k_search_df": k_search_df,
        "hierarchical_optimal_k": hier_optimal_k,
        "hierarchical_model": hier_model,
        "hierarchical_labels": hier_model.labels_,
        "hierarchical_k_search_df": hier_search_df,
        "linkage_matrix_ward": trainer_ward.linkage_matrix_,
        "dbscan_model": dbscan_model,
        "dbscan_labels": dbscan_model.labels_,
        "dbscan_params": (best_eps, best_ms),
        "kmedoids_optimal_k": kmedoids_optimal_k,
        "kmedoids_trainer": kmedoids_trainer,
        "kmedoids_labels": kmedoids_labels,
    }


def step_4_evaluation_and_profiling(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_matrix: Any,
    test_matrix: Any,
    scaler: FeatureScaler,
    clustering_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Bước 4: Đánh giá định lượng 4 mô hình, kiểm chứng tập Holdout Test, trực quan hóa và lập Persona.
    """
    print_section("BƯỚC 4: ĐÁNH GIÁ CHẤT LƯỢNG, TRỰC QUAN HÓA & CHÂN DUNG PERSONA")
    optimal_k = clustering_results["optimal_k"]
    hier_k = clustering_results.get("hierarchical_optimal_k", optimal_k)
    kmedoids_k = clustering_results.get("kmedoids_optimal_k", optimal_k)
    kmeans_model = clustering_results["kmeans_model"]

    # So sánh đối đầu 4 thuật toán trên tập Train
    eval_results = [
        evaluate_clustering(train_matrix, clustering_results["kmeans_labels"], f"K-Means (K={optimal_k})"),
        evaluate_clustering(train_matrix, clustering_results["hierarchical_labels"], f"Hierarchical Ward (K={hier_k})"),
        evaluate_clustering(
            train_matrix,
            clustering_results["dbscan_labels"],
            f"DBSCAN (eps={clustering_results['dbscan_params'][0]}, ms={clustering_results['dbscan_params'][1]})",
        ),
        evaluate_clustering(train_matrix, clustering_results["kmedoids_labels"], f"K-Medoids (K={kmedoids_k})"),
    ]
    comparison_df = compare_all_models(eval_results)
    print("\n [BẢNG ĐỐI CHIẾU CÁC THUẬT TOÁN TRÊN TẬP TRAIN]")
    print(comparison_df.to_string(index=False))

    # Kiểm chứng độ ổn định phân bố cụm trên tập Holdout Test độc lập (20%) cho các mô hình quy nạp
    # (Lưu ý: không dùng chỉ số nội tại trên Holdout để suy diễn overfitting như bài toán có giám sát)
    kmeans_test_labels = kmeans_model.predict(test_matrix)
    kmedoids_test_labels = clustering_results["kmedoids_trainer"].predict(test_matrix)
    holdout_df = pd.DataFrame(
        [
            evaluate_clustering(test_matrix, kmeans_test_labels, f"K-Means Test Holdout (K={optimal_k})"),
            evaluate_clustering(test_matrix, kmedoids_test_labels, f"K-Medoids Test Holdout (K={kmedoids_k})"),
        ]
    )
    holdout_df.to_csv(METRICS_DIR / "holdout_test_evaluation.csv", index=False)
    print("\n [KIỂM CHỨNG ĐỘ ỔN ĐỊNH PHÂN CỤM TRÊN TẬP HOLDOUT TEST (20%)]")
    print(holdout_df[["model_name", "n_clusters", "silhouette", "davies_bouldin", "calinski_harabasz"]].to_string(index=False))

    # Trực quan hóa mô hình Baseline chính (K-Means) và DBSCAN
    original_centroids = scaler.inverse_transform(kmeans_model.cluster_centers_)
    plot_clusters_2d(
        train_df,
        clustering_results["kmeans_labels"],
        centroids=original_centroids,
        title=f"Phân Cụm Khách Hàng K-Means (K={optimal_k}) - Thu Nhập & Chi Tiêu",
        filename="02_clusters_2d.png",
    )
    plot_clusters_2d(
        train_df,
        clustering_results["dbscan_labels"],
        centroids=None,
        title="Phân Cụm Mật Độ DBSCAN & Các Điểm Nhiễu (-1)",
        filename="02b_dbscan_clusters_2d.png",
    )
    plot_clusters_3d(
        train_df,
        clustering_results["kmeans_labels"],
        title=f"Không Gian Phân Cụm K-Means 3D (K={optimal_k})",
        filename="03_clusters_3d.png",
    )
    plot_silhouette_analysis(
        train_matrix,
        clustering_results["kmeans_labels"],
        optimal_k=optimal_k,
        filename="05_silhouette_analysis.png",
    )

    # Phân tích hồ sơ khách hàng & xây dựng Personas động
    profile_df = create_cluster_profiles(train_df, clustering_results["kmeans_labels"])
    print("\n [BẢNG HỒ SƠ THỐNG KÊ TỪNG PHÂN KHÚC (TRAIN)]")
    print(profile_df.to_string(index=False))

    personas = generate_customer_personas(profile_df)
    print("\n [CHÂN DUNG KHÁCH HÀNG & CHIẾN LƯỢC TIẾP THỊ]")
    for p in personas:
        print(
            f" - Cụm {p['cluster_id']}: {p['persona_name']} | "
            f"Quy mô: {p['size']} ({p['pct']}%) | Tuổi TB: {p['avg_age']} | "
            f"Thu nhập TB: {p['avg_income']:,.0f} | Chi tiêu TB: {p['avg_spending']:,.0f}\n"
            f"   -> Chiến lược: {p['marketing_strategy']}"
        )

    logger.info("Hoàn tất Bước 4 (Evaluation & Profiling).")
    return {
        "comparison_df": comparison_df,
        "holdout_df": holdout_df,
        "profile_df": profile_df,
        "personas": personas,
    }


def step_5_save_and_inference(
    final_model: Any,
    scaler: FeatureScaler,
    optimal_k: int,
    personas: list[dict[str, Any]],
    test_df: pd.DataFrame,
    feature_medians: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Bước 5: Lưu trữ gói mô hình hoàn chỉnh và thử nghiệm suy luận trên khách hàng mới & tập Test.
    """
    print_section("BƯỚC 5: LƯU TRỮ MÔ HÌNH & THỬ NGHIỆM SUY LUẬN")
    metadata = {
        "feature_cols": scaler.feature_cols,
        "scaler_method": scaler.method,
        "n_clusters": int(optimal_k),
        "reference_year": REFERENCE_YEAR,
        "feature_medians": feature_medians or {},
        "personas": personas,
    }
    save_model(final_model, scaler, model_name="final_kmeans_pipeline.pkl", metadata=metadata)

    predictor = CustomerSegmentPredictor("final_kmeans_pipeline.pkl")
    sample_prediction = predictor.predict_single_customer(age=35, annual_income=75000.0, total_spending=1400.0)
    print(" [THỬ NGHIỆM DỰ ĐOÁN 1 KHÁCH HÀNG MỚI]")
    print(
        f" Đầu vào: {sample_prediction['customer_input']}\n"
        f" -> Cụm dự đoán: {sample_prediction['assigned_cluster']} ({sample_prediction['persona_name']})\n"
        f" -> Khoảng cách tới tâm cụm (không gian chuẩn hóa): {sample_prediction['distance_to_center']}\n"
        f" -> Gợi ý: {sample_prediction['marketing_recommendation']}"
    )

    test_pred_df = predictor.predict_batch(test_df)
    test_pred_df.to_csv(METRICS_DIR / "test_predictions.csv", index=False)
    logger.info("Hoàn tất Bước 5 (Lưu trữ & Suy luận thử nghiệm).")
    return sample_prediction


def main():
    """Hàm điều phối chính chạy tuần tự từ Bước 0 đến Bước 5."""
    print_section(
        "BẮT ĐẦU PIPELINE PHÂN CỤM KHÁCH HÀNG (CUSTOMER CLUSTERING)",
        "Thực hiện theo Practice Exercise 3.pdf - Phạm vi bài lab học tập",
    )
    logger.info("Khởi chạy toàn bộ pipeline...")

    raw_df, train_df, test_df = step_0_data_ingestion()
    step_1_data_exploration(train_df, raw_df=raw_df)
    scaler, train_matrix, test_matrix = step_2_feature_scaling(train_df, test_df)
    clustering_results = step_3_clustering_experiments(train_matrix)
    eval_outputs = step_4_evaluation_and_profiling(
        train_df,
        test_df,
        train_matrix,
        test_matrix,
        scaler,
        clustering_results,
    )
    step_5_save_and_inference(
        clustering_results["kmeans_model"],
        scaler,
        clustering_results["optimal_k"],
        eval_outputs["personas"],
        test_df,
        feature_medians=compute_train_medians(train_df),
    )

    print_section(
        "HOÀN TẤT PIPELINE THÀNH CÔNG!",
        "Biểu đồ, bảng đánh giá và mô hình đã được xuất tại thư mục results/",
    )


if __name__ == "__main__":
    main()
