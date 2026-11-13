"""
=============================================================================
MODULE: CHUẨN HÓA ĐẶC TRƯNG (FEATURE SCALING)
=============================================================================
Mục đích (Đáp ứng Bước 2 trong đề bài Practice Exercise 3):
- Chuẩn hóa (Standardize / Normalize) các đặc trưng số: Age, Income, Total_Spending
  để đưa về cùng một thang đo, tránh biến có giá trị lớn (Income) chi phối khoảng cách Euclidean.
- Đảm bảo không rò rỉ dữ liệu: chỉ fit trên tập Train, sau đó transform trên Train/Test.
- Đảm bảo loại trừ cột định danh ID và kiểm tra tính hữu hạn (finite), phương sai > 0.
=============================================================================
"""

from typing import Iterable, List, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.config import FEATURE_COLS, ID_COLUMN, K_SEARCH_RANGE, NUMERICAL_FEATURES, RANDOM_SEED


_SCALER_FACTORY = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def _extract_valid_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    encode_gender: bool = False,
) -> np.ndarray:
    """Trích xuất ma trận số float64 và kiểm tra tính đầy đủ, hữu hạn."""
    if df.empty:
        raise ValueError("DataFrame đầu vào đang rỗng, không thể chuẩn hóa.")

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Thiếu cột đặc trưng để chuẩn hóa: {missing_cols}")

    if encode_gender:
        raise ValueError(
            "Tập dữ liệu marketing_campaign.csv không có cột Gender; "
            "không tự tạo cột giả khi chuẩn hóa."
        )

    matrix = df[feature_cols].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Dữ liệu đặc trưng chứa NaN hoặc Inf. Hãy làm sạch trước khi chuẩn hóa.")
    return matrix


class FeatureScaler:
    """
    Lớp chuẩn hóa đặc trưng số cho bài toán phân cụm khách hàng.
    """

    def __init__(
        self,
        method: str = "standard",
        feature_cols: Optional[List[str]] = None,
        encode_gender: bool = False,
    ):
        """
        Khởi tạo bộ chuẩn hóa theo phương pháp ('standard', 'minmax' hoặc 'robust').
        """
        normalized_method = method.lower().strip()
        if normalized_method not in _SCALER_FACTORY:
            raise ValueError(
                f"Phương pháp chuẩn hóa '{method}' không hợp lệ. "
                f"Chỉ hỗ trợ: {list(_SCALER_FACTORY.keys())}."
            )

        self.method = normalized_method
        self.feature_cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
        if ID_COLUMN in self.feature_cols:
            raise ValueError(f"Không được đưa cột định danh '{ID_COLUMN}' vào danh sách đặc trưng cần chuẩn hóa.")

        self.encode_gender = encode_gender
        self.scaler = _SCALER_FACTORY[self.method]()
        self.is_fitted_ = False

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        """
        Học tham số chuẩn hóa (Mean/Std hoặc Min/Max) chỉ trên tập Train.
        """
        matrix = _extract_valid_matrix(df, self.feature_cols, self.encode_gender)

        # Kiểm tra cột hằng số (phương sai bằng 0) trước khi fit
        variances = np.var(matrix, axis=0)
        constant_cols = [
            col for col, var in zip(self.feature_cols, variances) if np.isclose(var, 0.0)
        ]
        if constant_cols:
            raise ValueError(
                f"Phát hiện cột hằng số (phương sai = 0): {constant_cols}. "
                "Cột không mang thông tin phân biệt cho phân cụm."
            )

        self.train_min_ = np.min(matrix, axis=0)
        self.train_max_ = np.max(matrix, axis=0)
        self.scaler.fit(matrix)
        self.is_fitted_ = True
        print(f" [CHUẨN HÓA] Đã fit {self.method.upper()} scaler trên {len(df):,} mẫu với {len(self.feature_cols)} đặc trưng.")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Biến đổi dữ liệu bằng tham số đã fit từ tập Train.
        """
        if not self.is_fitted_:
            raise RuntimeError("FeatureScaler chưa được fit. Hãy gọi fit() trên tập Train trước khi transform().")

        matrix = _extract_valid_matrix(df, self.feature_cols, self.encode_gender)
        scaled_matrix = self.scaler.transform(matrix)
        if not np.all(np.isfinite(scaled_matrix)):
            raise ValueError("Ma trận sau khi chuẩn hóa phát sinh giá trị NaN/Inf bất thường.")
        return scaled_matrix

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Thực hiện tuần tự fit(df) rồi transform(df) trên tập Train.
        """
        return self.fit(df).transform(df)

    def inverse_transform(self, scaled_matrix: np.ndarray) -> np.ndarray:
        """
        Quy đổi ma trận hoặc tâm cụm từ không gian chuẩn hóa về lại đơn vị gốc.
        """
        if not self.is_fitted_:
            raise RuntimeError("FeatureScaler chưa được fit nên không thể inverse_transform.")
        arr = np.asarray(scaled_matrix, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != len(self.feature_cols):
            raise ValueError(
                f"Ma trận cần inverse_transform phải có 2 chiều (*, {len(self.feature_cols)}), "
                f"nhưng nhận được shape {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Ma trận đầu vào inverse_transform chứa NaN hoặc Inf.")
        return self.scaler.inverse_transform(arr)

    def get_learned_params(self) -> pd.DataFrame:
        """
        Trích xuất toàn bộ tham số thống kê mà scaler đã học từ tập Train
        (phục vụ giải trình hậu kiểm và minh bạch công thức biến đổi).
        """
        if not self.is_fitted_:
            raise RuntimeError("FeatureScaler chưa được fit nên chưa có tham số đã học.")

        formula_map = {
            "standard": "Z = (x - Mean_Train) / Std_Train",
            "minmax": "X_norm = (x - Min_Train) / (Max_Train - Min_Train)",
            "robust": "X_robust = (x - Median_Train) / IQR_Train",
        }
        param_rows = []
        for idx, col_name in enumerate(self.feature_cols):
            if self.method == "standard":
                center_value = float(self.scaler.mean_[idx])
                scale_value = float(self.scaler.scale_[idx])
            elif self.method == "minmax":
                center_value = float(self.scaler.data_min_[idx])
                scale_value = float(self.scaler.data_range_[idx])
            else:
                center_value = float(self.scaler.center_[idx])
                scale_value = float(self.scaler.scale_[idx])

            train_min = float(self.train_min_[idx]) if hasattr(self, "train_min_") else np.nan
            train_max = float(self.train_max_[idx]) if hasattr(self, "train_max_") else np.nan
            param_rows.append(
                {
                    "Phương pháp": self.method,
                    "Đặc trưng": col_name,
                    "Tham số tâm": round(center_value, 4),
                    "Hệ số chia": round(scale_value, 4),
                    "Cận Min Train": round(train_min, 4),
                    "Cận Max Train": round(train_max, 4),
                    "Công thức": formula_map[self.method],
                }
            )
        return pd.DataFrame(param_rows)

    def verify_scaled_matrix(
        self,
        scaled_matrix: np.ndarray,
        is_train: bool = True,
    ) -> pd.DataFrame:
        """
        Hậu kiểm chất lượng ma trận sau chuẩn hóa trên Train hoặc Holdout Test.
        """
        if not self.is_fitted_:
            raise RuntimeError("FeatureScaler chưa được fit nên không thể kiểm tra ma trận chuẩn hóa.")

        arr = np.asarray(scaled_matrix, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != len(self.feature_cols):
            raise ValueError(
                f"Ma trận chuẩn hóa phải có shape (*, {len(self.feature_cols)}), nhận được {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Ma trận sau chuẩn hóa chứa NaN hoặc Inf.")

        split_label = "Train" if is_train else "Holdout Test"
        audit_rows = []
        for idx, col_name in enumerate(self.feature_cols):
            col_values = arr[:, idx]
            val_min = float(np.min(col_values))
            val_max = float(np.max(col_values))
            val_mean = float(np.mean(col_values))
            val_std = float(np.std(col_values))
            val_median = float(np.median(col_values))
            val_iqr = float(np.percentile(col_values, 75) - np.percentile(col_values, 25))

            if self.method == "standard":
                if is_train:
                    is_passed = abs(val_mean) < 1e-6 and abs(val_std - 1.0) < 1e-2
                    note = "Mean ≈ 0, Std ≈ 1"
                else:
                    is_passed = True
                    note = "Dùng tham số Mean/Std của Train"
            elif self.method == "minmax":
                if is_train:
                    is_passed = val_min >= -1e-9 and val_max <= 1.0 + 1e-9
                    note = "Nằm trọn trong [0, 1]"
                else:
                    out_of_range_count = int(np.sum((col_values < 0.0) | (col_values > 1.0)))
                    is_passed = True
                    note = f"Ngoài [0, 1]: {out_of_range_count} mẫu"
            else:
                if is_train:
                    is_passed = abs(val_median) < 1e-6
                    note = "Median ≈ 0, chuẩn hóa theo IQR"
                else:
                    is_passed = True
                    note = "Dùng tham số Median/IQR của Train"

            audit_rows.append(
                {
                    "Tập dữ liệu": split_label,
                    "Phương pháp": self.method,
                    "Đặc trưng": col_name,
                    "Min": round(val_min, 4),
                    "Max": round(val_max, 4),
                    "Mean": round(val_mean, 4),
                    "Std": round(val_std, 4),
                    "Median": round(val_median, 4),
                    "IQR": round(val_iqr, 4),
                    "Hậu kiểm": note,
                    "Đạt chuẩn": bool(is_passed),
                }
            )
        return pd.DataFrame(audit_rows)


def compare_scalers(
    df: pd.DataFrame,
    feature_cols: List[str] = NUMERICAL_FEATURES,
) -> pd.DataFrame:
    """
    So sánh thống kê phân phối (Min, Max, Mean, Std, Median, IQR, Skew)
    giữa dữ liệu gốc, StandardScaler, MinMaxScaler và RobustScaler.
    """
    valid_cols = [col for col in feature_cols if col in df.columns]
    if not valid_cols:
        raise KeyError(f"Không tìm thấy đặc trưng nào trong {feature_cols} để so sánh chuẩn hóa.")

    raw_matrix = _extract_valid_matrix(df, valid_cols)
    matrices_by_method = {
        "Gốc (Raw)": raw_matrix,
        "StandardScaler": FeatureScaler(method="standard", feature_cols=valid_cols).fit_transform(df),
        "MinMaxScaler": FeatureScaler(method="minmax", feature_cols=valid_cols).fit_transform(df),
        "RobustScaler": FeatureScaler(method="robust", feature_cols=valid_cols).fit_transform(df),
    }

    summary_rows = []
    for method_label, method_matrix in matrices_by_method.items():
        for col_idx, col_name in enumerate(valid_cols):
            col_values = method_matrix[:, col_idx]
            skewness = float(pd.Series(col_values).skew())
            summary_rows.append(
                {
                    "Phương pháp": method_label,
                    "Đặc trưng": col_name,
                    "Min": round(float(np.min(col_values)), 4),
                    "Max": round(float(np.max(col_values)), 4),
                    "Mean": round(float(np.mean(col_values)), 4),
                    "Std": round(float(np.std(col_values)), 4),
                    "Median": round(float(np.median(col_values)), 4),
                    "IQR": round(float(np.percentile(col_values, 75) - np.percentile(col_values, 25)), 4),
                    "Skew": 0.0 if np.isnan(skewness) else round(skewness, 4),
                }
            )

    comparison_df = pd.DataFrame(summary_rows)
    print(f" [SO SÁNH SCALER] Đã tổng hợp chỉ số cho {len(valid_cols)} đặc trưng qua 4 trạng thái thang đo (Raw, Standard, MinMax, Robust).")
    return comparison_df


def evaluate_scalers_on_clustering(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    feature_cols: List[str] = NUMERICAL_FEATURES,
    fixed_k: int = 5,
    k_range: Iterable[int] = K_SEARCH_RANGE,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Khảo sát thực nghiệm tác động của 3 bộ chuẩn hóa (Standard, MinMax, Robust) lên chất lượng phân cụm K-Means
    khi giữ cố định tập khách hàng, danh sách đặc trưng và thuật toán khởi tạo k-means++.
    """
    valid_cols = [col for col in feature_cols if col in train_df.columns]
    if not valid_cols:
        raise KeyError(f"Không tìm thấy đặc trưng nào trong {feature_cols} để đánh giá scaler.")

    methods = ["standard", "minmax", "robust"]
    evaluation_rows = []
    unique_points_count = len(np.unique(_extract_valid_matrix(train_df, valid_cols), axis=0))
    valid_k_values = [int(k) for k in k_range if 2 <= int(k) < unique_points_count]
    if not valid_k_values:
        raise ValueError("Không đủ số điểm phân biệt trên tập Train để tìm cụm K >= 2.")

    for method_name in methods:
        scaler = FeatureScaler(method=method_name, feature_cols=valid_cols).fit(train_df)
        scaled_train = scaler.transform(train_df)

        best_k = valid_k_values[0]
        best_sil = -1.0
        best_model = None
        best_labels = None

        for k in valid_k_values:
            km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
            labels = km.fit_predict(scaled_train)
            sil_score = float(silhouette_score(scaled_train, labels))
            if sil_score > best_sil:
                best_sil = sil_score
                best_k = k
                best_model = km
                best_labels = labels

        cluster_counts = pd.Series(best_labels).value_counts()
        balance_ratio = round(float(cluster_counts.min() / max(cluster_counts.max(), 1)), 4)

        fixed_sil = np.nan
        fixed_dbi = np.nan
        if 2 <= fixed_k < unique_points_count:
            fixed_km = KMeans(
                n_clusters=fixed_k,
                init="k-means++",
                n_init=10,
                random_state=random_state,
            )
            fixed_labels = fixed_km.fit_predict(scaled_train)
            fixed_sil = round(float(silhouette_score(scaled_train, fixed_labels)), 4)
            fixed_dbi = round(float(davies_bouldin_score(scaled_train, fixed_labels)), 4)

        test_sil = np.nan
        test_out_of_bounds_count = 0
        if test_df is not None and not test_df.empty:
            scaled_test = scaler.transform(test_df)
            test_labels = best_model.predict(scaled_test)
            if len(np.unique(test_labels)) >= 2 and len(scaled_test) > len(np.unique(test_labels)):
                test_sil = round(float(silhouette_score(scaled_test, test_labels)), 4)
            train_scaled_min = np.min(scaled_train, axis=0)
            train_scaled_max = np.max(scaled_train, axis=0)
            test_out_of_bounds_count = int(
                np.sum(np.any((scaled_test < train_scaled_min) | (scaled_test > train_scaled_max), axis=1))
            )

        evaluation_rows.append(
            {
                "Scaler": method_name,
                "K tối ưu": int(best_k),
                "Silhouette Train": round(best_sil, 4),
                "DBI Train": round(float(davies_bouldin_score(scaled_train, best_labels)), 4),
                "CHI Train": round(float(calinski_harabasz_score(scaled_train, best_labels)), 2),
                "Inertia Train": round(float(best_model.inertia_), 2),
                "Cân bằng cụm": balance_ratio,
                f"Silhouette K={fixed_k}": fixed_sil,
                f"DBI K={fixed_k}": fixed_dbi,
                "Silhouette Test": test_sil,
                "Mẫu Test vượt cận Train": test_out_of_bounds_count,
            }
        )

    evaluation_df = pd.DataFrame(evaluation_rows)
    print(f" [ĐỐI CHIẾU SCALER TRÊN K-MEANS] Đã so sánh thực nghiệm 3 scaler qua dải K={valid_k_values}.")
    return evaluation_df
