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

from src.config import ID_COLUMN, K_SEARCH_RANGE, NUMERICAL_FEATURES, RANDOM_SEED


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

    cot_thieu = [col for col in feature_cols if col not in df.columns]
    if cot_thieu:
        raise KeyError(f"Thiếu cột đặc trưng để chuẩn hóa: {cot_thieu}")

    if encode_gender:
        raise ValueError(
            "Tập dữ liệu marketing_campaign.csv không có cột Gender; "
            "không tự tạo cột giả khi chuẩn hóa."
        )

    ma_tran = df[feature_cols].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(ma_tran)):
        raise ValueError("Dữ liệu đặc trưng chứa NaN hoặc Inf. Hãy làm sạch trước khi chuẩn hóa.")
    return ma_tran


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
        ten_phuong_phap = method.lower().strip()
        if ten_phuong_phap not in _SCALER_FACTORY:
            raise ValueError(
                f"Phương pháp chuẩn hóa '{method}' không hợp lệ. "
                f"Chỉ hỗ trợ: {list(_SCALER_FACTORY.keys())}."
            )

        self.method = ten_phuong_phap
        self.feature_cols = list(feature_cols) if feature_cols is not None else list(NUMERICAL_FEATURES)
        if ID_COLUMN in self.feature_cols:
            raise ValueError(f"Không được đưa cột định danh '{ID_COLUMN}' vào danh sách đặc trưng cần chuẩn hóa.")

        self.encode_gender = encode_gender
        self.scaler = _SCALER_FACTORY[self.method]()
        self.is_fitted_ = False

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        """
        Học tham số chuẩn hóa (Mean/Std hoặc Min/Max) chỉ trên tập Train.
        """
        ma_tran = _extract_valid_matrix(df, self.feature_cols, self.encode_gender)

        # Kiểm tra cột hằng số (phương sai bằng 0) trước khi fit
        phuong_sai = np.var(ma_tran, axis=0)
        cot_hang_so = [
            col for col, var in zip(self.feature_cols, phuong_sai) if np.isclose(var, 0.0)
        ]
        if cot_hang_so:
            raise ValueError(
                f"Phát hiện cột hằng số (phương sai = 0): {cot_hang_so}. "
                "Cột không mang thông tin phân biệt cho phân cụm."
            )

        self.train_min_ = np.min(ma_tran, axis=0)
        self.train_max_ = np.max(ma_tran, axis=0)
        self.scaler.fit(ma_tran)
        self.is_fitted_ = True
        print(f" [CHUẨN HÓA] Đã fit {self.method.upper()} scaler trên {len(df):,} mẫu với {len(self.feature_cols)} đặc trưng.")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Biến đổi dữ liệu bằng tham số đã fit từ tập Train.
        """
        if not self.is_fitted_:
            raise RuntimeError("FeatureScaler chưa được fit. Hãy gọi fit() trên tập Train trước khi transform().")

        ma_tran = _extract_valid_matrix(df, self.feature_cols, self.encode_gender)
        ket_qua = self.scaler.transform(ma_tran)
        if not np.all(np.isfinite(ket_qua)):
            raise ValueError("Ma trận sau khi chuẩn hóa phát sinh giá trị NaN/Inf bất thường.")
        return ket_qua

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Thực hiện tuần tự fit(df) rồi transform(df) trên tập Train.
        """
        return self.fit(df).transform(df)

    def inverse_transform(self, ma_tran_scaled: np.ndarray) -> np.ndarray:
        """
        Quy đổi ma trận hoặc tâm cụm từ không gian chuẩn hóa về lại đơn vị gốc.
        """
        if not self.is_fitted_:
            raise RuntimeError("FeatureScaler chưa được fit nên không thể inverse_transform.")
        arr = np.asarray(ma_tran_scaled, dtype=np.float64)
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

        cong_thuc_map = {
            "standard": "Z = (x - Mean_Train) / Std_Train",
            "minmax": "X_norm = (x - Min_Train) / (Max_Train - Min_Train)",
            "robust": "X_robust = (x - Median_Train) / IQR_Train",
        }
        cac_dong = []
        for idx, ten_cot in enumerate(self.feature_cols):
            if self.method == "standard":
                tham_so_tam = float(self.scaler.mean_[idx])
                he_so_chia = float(self.scaler.scale_[idx])
            elif self.method == "minmax":
                tham_so_tam = float(self.scaler.data_min_[idx])
                he_so_chia = float(self.scaler.data_range_[idx])
            else:
                tham_so_tam = float(self.scaler.center_[idx])
                he_so_chia = float(self.scaler.scale_[idx])

            min_train = float(self.train_min_[idx]) if hasattr(self, "train_min_") else np.nan
            max_train = float(self.train_max_[idx]) if hasattr(self, "train_max_") else np.nan
            cac_dong.append(
                {
                    "Phương pháp": self.method,
                    "Đặc trưng": ten_cot,
                    "Tham số tâm": round(tham_so_tam, 4),
                    "Hệ số chia": round(he_so_chia, 4),
                    "Cận Min Train": round(min_train, 4),
                    "Cận Max Train": round(max_train, 4),
                    "Công thức": cong_thuc_map[self.method],
                }
            )
        return pd.DataFrame(cac_dong)

    def verify_scaled_matrix(
        self,
        ma_tran_scaled: np.ndarray,
        is_train: bool = True,
    ) -> pd.DataFrame:
        """
        Hậu kiểm chất lượng ma trận sau chuẩn hóa trên Train hoặc Holdout Test.
        """
        if not self.is_fitted_:
            raise RuntimeError("FeatureScaler chưa được fit nên không thể kiểm tra ma trận chuẩn hóa.")

        arr = np.asarray(ma_tran_scaled, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != len(self.feature_cols):
            raise ValueError(
                f"Ma trận chuẩn hóa phải có shape (*, {len(self.feature_cols)}), nhận được {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Ma trận sau chuẩn hóa chứa NaN hoặc Inf.")

        nhan_tap = "Train" if is_train else "Holdout Test"
        cac_dong = []
        for idx, ten_cot in enumerate(self.feature_cols):
            cot = arr[:, idx]
            gt_min = float(np.min(cot))
            gt_max = float(np.max(cot))
            gt_mean = float(np.mean(cot))
            gt_std = float(np.std(cot))
            gt_median = float(np.median(cot))
            gt_iqr = float(np.percentile(cot, 75) - np.percentile(cot, 25))

            if self.method == "standard":
                if is_train:
                    dat_chuan = abs(gt_mean) < 1e-6 and abs(gt_std - 1.0) < 1e-2
                    ghi_chu = "Mean ≈ 0, Std ≈ 1"
                else:
                    dat_chuan = True
                    ghi_chu = "Dùng tham số Mean/Std của Train"
            elif self.method == "minmax":
                if is_train:
                    dat_chuan = gt_min >= -1e-9 and gt_max <= 1.0 + 1e-9
                    ghi_chu = "Nằm trọn trong [0, 1]"
                else:
                    so_ngoai_bien = int(np.sum((cot < 0.0) | (cot > 1.0)))
                    dat_chuan = True
                    ghi_chu = f"Ngoài [0, 1]: {so_ngoai_bien} mẫu"
            else:
                if is_train:
                    dat_chuan = abs(gt_median) < 1e-6
                    ghi_chu = "Median ≈ 0, chuẩn hóa theo IQR"
                else:
                    dat_chuan = True
                    ghi_chu = "Dùng tham số Median/IQR của Train"

            cac_dong.append(
                {
                    "Tập dữ liệu": nhan_tap,
                    "Phương pháp": self.method,
                    "Đặc trưng": ten_cot,
                    "Min": round(gt_min, 4),
                    "Max": round(gt_max, 4),
                    "Mean": round(gt_mean, 4),
                    "Std": round(gt_std, 4),
                    "Median": round(gt_median, 4),
                    "IQR": round(gt_iqr, 4),
                    "Hậu kiểm": ghi_chu,
                    "Đạt chuẩn": bool(dat_chuan),
                }
            )
        return pd.DataFrame(cac_dong)


def compare_scalers(
    df: pd.DataFrame,
    feature_cols: List[str] = NUMERICAL_FEATURES,
) -> pd.DataFrame:
    """
    So sánh thống kê phân phối (Min, Max, Mean, Std, Median, IQR, Skew)
    giữa dữ liệu gốc, StandardScaler, MinMaxScaler và RobustScaler.
    """
    cot_hop_le = [col for col in feature_cols if col in df.columns]
    if not cot_hop_le:
        raise KeyError(f"Không tìm thấy đặc trưng nào trong {feature_cols} để so sánh chuẩn hóa.")

    ma_tran_goc = _extract_valid_matrix(df, cot_hop_le)
    du_lieu_theo_phuong_phap = {
        "Gốc (Raw)": ma_tran_goc,
        "StandardScaler": FeatureScaler(method="standard", feature_cols=cot_hop_le).fit_transform(df),
        "MinMaxScaler": FeatureScaler(method="minmax", feature_cols=cot_hop_le).fit_transform(df),
        "RobustScaler": FeatureScaler(method="robust", feature_cols=cot_hop_le).fit_transform(df),
    }

    danh_sach_dong = []
    for ten_pp, mang_gt in du_lieu_theo_phuong_phap.items():
        for chi_so, ten_cot in enumerate(cot_hop_le):
            cot_gt = mang_gt[:, chi_so]
            do_lech = float(pd.Series(cot_gt).skew())
            danh_sach_dong.append(
                {
                    "Phương pháp": ten_pp,
                    "Đặc trưng": ten_cot,
                    "Min": round(float(np.min(cot_gt)), 4),
                    "Max": round(float(np.max(cot_gt)), 4),
                    "Mean": round(float(np.mean(cot_gt)), 4),
                    "Std": round(float(np.std(cot_gt)), 4),
                    "Median": round(float(np.median(cot_gt)), 4),
                    "IQR": round(float(np.percentile(cot_gt, 75) - np.percentile(cot_gt, 25)), 4),
                    "Skew": 0.0 if np.isnan(do_lech) else round(do_lech, 4),
                }
            )

    bang_so_sanh = pd.DataFrame(danh_sach_dong)
    print(f" [SO SÁNH SCALER] Đã tổng hợp chỉ số cho {len(cot_hop_le)} đặc trưng qua 4 trạng thái thang đo (Raw, Standard, MinMax, Robust).")
    return bang_so_sanh


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
    cot_hop_le = [col for col in feature_cols if col in train_df.columns]
    if not cot_hop_le:
        raise KeyError(f"Không tìm thấy đặc trưng nào trong {feature_cols} để đánh giá scaler.")

    cac_phuong_phap = ["standard", "minmax", "robust"]
    danh_sach_dong = []
    so_diem_duy_nhat = len(np.unique(_extract_valid_matrix(train_df, cot_hop_le), axis=0))
    cac_k_hop_le = [int(k) for k in k_range if 2 <= int(k) < so_diem_duy_nhat]
    if not cac_k_hop_le:
        raise ValueError("Không đủ số điểm phân biệt trên tập Train để tìm cụm K >= 2.")

    for ten_pp in cac_phuong_phap:
        scaler = FeatureScaler(method=ten_pp, feature_cols=cot_hop_le).fit(train_df)
        x_train = scaler.transform(train_df)

        k_tot_nhat = cac_k_hop_le[0]
        sil_tot_nhat = -1.0
        mo_hinh_tot_nhat = None
        nhan_tot_nhat = None

        for k in cac_k_hop_le:
            km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
            nhan = km.fit_predict(x_train)
            sil = float(silhouette_score(x_train, nhan))
            if sil > sil_tot_nhat:
                sil_tot_nhat = sil
                k_tot_nhat = k
                mo_hinh_tot_nhat = km
                nhan_tot_nhat = nhan

        so_luong_cum = pd.Series(nhan_tot_nhat).value_counts()
        ty_le_can_bang = round(float(so_luong_cum.min() / max(so_luong_cum.max(), 1)), 4)

        sil_co_dinh = np.nan
        dbi_co_dinh = np.nan
        if 2 <= fixed_k < so_diem_duy_nhat:
            km_fixed = KMeans(
                n_clusters=fixed_k,
                init="k-means++",
                n_init=10,
                random_state=random_state,
            )
            nhan_fixed = km_fixed.fit_predict(x_train)
            sil_co_dinh = round(float(silhouette_score(x_train, nhan_fixed)), 4)
            dbi_co_dinh = round(float(davies_bouldin_score(x_train, nhan_fixed)), 4)

        sil_test = np.nan
        so_test_vuot_can = 0
        if test_df is not None and not test_df.empty:
            x_test = scaler.transform(test_df)
            nhan_test = mo_hinh_tot_nhat.predict(x_test)
            if len(np.unique(nhan_test)) >= 2 and len(x_test) > len(np.unique(nhan_test)):
                sil_test = round(float(silhouette_score(x_test, nhan_test)), 4)
            x_train_min = np.min(x_train, axis=0)
            x_train_max = np.max(x_train, axis=0)
            so_test_vuot_can = int(
                np.sum(np.any((x_test < x_train_min) | (x_test > x_train_max), axis=1))
            )

        danh_sach_dong.append(
            {
                "Scaler": ten_pp,
                "K tối ưu": int(k_tot_nhat),
                "Silhouette Train": round(sil_tot_nhat, 4),
                "DBI Train": round(float(davies_bouldin_score(x_train, nhan_tot_nhat)), 4),
                "CHI Train": round(float(calinski_harabasz_score(x_train, nhan_tot_nhat)), 2),
                "Inertia Train": round(float(mo_hinh_tot_nhat.inertia_), 2),
                "Cân bằng cụm": ty_le_can_bang,
                f"Silhouette K={fixed_k}": sil_co_dinh,
                f"DBI K={fixed_k}": dbi_co_dinh,
                "Silhouette Test": sil_test,
                "Mẫu Test vượt cận Train": so_test_vuot_can,
            }
        )

    bang_danh_gia = pd.DataFrame(danh_sach_dong)
    print(f" [ĐỐI CHIẾU SCALER TRÊN K-MEANS] Đã so sánh thực nghiệm 3 scaler qua dải K={cac_k_hop_le}.")
    return bang_danh_gia
