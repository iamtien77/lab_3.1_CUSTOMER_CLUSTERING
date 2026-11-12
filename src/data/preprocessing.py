"""
=============================================================================
MODULE: TIỀN XỬ LÝ DỮ LIỆU CƠ BẢN (DATA PREPROCESSING & CLEANING)
=============================================================================
Mục đích:
- Tạo các đặc trưng cốt lõi từ marketing_campaign.csv (Age, Total_Spending).
- Làm sạch dữ liệu cơ bản, lọc ngoại lai phi lý (năm sinh < 1920, thu nhập lỗi),
  và điền giá trị thiếu (tránh Data Leakage khi truyền median từ tập Train).
- Kiểm tra chi tiết giá trị thiếu và giá trị ngoại lai bằng phương pháp IQR.
=============================================================================
"""

from typing import Dict, List, Optional
import pandas as pd

from src.config import (
    AGE_COLUMN,
    CATEGORICAL_PROFILE_COLUMNS,
    ID_COLUMN,
    INCOME_COLUMN,
    MAX_VALID_INCOME,
    MIN_VALID_YEAR_BIRTH,
    NUMERICAL_FEATURES,
    REFERENCE_YEAR,
    SPENDING_COLUMN,
    SPENDING_COMPONENT_COLUMNS,
    YEAR_BIRTH_COLUMN,
)


def add_derived_features(
    df: pd.DataFrame,
    reference_year: int = REFERENCE_YEAR,
) -> pd.DataFrame:
    """
    Tính toán hai đặc trưng phục vụ phân cụm: Age và Total_Spending (không sửa bản gốc).
    """
    ket_qua = df.copy()
    if YEAR_BIRTH_COLUMN in ket_qua.columns:
        nam_sinh = pd.to_numeric(ket_qua[YEAR_BIRTH_COLUMN], errors="coerce")
        ket_qua[AGE_COLUMN] = reference_year - nam_sinh

    if all(col in ket_qua.columns for col in SPENDING_COMPONENT_COLUMNS):
        for col in SPENDING_COMPONENT_COLUMNS:
            ket_qua[col] = pd.to_numeric(ket_qua[col], errors="coerce").fillna(0.0)
        ket_qua[SPENDING_COLUMN] = ket_qua[SPENDING_COMPONENT_COLUMNS].sum(axis=1)

    return ket_qua


def filter_invalid_records(
    df: pd.DataFrame,
    preserve_index: bool = False,
) -> pd.DataFrame:
    """
    Loại bỏ các bản ghi trùng ID hoặc mang giá trị sai lệch miền dữ liệu
    (năm sinh < 1920 hoặc > REFERENCE_YEAR, thu nhập lỗi, chi tiêu âm).
    Hàm chỉ lọc theo điều kiện cố định từng dòng nên không gây rò rỉ dữ liệu giữa Train và Test.
    """
    ket_qua = df.copy()
    so_ban_dau = len(ket_qua)

    if ID_COLUMN in ket_qua.columns:
        ket_qua = ket_qua.drop_duplicates(subset=[ID_COLUMN], keep="first")

    if YEAR_BIRTH_COLUMN in ket_qua.columns:
        nam_sinh = pd.to_numeric(ket_qua[YEAR_BIRTH_COLUMN], errors="coerce")
        ket_qua = ket_qua[
            nam_sinh.isna()
            | ((nam_sinh >= MIN_VALID_YEAR_BIRTH) & (nam_sinh <= REFERENCE_YEAR))
        ]
    elif AGE_COLUMN in ket_qua.columns:
        tuoi = pd.to_numeric(ket_qua[AGE_COLUMN], errors="coerce")
        tuoi_toi_da = REFERENCE_YEAR - MIN_VALID_YEAR_BIRTH
        ket_qua = ket_qua[tuoi.isna() | ((tuoi > 0) & (tuoi <= tuoi_toi_da))]

    if INCOME_COLUMN in ket_qua.columns:
        thu_nhap = pd.to_numeric(ket_qua[INCOME_COLUMN], errors="coerce")
        hop_le_thu_nhap = thu_nhap.isna() | (
            (thu_nhap >= 0) & (thu_nhap <= MAX_VALID_INCOME)
        )
        ket_qua = ket_qua[hop_le_thu_nhap]

    for col in SPENDING_COMPONENT_COLUMNS:
        if col in ket_qua.columns:
            chi_tieu = pd.to_numeric(ket_qua[col], errors="coerce")
            ket_qua = ket_qua[chi_tieu.isna() | (chi_tieu >= 0)]

    if SPENDING_COLUMN in ket_qua.columns:
        tong_chi = pd.to_numeric(ket_qua[SPENDING_COLUMN], errors="coerce")
        ket_qua = ket_qua[tong_chi.isna() | (tong_chi >= 0)]

    so_da_loai = so_ban_dau - len(ket_qua)
    if so_da_loai > 0:
        print(f" [LỌC BẢN GHI BẤT HỢP LÝ] Đã loại {so_da_loai} dòng nhiễu cực đoan/trùng lặp.")

    return ket_qua if preserve_index else ket_qua.reset_index(drop=True)


def clean_data(
    df: pd.DataFrame,
    fill_medians: Optional[Dict[str, float]] = None,
    drop_invalid_outliers: bool = True,
    derive_features: bool = True,
    preserve_index: bool = False,
) -> pd.DataFrame:
    """
    Làm sạch dữ liệu: ép kiểu số, lọc bản ghi lỗi miền (nếu bật), tạo Age/Total_Spending và điền giá trị thiếu.
    Khi chạy trên tập Test, cần truyền fill_medians tính từ tập Train để tránh Data Leakage.
    """
    ket_qua = (
        filter_invalid_records(df, preserve_index=preserve_index)
        if drop_invalid_outliers
        else df.copy()
    )
    if derive_features:
        ket_qua = add_derived_features(ket_qua)

    # Ép kiểu numeric an toàn cho các đặc trưng phân cụm
    for col in NUMERICAL_FEATURES:
        if col in ket_qua.columns:
            ket_qua[col] = pd.to_numeric(ket_qua[col], errors="coerce")

    # Điền giá trị khuyết ở các đặc trưng số bằng trung vị (từ Train hoặc từ chính df)
    for col in NUMERICAL_FEATURES:
        if col not in ket_qua.columns or not ket_qua[col].isnull().any():
            continue
        gia_tri_dien = (
            fill_medians[col]
            if fill_medians is not None and col in fill_medians
            else float(ket_qua[col].median())
        )
        if not pd.notna(gia_tri_dien):
            raise ValueError(
                f"Cột '{col}' chỉ chứa giá trị thiếu (NaN) và không có trung vị hợp lệ để điền."
            )
        so_o_thieu = int(ket_qua[col].isnull().sum())
        ket_qua[col] = ket_qua[col].fillna(gia_tri_dien)
        print(f" [XỬ LÝ KHUYẾT] Cột '{col}': đã điền {so_o_thieu} giá trị thiếu bằng trung vị ({gia_tri_dien:,.2f}).")

    # Điền mode cho các cột phân loại phục vụ mô tả hồ sơ khách hàng
    for col in CATEGORICAL_PROFILE_COLUMNS:
        if col in ket_qua.columns and ket_qua[col].isnull().any():
            gia_tri_pho_bien = ket_qua[col].mode(dropna=True)
            if not gia_tri_pho_bien.empty:
                ket_qua[col] = ket_qua[col].fillna(gia_tri_pho_bien.iloc[0])

    return ket_qua if preserve_index else ket_qua.reset_index(drop=True)


def compute_train_medians(
    train_df: pd.DataFrame,
    numerical_cols: List[str] = NUMERICAL_FEATURES,
) -> Dict[str, float]:
    """
    Tính trung vị của các cột số chỉ trên tập Train để dùng lại khi làm sạch tập Test.
    """
    df_features = add_derived_features(train_df)
    ket_qua: Dict[str, float] = {}
    for col in numerical_cols:
        if col in df_features.columns:
            series = pd.to_numeric(df_features[col], errors="coerce").dropna()
            if not series.empty:
                ket_qua[col] = float(series.median())
    return ket_qua


def audit_missing_and_outliers(
    df: pd.DataFrame,
    numerical_cols: List[str] = NUMERICAL_FEATURES,
) -> pd.DataFrame:
    """
    Kiểm toán giá trị khuyết (Null) và ngoại lai theo quy tắc IQR (Tukey's Fences) cho từng cột số.
    """
    df_kiem_toan = add_derived_features(df)
    tong_dong = max(len(df_kiem_toan), 1)
    danh_sach_bao_cao = []

    for col in numerical_cols:
        if col not in df_kiem_toan.columns:
            continue
        series = pd.to_numeric(df_kiem_toan[col], errors="coerce")
        null_count = int(series.isnull().sum())
        null_pct = round(null_count / tong_dong * 100.0, 2)

        hop_le = series.dropna()
        if hop_le.empty:
            continue

        q1 = float(hop_le.quantile(0.25))
        q3 = float(hop_le.quantile(0.75))
        iqr = q3 - q1
        nguong_duoi = q1 - 1.5 * iqr
        nguong_tren = q3 + 1.5 * iqr

        ngoai_lai = int(((hop_le < nguong_duoi) | (hop_le > nguong_tren)).sum())
        ty_le_ngoai_lai = round(ngoai_lai / len(hop_le) * 100.0, 2)

        danh_sach_bao_cao.append(
            {
                "Đặc trưng": col,
                "Null Count": null_count,
                "Null %": null_pct,
                "Q1": round(q1, 2),
                "Q3": round(q3, 2),
                "IQR": round(iqr, 2),
                "Ngưỡng dưới": round(nguong_duoi, 2),
                "Ngưỡng trên": round(nguong_tren, 2),
                "Số lượng Outlier": ngoai_lai,
                "Tỷ lệ Outlier %": ty_le_ngoai_lai,
            }
        )

    bao_cao_df = pd.DataFrame(danh_sach_bao_cao)
    print(f" [KIỂM TOÁN DỮ LIỆU] Đã hoàn tất thống kê Null & IQR Outliers cho {len(bao_cao_df)} đặc trưng.")
    return bao_cao_df
