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
    result_df = df.copy()
    if YEAR_BIRTH_COLUMN in result_df.columns:
        year_birth = pd.to_numeric(result_df[YEAR_BIRTH_COLUMN], errors="coerce")
        result_df[AGE_COLUMN] = reference_year - year_birth

    if all(col in result_df.columns for col in SPENDING_COMPONENT_COLUMNS):
        for col in SPENDING_COMPONENT_COLUMNS:
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce").fillna(0.0)
        result_df[SPENDING_COLUMN] = result_df[SPENDING_COMPONENT_COLUMNS].sum(axis=1)

    return result_df


def filter_invalid_records(
    df: pd.DataFrame,
    preserve_index: bool = False,
) -> pd.DataFrame:
    """
    Loại bỏ các bản ghi trùng ID hoặc mang giá trị sai lệch miền dữ liệu
    (năm sinh < 1920 hoặc > REFERENCE_YEAR, thu nhập lỗi, chi tiêu âm).
    Hàm chỉ lọc theo điều kiện cố định từng dòng nên không gây rò rỉ dữ liệu giữa Train và Test.
    """
    result_df = df.copy()
    initial_count = len(result_df)

    if ID_COLUMN in result_df.columns:
        result_df = result_df.drop_duplicates(subset=[ID_COLUMN], keep="first")

    if YEAR_BIRTH_COLUMN in result_df.columns:
        year_birth = pd.to_numeric(result_df[YEAR_BIRTH_COLUMN], errors="coerce")
        result_df = result_df[
            year_birth.isna()
            | ((year_birth >= MIN_VALID_YEAR_BIRTH) & (year_birth <= REFERENCE_YEAR))
        ]
    elif AGE_COLUMN in result_df.columns:
        age = pd.to_numeric(result_df[AGE_COLUMN], errors="coerce")
        max_valid_age = REFERENCE_YEAR - MIN_VALID_YEAR_BIRTH
        result_df = result_df[age.isna() | ((age > 0) & (age <= max_valid_age))]

    if INCOME_COLUMN in result_df.columns:
        income = pd.to_numeric(result_df[INCOME_COLUMN], errors="coerce")
        valid_income_mask = income.isna() | (
            (income >= 0) & (income <= MAX_VALID_INCOME)
        )
        result_df = result_df[valid_income_mask]

    for col in SPENDING_COMPONENT_COLUMNS:
        if col in result_df.columns:
            spending = pd.to_numeric(result_df[col], errors="coerce")
            result_df = result_df[spending.isna() | (spending >= 0)]

    if SPENDING_COLUMN in result_df.columns:
        total_spending = pd.to_numeric(result_df[SPENDING_COLUMN], errors="coerce")
        result_df = result_df[total_spending.isna() | (total_spending >= 0)]

    removed_count = initial_count - len(result_df)
    if removed_count > 0:
        print(f" [LỌC BẢN GHI BẤT HỢP LÝ] Đã loại {removed_count} dòng nhiễu cực đoan/trùng lặp.")

    return result_df if preserve_index else result_df.reset_index(drop=True)


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
    result_df = (
        filter_invalid_records(df, preserve_index=preserve_index)
        if drop_invalid_outliers
        else df.copy()
    )
    if derive_features:
        result_df = add_derived_features(result_df)

    # Ép kiểu numeric an toàn cho các đặc trưng phân cụm
    for col in NUMERICAL_FEATURES:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce")

    # Điền giá trị khuyết ở các đặc trưng số bằng trung vị (từ Train hoặc từ chính df)
    for col in NUMERICAL_FEATURES:
        if col not in result_df.columns or not result_df[col].isnull().any():
            continue
        fill_value = (
            fill_medians[col]
            if fill_medians is not None and col in fill_medians
            else float(result_df[col].median())
        )
        if not pd.notna(fill_value):
            raise ValueError(
                f"Cột '{col}' chỉ chứa giá trị thiếu (NaN) và không có trung vị hợp lệ để điền."
            )
        missing_count = int(result_df[col].isnull().sum())
        result_df[col] = result_df[col].fillna(fill_value)
        print(f" [XỬ LÝ KHUYẾT] Cột '{col}': đã điền {missing_count} giá trị thiếu bằng trung vị ({fill_value:,.2f}).")

    # Điền mode cho các cột phân loại phục vụ mô tả hồ sơ khách hàng
    for col in CATEGORICAL_PROFILE_COLUMNS:
        if col in result_df.columns and result_df[col].isnull().any():
            mode_values = result_df[col].mode(dropna=True)
            if not mode_values.empty:
                result_df[col] = result_df[col].fillna(mode_values.iloc[0])

    return result_df if preserve_index else result_df.reset_index(drop=True)


def compute_train_medians(
    train_df: pd.DataFrame,
    numerical_cols: List[str] = NUMERICAL_FEATURES,
) -> Dict[str, float]:
    """
    Tính trung vị của các cột số chỉ trên tập Train để dùng lại khi làm sạch tập Test.
    """
    df_features = add_derived_features(train_df)
    medians: Dict[str, float] = {}
    for col in numerical_cols:
        if col in df_features.columns:
            series = pd.to_numeric(df_features[col], errors="coerce").dropna()
            if not series.empty:
                medians[col] = float(series.median())
    return medians


def audit_missing_and_outliers(
    df: pd.DataFrame,
    numerical_cols: List[str] = NUMERICAL_FEATURES,
) -> pd.DataFrame:
    """
    Kiểm toán giá trị khuyết (Null) và ngoại lai theo quy tắc IQR (Tukey's Fences) cho từng cột số.
    """
    audited_df = add_derived_features(df)
    total_rows = max(len(audited_df), 1)
    report_rows = []

    for col in numerical_cols:
        if col not in audited_df.columns:
            continue
        series = pd.to_numeric(audited_df[col], errors="coerce")
        null_count = int(series.isnull().sum())
        null_pct = round(null_count / total_rows * 100.0, 2)

        valid_series = series.dropna()
        if valid_series.empty:
            continue

        q1 = float(valid_series.quantile(0.25))
        q3 = float(valid_series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = int(((valid_series < lower_bound) | (valid_series > upper_bound)).sum())
        outlier_pct = round(outlier_count / len(valid_series) * 100.0, 2)

        report_rows.append(
            {
                "Đặc trưng": col,
                "Null Count": null_count,
                "Null %": null_pct,
                "Q1": round(q1, 2),
                "Q3": round(q3, 2),
                "IQR": round(iqr, 2),
                "Ngưỡng dưới": round(lower_bound, 2),
                "Ngưỡng trên": round(upper_bound, 2),
                "Số lượng Outlier": outlier_count,
                "Tỷ lệ Outlier %": outlier_pct,
            }
        )

    report_df = pd.DataFrame(report_rows)
    print(f" [KIỂM TOÁN DỮ LIỆU] Đã hoàn tất thống kê Null & IQR Outliers cho {len(report_df)} đặc trưng.")
    return report_df
