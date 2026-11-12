"""
=============================================================================
MODULE: NẠP VÀ PHÂN CHIA DỮ LIỆU (DATA LOADER)
=============================================================================
Mục đích:
- Đọc file dữ liệu thô (marketing_campaign.csv) từ thư mục data/raw/.
- Kiểm tra tính toàn vẹn cấu trúc (schema validation).
- Phân chia tập Train / Test (80/20) và lưu trữ vào thư mục data/processed/.
=============================================================================
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_SEPARATOR,
    DEFAULT_DATA_PATH,
    ID_COLUMN,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    REQUIRED_RAW_COLUMNS,
    TRAIN_TEST_SPLIT_RATIO,
)


def _detect_separator(path: Path) -> str:
    """Tự động nhận diện dấu phân tách cột (tab hoặc dấu phẩy) từ dòng tiêu đề."""
    if path.suffix.lower() in {".tsv", ".txt"}:
        return "\t"
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    if "\t" in first_line:
        return "\t"
    return "," if "," in first_line else DATA_SEPARATOR


def load_raw_data(
    file_path: Path | str | None = None,
    sep: str | None = None,
) -> pd.DataFrame:
    """
    Nạp tập dữ liệu thô từ ổ đĩa, chuẩn hóa tên cột và giữ nguyên dữ liệu gốc.
    """
    resolved_path = Path(file_path) if file_path is not None else DEFAULT_DATA_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu tại: {resolved_path}")

    file_ext = resolved_path.suffix.lower()
    if file_ext in {".xlsx", ".xls"}:
        df = pd.read_excel(resolved_path)
    else:
        delimiter = sep if sep is not None else _detect_separator(resolved_path)
        df = pd.read_csv(resolved_path, sep=delimiter)

    # Xóa khoảng trắng thừa ở tên cột để tránh lỗi truy xuất key
    df.columns = df.columns.astype(str).str.strip()
    print(f" [NẠP DỮ LIỆU] Đã đọc thành công {len(df):,} dòng x {df.shape[1]} cột từ {resolved_path.name}")
    return df


def validate_schema(df: pd.DataFrame, allow_duplicate_ids: bool = False) -> Dict[str, Any]:
    """
    Kiểm tra cấu trúc cột bắt buộc, dòng trùng lặp, tính duy nhất của ID và giá trị thiếu.
    allow_duplicate_ids=True chỉ dùng trước bước loại trùng; vẫn chặn thiếu cột/ID.
    """
    missing_columns = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
    duplicate_rows = int(df.duplicated().sum())
    missing_ids = int(df[ID_COLUMN].isnull().sum()) if ID_COLUMN in df.columns else -1
    duplicate_ids = (
        int(df[ID_COLUMN].dropna().duplicated().sum())
        if ID_COLUMN in df.columns
        else -1
    )

    null_by_column = {
        col: int(count) for col, count in df.isnull().sum().items() if count > 0
    }
    total_nulls = int(sum(null_by_column.values()))
    is_valid = (
        len(missing_columns) == 0
        and (allow_duplicate_ids or duplicate_ids == 0)
        and missing_ids == 0
        and len(df) > 0
    )

    validation_result = {
        "is_valid": is_valid,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_columns": missing_columns,
        "duplicate_rows": duplicate_rows,
        "duplicate_ids": duplicate_ids,
        "missing_ids": missing_ids,
        "total_nulls": total_nulls,
        "null_by_column": null_by_column,
    }

    status_label = "HỢP LỆ" if is_valid else "KHÔNG HỢP LỆ"
    print(
        f" [KIỂM ĐỊNH SCHEMA] Trạng thái: {status_label} | "
        f"Dòng: {validation_result['n_rows']:,} | Cột thiếu: {len(missing_columns)} | "
        f"ID trùng: {duplicate_ids} | ID thiếu: {missing_ids} | Giá trị khuyết: {total_nulls}"
    )
    return validation_result


def split_data(
    df: pd.DataFrame,
    test_size: float = TRAIN_TEST_SPLIT_RATIO,
    random_state: int = RANDOM_SEED,
    output_dir: Optional[Path | str] = PROCESSED_DATA_DIR,
    save_files: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chia tập dữ liệu thành tập Huấn luyện (80%) và tập Kiểm chứng Holdout (20%).
    Reset index và lưu ra CSV để phục vụ tái lập mà không gây rò rỉ dữ liệu.
    """
    if df.empty:
        raise ValueError("Không thể phân chia dữ liệu vì DataFrame đầu vào đang rỗng.")
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"Tỷ lệ test_size phải nằm trong (0, 1), nhận được: {test_size}")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if save_files and output_dir is not None:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(target_dir / "train.csv", index=False)
        test_df.to_csv(target_dir / "test.csv", index=False)
        df.reset_index(drop=True).to_csv(target_dir / "full_customers.csv", index=False)
        print(f" [ĐÃ LƯU TẬP DỮ LIỆU] Train/Test/Full đã được lưu tại: {target_dir}")

    print(
        f" [CHIA TẬP 80/20] Tổng: {len(df):,} -> "
        f"Train: {len(train_df):,} ({(1 - test_size) * 100:.0f}%) | "
        f"Test: {len(test_df):,} ({test_size * 100:.0f}%)"
    )
    return train_df, test_df
