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
        dong_dau = f.readline()
    if "\t" in dong_dau:
        return "\t"
    return "," if "," in dong_dau else DATA_SEPARATOR


def load_raw_data(
    file_path: Path | str | None = None,
    sep: str | None = None,
) -> pd.DataFrame:
    """
    Nạp tập dữ liệu thô từ ổ đĩa, chuẩn hóa tên cột và giữ nguyên dữ liệu gốc.
    """
    duong_dan = Path(file_path) if file_path is not None else DEFAULT_DATA_PATH
    if not duong_dan.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu tại: {duong_dan}")

    duoi_file = duong_dan.suffix.lower()
    if duoi_file in {".xlsx", ".xls"}:
        df = pd.read_excel(duong_dan)
    else:
        dau_tach = sep if sep is not None else _detect_separator(duong_dan)
        df = pd.read_csv(duong_dan, sep=dau_tach)

    # Xóa khoảng trắng thừa ở tên cột để tránh lỗi truy xuất key
    df.columns = df.columns.astype(str).str.strip()
    print(f" [NẠP DỮ LIỆU] Đã đọc thành công {len(df):,} dòng x {df.shape[1]} cột từ {duong_dan.name}")
    return df


def validate_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Kiểm tra cấu trúc cột bắt buộc, dòng trùng lặp, tính duy nhất của ID và giá trị thiếu.
    """
    cot_thieu = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
    so_dong_trung = int(df.duplicated().sum())
    so_id_thieu = int(df[ID_COLUMN].isnull().sum()) if ID_COLUMN in df.columns else -1
    so_id_trung = (
        int(df[ID_COLUMN].dropna().duplicated().sum())
        if ID_COLUMN in df.columns
        else -1
    )

    thieu_theo_cot = {
        col: int(cnt) for col, cnt in df.isnull().sum().items() if cnt > 0
    }
    tong_o_thieu = int(sum(thieu_theo_cot.values()))
    hop_le = (
        len(cot_thieu) == 0
        and so_id_trung == 0
        and so_id_thieu == 0
        and len(df) > 0
    )

    ket_qua = {
        "is_valid": hop_le,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_columns": cot_thieu,
        "duplicate_rows": so_dong_trung,
        "duplicate_ids": so_id_trung,
        "missing_ids": so_id_thieu,
        "total_nulls": tong_o_thieu,
        "null_by_column": thieu_theo_cot,
    }

    trang_thai = "HỢP LỆ" if hop_le else "KHÔNG HỢP LỆ"
    print(
        f" [KIỂM ĐỊNH SCHEMA] Trạng thái: {trang_thai} | "
        f"Dòng: {ket_qua['n_rows']:,} | Cột thiếu: {len(cot_thieu)} | "
        f"ID trùng: {so_id_trung} | ID thiếu: {so_id_thieu} | Giá trị khuyết: {tong_o_thieu}"
    )
    return ket_qua


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
        thu_muc_luu = Path(output_dir)
        thu_muc_luu.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(thu_muc_luu / "train.csv", index=False)
        test_df.to_csv(thu_muc_luu / "test.csv", index=False)
        df.reset_index(drop=True).to_csv(thu_muc_luu / "full_customers.csv", index=False)
        print(f" [ĐÃ LƯU TẬP DỮ LIỆU] Train/Test/Full đã được lưu tại: {thu_muc_luu}")

    print(
        f" [CHIA TẬP 80/20] Tổng: {len(df):,} -> "
        f"Train: {len(train_df):,} ({(1 - test_size) * 100:.0f}%) | "
        f"Test: {len(test_df):,} ({test_size * 100:.0f}%)"
    )
    return train_df, test_df
