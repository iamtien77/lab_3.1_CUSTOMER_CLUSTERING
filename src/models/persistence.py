"""
=============================================================================
MODULE: LƯU TRỮ VÀ TẢI MÔ HÌNH (MODEL PERSISTENCE)
=============================================================================
Mục đích:
- Đóng gói toàn bộ pipeline (Mô hình phân cụm + Scaler + Metadata) bằng joblib
  vào thư mục results/saved_models/.
- Nạp lại gói mô hình để phục vụ suy luận trên tập Test hoặc khách hàng mới.
=============================================================================
"""

from pathlib import Path
from typing import Any, Dict, Optional
import joblib

from src.config import SAVED_MODELS_DIR


def save_model(
    model_object: Any,
    scaler_object: Any,
    model_name: str = "final_kmeans_pipeline.pkl",
    metadata: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path | str] = None,
) -> Path:
    """
    Đóng gói model, scaler và metadata thành file .pkl bằng joblib.
    """
    thu_muc = Path(output_dir) if output_dir is not None else SAVED_MODELS_DIR
    thu_muc.mkdir(parents=True, exist_ok=True)
    duong_dan = thu_muc / model_name

    goi_mo_hinh = {
        "model": model_object,
        "scaler": scaler_object,
        "metadata": metadata or {},
    }
    joblib.dump(goi_mo_hinh, duong_dan)
    print(f" [ĐÃ LƯU MÔ HÌNH] Gói pipeline phân cụm đã lưu tại: {duong_dan}")
    return duong_dan


def load_model(
    model_name: str = "final_kmeans_pipeline.pkl",
    input_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Kiểm tra sự tồn tại và nạp gói mô hình ('model', 'scaler', 'metadata') từ ổ đĩa.
    """
    thu_muc = Path(input_dir) if input_dir is not None else SAVED_MODELS_DIR
    duong_dan = thu_muc / model_name
    if not duong_dan.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp mô hình đã lưu tại: {duong_dan}")

    goi_mo_hinh = joblib.load(duong_dan)
    if not isinstance(goi_mo_hinh, dict) or not {"model", "scaler", "metadata"}.issubset(goi_mo_hinh.keys()):
        raise ValueError(f"Tệp '{duong_dan}' không đúng cấu trúc gói pipeline mong đợi.")

    print(f" [ĐÃ NẠP MÔ HÌNH] Tải thành công gói pipeline từ: {duong_dan.name}")
    return goi_mo_hinh
