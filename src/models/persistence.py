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
    target_dir = Path(output_dir) if output_dir is not None else SAVED_MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / model_name

    model_bundle = {
        "model": model_object,
        "scaler": scaler_object,
        "metadata": metadata or {},
    }
    joblib.dump(model_bundle, model_path)
    print(f" [ĐÃ LƯU MÔ HÌNH] Gói pipeline phân cụm đã lưu tại: {model_path}")
    return model_path


def load_model(
    model_name: str = "final_kmeans_pipeline.pkl",
    input_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Kiểm tra sự tồn tại và nạp gói mô hình ('model', 'scaler', 'metadata') từ ổ đĩa.
    """
    target_dir = Path(input_dir) if input_dir is not None else SAVED_MODELS_DIR
    model_path = target_dir / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp mô hình đã lưu tại: {model_path}")

    model_bundle = joblib.load(model_path)
    if not isinstance(model_bundle, dict) or not {"model", "scaler", "metadata"}.issubset(model_bundle.keys()):
        raise ValueError(f"Tệp '{model_path}' không đúng cấu trúc gói pipeline mong đợi.")

    print(f" [ĐÃ NẠP MÔ HÌNH] Tải thành công gói pipeline từ: {model_path.name}")
    return model_bundle
