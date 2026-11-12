"""
=============================================================================
MODULE: TRIỂN KHAI VÀ DỰ ĐOÁN PHÂN KHÚC KHÁCH HÀNG MỚI (INFERENCE / PREDICTOR)
=============================================================================
Mục đích (mức độ bài lab học tập - CLI/Python):
- Nạp gói mô hình hoàn chỉnh đã lưu từ thư mục results/saved_models/.
- Tự động chuẩn hóa dữ liệu đầu vào bằng bộ Scaler đã fit trên tập Train.
- Gán nhãn cụm, xác định tên Persona và tính khoảng cách tới tâm cụm đại diện.
=============================================================================
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from src.config import AGE_COLUMN, INCOME_COLUMN, SPENDING_COLUMN
from src.data.preprocessing import clean_data
from src.models.persistence import load_model


class CustomerSegmentPredictor:
    """
    Lớp suy luận phân khúc cho khách hàng mới dựa trên pipeline đã huấn luyện.
    """

    def __init__(
        self,
        model_filename: str = "final_kmeans_pipeline.pkl",
        model_dir: Optional[Path | str] = None,
    ):
        """Nạp gói mô hình gồm model, scaler và metadata chứa định nghĩa Persona."""
        goi_mo_hinh = load_model(model_name=model_filename, input_dir=model_dir)
        self.model = goi_mo_hinh["model"]
        if not hasattr(self.model, "predict"):
            raise TypeError("Đối tượng mô hình nạp vào không hỗ trợ phương thức predict().")
        self.scaler = goi_mo_hinh["scaler"]
        self.metadata = goi_mo_hinh.get("metadata", {})
        self._personas_map = self._chuan_hoa_bang_persona(self.metadata.get("personas", {}))

    @staticmethod
    def _chuan_hoa_bang_persona(raw_personas: Any) -> Dict[int, Dict[str, Any]]:
        """Chuyển đổi danh sách hoặc từ điển Persona trong metadata sang map theo cluster_id."""
        if isinstance(raw_personas, dict):
            return {int(k): v for k, v in raw_personas.items()}
        if isinstance(raw_personas, list):
            return {int(p["cluster_id"]): p for p in raw_personas if "cluster_id" in p}
        return {}

    def _tra_cuu_persona(self, cluster_id: int) -> tuple[str, str]:
        """Lấy tên Persona và khuyến nghị tiếp thị tương ứng với mã cụm."""
        thong_tin = self._personas_map.get(int(cluster_id))
        if not thong_tin:
            return f"Cụm {cluster_id}", "Chưa có mô tả Persona riêng trong metadata."
        ten = thong_tin.get("persona_name", f"Cụm {cluster_id}")
        chien_luoc = thong_tin.get("marketing_strategy", "")
        return ten, chien_luoc

    def predict_single_customer(
        self,
        age: float,
        annual_income: float,
        total_spending: float,
    ) -> Dict[str, Any]:
        """
        Dự đoán phân khúc và khoảng cách tới tâm cụm cho 1 khách hàng mới.
        """
        if not np.all(np.isfinite([age, annual_income, total_spending])):
            raise ValueError("Age, Income và Total_Spending phải là các giá trị số hữu hạn (không chứa NaN/Inf).")
        if age <= 0 or age > 120:
            raise ValueError(f"Độ tuổi không hợp lệ: {age}. Phải nằm trong khoảng (0, 120].")
        if annual_income < 0 or total_spending < 0:
            raise ValueError("Thu nhập và tổng chi tiêu không được âm.")

        df_mot_dong = pd.DataFrame(
            [
                {
                    AGE_COLUMN: float(age),
                    INCOME_COLUMN: float(annual_income),
                    SPENDING_COLUMN: float(total_spending),
                }
            ]
        )
        ma_tran_scaled = self.scaler.transform(df_mot_dong)
        cluster_id = int(self.model.predict(ma_tran_scaled)[0])

        khoang_cach = float("nan")
        if hasattr(self.model, "cluster_centers_"):
            tam_cum = self.model.cluster_centers_[cluster_id]
            delta = ma_tran_scaled[0] - tam_cum
            if getattr(self.model, "_scipy_metric", None) == "cityblock":
                khoang_cach = round(float(np.sum(np.abs(delta))), 4)
            else:
                khoang_cach = round(float(np.linalg.norm(delta)), 4)

        ten_persona, khuyen_nghi = self._tra_cuu_persona(cluster_id)
        return {
            "customer_input": {
                AGE_COLUMN: float(age),
                INCOME_COLUMN: float(annual_income),
                SPENDING_COLUMN: float(total_spending),
            },
            "assigned_cluster": cluster_id,
            "persona_name": ten_persona,
            "marketing_recommendation": khuyen_nghi,
            "distance_to_center": khoang_cach,
        }

    def predict_batch(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Dự đoán phân khúc hàng loạt cho bảng khách hàng mới (tự tính Age/Total_Spending và điền thiếu an toàn).
        """
        if df_new.empty:
            raise ValueError("DataFrame khách hàng mới đang rỗng, không thể dự đoán.")

        # Dùng trung vị đã lưu từ tập Train (nếu có) để điền thiếu mà không gây Data Leakage
        trung_vi_train = self.metadata.get("feature_medians")
        df_ket_qua = clean_data(
            df_new,
            fill_medians=trung_vi_train,
            drop_invalid_outliers=False,
            derive_features=True,
            preserve_index=True,
        )
        ma_tran_scaled = self.scaler.transform(df_ket_qua)
        nhan_du_doan = self.model.predict(ma_tran_scaled)

        ds_ten_persona = []
        ds_khuyen_nghi = []
        for cum_id in nhan_du_doan:
            ten, chien_luoc = self._tra_cuu_persona(int(cum_id))
            ds_ten_persona.append(ten)
            ds_khuyen_nghi.append(chien_luoc)

        df_ket_qua["Predicted_Cluster"] = nhan_du_doan.astype(int)
        df_ket_qua["Persona_Name"] = ds_ten_persona
        df_ket_qua["Marketing_Recommendation"] = ds_khuyen_nghi
        print(f" [SUY LUẬN HÀNG LOẠT] Đã gắn nhãn phân khúc cho {len(df_ket_qua):,} khách hàng mới.")
        return df_ket_qua


def run_cli():
    """Giao diện dòng lệnh (CLI) mức bài lab để dự đoán phân khúc 1 khách hàng mới."""
    parser = argparse.ArgumentParser(description="Dự đoán phân khúc cho 1 khách hàng mới từ mô hình đã lưu.")
    parser.add_argument("--age", type=float, required=True, help="Độ tuổi khách hàng (năm)")
    parser.add_argument("--income", type=float, required=True, help="Thu nhập hộ gia đình hàng năm (USD)")
    parser.add_argument("--spending", type=float, required=True, help="Tổng chi tiêu (USD)")
    parser.add_argument("--model", type=str, default="final_kmeans_pipeline.pkl", help="Tên tệp mô hình trong results/saved_models/")
    parser.add_argument("--model-dir", type=str, default=None, help="Thư mục tùy chọn chứa tệp mô hình (.pkl)")
    args = parser.parse_args()

    predictor = CustomerSegmentPredictor(model_filename=args.model, model_dir=args.model_dir)
    ket_qua = predictor.predict_single_customer(
        age=args.age,
        annual_income=args.income,
        total_spending=args.spending,
    )
    print(f"Cụm dự đoán: {ket_qua['assigned_cluster']} - {ket_qua['persona_name']}")
    print(f"Khoảng cách tới tâm cụm: {ket_qua['distance_to_center']}")
    print(f"Chiến lược gợi ý: {ket_qua['marketing_recommendation']}")


if __name__ == "__main__":
    run_cli()
