"""
=============================================================================
MODULE: TIỆN ÍCH GHI NHẬT KÝ (LOGGING UTILITIES)
=============================================================================
Cung cấp logger cấu hình chuẩn ghi nhật ký thực thi đồng thời ra Terminal
và lưu trữ vào file trong thư mục results/metrics/run.log.
=============================================================================
"""

import sys
import logging
from pathlib import Path
from src.config import METRICS_DIR


def get_logger(name: str = "CustomerClustering") -> logging.Logger:
    """
    Khởi tạo và cấu hình Logger chuẩn cho toàn bộ hệ thống.

    Logic thực hiện:
    - Tạo logger với định dạng: [Thời gian] [Cấp độ] [Tên module] - Nội dung thông điệp.
    - Cấu hình 2 Handler:
      1. StreamHandler: Hiển thị thông báo trực tiếp ra màn hình Console.
      2. FileHandler: Ghi toàn bộ lịch sử chạy vào file results/metrics/run.log.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Tránh gắn trùng lặp handler nếu hàm được gọi nhiều lần
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 1. Handler xuất ra màn hình Terminal
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. Handler ghi vào file log trên đĩa
        log_file = METRICS_DIR / "pipeline_execution.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
