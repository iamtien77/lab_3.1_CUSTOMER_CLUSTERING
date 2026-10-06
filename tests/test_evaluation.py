"""
=============================================================================
KIỂM THỬ ĐƠN VỊ: MODULE ĐÁNH GIÁ PHÂN CỤM (TEST EVALUATION METRICS)
=============================================================================
"""

import unittest
import numpy as np


class TestEvaluationMetrics(unittest.TestCase):
    """
    Ý LÀM:
    1. test_silhouette_score_range: Kiểm tra Silhouette Score nằm trong khoảng [-1.0, 1.0].
    2. test_davies_bouldin_non_negative: Kiểm tra Davies-Bouldin Index luôn >= 0.0.
    3. test_calinski_harabasz_positive: Kiểm tra Calinski-Harabasz Score luôn >= 0.0.
    4. test_profiling_shapes: Kiểm tra bảng hồ sơ phân khúc tính đúng số dòng tương ứng số cụm.
    """
    pass


if __name__ == "__main__":
    unittest.main()
