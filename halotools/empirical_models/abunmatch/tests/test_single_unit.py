"""
"""
import numpy as np
from ..bin_free_cam import bin_free_conditional_abunmatch
from ....utils.conditional_percentile import cython_sliding_rank, rank_order_function


def test3():
    """ Test hard-coded case where x and x2 are sorted, y and y2 are sorted,
    but the nearest x--x2 correspondence is no longer simple
    """
    nwin = 3

    x = np.array([0.1,  0.36, 0.36, 0.74, 0.83])
    x2 = np.array([0.54, 0.54, 0.55, 0.56, 0.57])

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.03, 0.54, 0.67, 0.73, 0.86])

    i2_matched = np.searchsorted(x2, x)
    i2_matched = np.where(i2_matched >= len(y2), len(y2)-1, i2_matched)
    i2_matched = np.array([0, 0, 0, 4, 4])

    result = bin_free_conditional_abunmatch(x, y, x2, y2, nwin)
    correct_result = [0.03, 0.54, 0.54, 0.73, 0.86]

    assert np.allclose(result, correct_result)
