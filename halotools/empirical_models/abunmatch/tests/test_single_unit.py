"""
"""
import numpy as np
from ..bin_free_cam import bin_free_conditional_abunmatch
from ....utils.conditional_percentile import cython_sliding_rank, rank_order_function


def test3():
    """ Test case where x and x2 are sorted, y and y2 are sorted,
    but the nearest x--x2 correspondence is no longer simple
    """
    nwin = 3
    nhalfwin = int(nwin/2)

    x = np.array([0.1,  0.36, 0.36, 0.74, 0.83])
    x2 = np.array([0.54, 0.54, 0.55, 0.56, 0.57])

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.03, 0.54, 0.67, 0.73, 0.86])

    i2_matched = np.searchsorted(x2, x)
    i2_matched = np.where(i2_matched >= len(y2), len(y2)-1, i2_matched)

    result = bin_free_conditional_abunmatch(x, y, x2, y2, nwin)

    #  Test all points except edges
    for itest in range(nhalfwin, len(x)-nhalfwin):
        low = itest-nhalfwin
        high = itest+nhalfwin+1
        window = y[low:high]
        window2 = y2[low:high]
        sorted_window2 = np.sort(window2)
        window_ranks = rank_order_function(window)
        itest_rank = window_ranks[nhalfwin]
        itest_correct_result = sorted_window2[itest_rank]
        itest_result = result[itest]
        assert itest_result == itest_correct_result
