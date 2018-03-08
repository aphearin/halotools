"""
"""
import numpy as np
from ....utils.conditional_percentile import cython_sliding_rank


def pure_python_rank_matching(x_sample1, ranks_sample1,
            x_sample2, ranks_sample2, y_sample2, nwin):
    """
    """
    result = np.zeros_like(x_sample1)
    n1 = len(x_sample1)
    n2 = len(x_sample2)
    nhalfwin = int(nwin/2)
    for i in range(n1):
        idx2 = np.searchsorted(x_sample2, x_sample1[i])
        low = idx2-nhalfwin
        high = idx2+nhalfwin+1

        if low < 0:
            low, high = 0, nwin
        elif high >= n2:
            low, high = n2-nwin, n2

        window = y_sample2[low:high]

        idx_sorted_window = np.argsort(window)
        result[i] = window[idx_sorted_window[ranks_sample1[i]]]
    return result


def test_pure_python1():
    """
    """
    n1, n2, nwin = 5001, 1001, 11
    nhalfwin = nwin/2
    x_sample1 = np.linspace(0, 1, n1)
    y_sample1 = np.random.uniform(0, 1, n1)
    ranks_sample1 = cython_sliding_rank(x_sample1, y_sample1, nwin)

    x_sample2 = np.linspace(0, 1, n2)
    y_sample2 = np.random.uniform(-4, -3, n2)
    ranks_sample2 = cython_sliding_rank(x_sample2, y_sample2, nwin)

    result = pure_python_rank_matching(x_sample1, ranks_sample1,
            x_sample2, ranks_sample2, y_sample2, nwin)

    for ix1 in range(n1):

        x1 = x_sample1[ix1]
        rank1 = ranks_sample1[ix1]

        ix2 = np.searchsorted(x_sample2, x1)

        low = ix2-nhalfwin
        high = ix2+nhalfwin+1

        if low < 0:
            low, high = 0, nwin
        elif high >= n2:
            low, high = n2-nwin, n2

        sorted_window2 = np.sort(y_sample2[low:high])
        correct_result_ix1 = sorted_window2[rank1]
        assert correct_result_ix1 == result[ix1]