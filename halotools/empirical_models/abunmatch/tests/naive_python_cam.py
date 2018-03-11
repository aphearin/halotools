""" Naive python implementation of bin-free conditional abundance matching
"""
import numpy as np


def sample2_window_indices(x1, x_sample2, nwin):
    nhalfwin = int(nwin/2)
    npts2 = len(x_sample2)

    iy2 = min(np.searchsorted(x_sample2, x1), len(x_sample2)-1)

    init_iy2_low = iy2 - nhalfwin
    init_iy2_high = init_iy2_low+nwin

    if init_iy2_low < 0:
        init_iy2_low = 0
        init_iy2_high = init_iy2_low + nwin
        iy2 = init_iy2_low + nhalfwin
    elif init_iy2_high > npts2 - nhalfwin:
        init_iy2_high = npts2
        init_iy2_low = init_iy2_high - nwin
        iy2 = init_iy2_low + nhalfwin

    return init_iy2_low, init_iy2_high


def pure_python_rank_matching(x_sample1, ranks_sample1,
            x_sample2, ranks_sample2, y_sample2, nwin):
    """
    """
    result = np.zeros_like(x_sample1)

    n1 = len(x_sample1)
    n2 = len(x_sample2)
    nhalfwin = int(nwin/2)

    for i in range(n1):
        x1 = x_sample1[i]
        low, high = sample2_window_indices(x1, x_sample2, nwin)

        sorted_window = np.sort(y_sample2[low:high])
        rank1 = ranks_sample1[i]
        result[i] = sorted_window[rank1]

    return result
