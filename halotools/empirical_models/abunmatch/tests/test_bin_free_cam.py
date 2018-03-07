"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from ..bin_free_cam import bin_free_conditional_abunmatch
from ....utils.conditional_percentile import cython_sliding_rank, rank_order_function


fixed_seed = 43


def test1():
    nwin = 3

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    x2 = x

    y = np.arange(1, len(x)+1)
    y2 = y*10.

    i2_matched = np.searchsorted(x2, x)
    i2_matched = np.where(i2_matched >= len(y2), len(y2)-1, i2_matched)

    print("y  = {0}".format(y))
    print("y2 = {0}\n".format(y2))

    result = bin_free_conditional_abunmatch(x, y, x2, y2, nwin)
    print("ynew  = {0}".format(result.astype('i4')))

    assert np.all(result == y2)


def test2():
    nwin = 3
    nhalfwin = int(nwin/2)

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    x2 = x+0.01

    with NumpyRNGContext(fixed_seed):
        y = np.round(np.random.rand(len(x)), 2)
        y2 = np.round(np.random.rand(len(x2)), 2)

    i2_matched = np.searchsorted(x2, x)
    i2_matched = np.where(i2_matched >= len(y2), len(y2)-1, i2_matched)

    print("y  = {0}".format(y))
    print("y2 = {0}\n".format(y2))

    print("ranks1  = {0}".format(cython_sliding_rank(x, y, nwin)))
    print("ranks2  = {0}".format(cython_sliding_rank(x2, y2, nwin)))

    result = bin_free_conditional_abunmatch(x, y, x2, y2, nwin)

    print("\n\nynew  = {0}".format(np.abs(result)))
    print("y2    = {0}".format(y2))
    print("y     = {0}\n".format(y))

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

    #  Test left edge
    for itest in range(nhalfwin):
        low, high = 0, nwin
        window = y[low:high]
        window2 = y2[low:high]
        sorted_window2 = np.sort(window2)
        window_ranks = rank_order_function(window)
        itest_rank = window_ranks[itest]
        itest_correct_result = sorted_window2[itest_rank]
        itest_result = result[itest]
        msg = "itest_result = {0}, correct result = {1}"
        assert itest_result == itest_correct_result, msg.format(
            itest_result, itest_correct_result)

    #  Test right edge
    for iwin in range(nhalfwin+1, nwin):
        itest = iwin + len(x) - nwin
        low, high = len(x)-nwin, len(x)
        window = y[low:high]
        window2 = y2[low:high]
        sorted_window2 = np.sort(window2)
        window_ranks = rank_order_function(window)
        itest_rank = window_ranks[iwin]
        itest_correct_result = sorted_window2[itest_rank]
        itest_result = result[itest]
        msg = "itest_result = {0}, correct result = {1}"
        assert itest_result == itest_correct_result, msg.format(
            itest_result, itest_correct_result)


def test3():
    nwin = 3
    nhalfwin = int(nwin/2)

    x = np.sort(np.random.rand(100))
    x2 = np.sort(np.random.rand(100))

    with NumpyRNGContext(fixed_seed):
        y = np.round(np.random.rand(len(x)), 2)
        y2 = np.round(np.random.rand(len(x2)), 2)

    i2_matched = np.searchsorted(x2, x)
    i2_matched = np.where(i2_matched >= len(y2), len(y2)-1, i2_matched)

    result = bin_free_conditional_abunmatch(x, y, x2, y2, nwin)

