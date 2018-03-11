"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from ..bin_free_cam import bin_free_conditional_abunmatch
from ....utils.conditional_percentile import cython_sliding_rank, rank_order_function
from .test_pure_python import pure_python_rank_matching


fixed_seed = 43


def test1():
    """ Test case where x and x2 are sorted, y and y2 are sorted,
    and the nearest x2 value is lined up with x
    """
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
    """ Test case where x and x2 are sorted, y and y2 are not sorted,
    and the nearest x2 value is lined up with x
    """
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


def test4():
    """ Regression test for buggy treatment of rightmost endpoint behavior
    """

    n1, n2, nwin = 8, 8, 3
    x = np.round(np.linspace(0.15, 1.3, n1), 2)
    with NumpyRNGContext(fixed_seed):
        y = np.round(np.random.uniform(0, 1, n1), 2)
    ranks_sample1 = cython_sliding_rank(x, y, nwin)

    x2 = np.round(np.linspace(0.15, 1.3, n2), 2)
    with NumpyRNGContext(fixed_seed):
        y2 = np.round(np.random.uniform(-4, -3, n2), 2)
    ranks_sample2 = cython_sliding_rank(x2, y2, nwin)

    pure_python_result = pure_python_rank_matching(x, ranks_sample1,
            x2, ranks_sample2, y2, nwin)

    result = bin_free_conditional_abunmatch(x, y, x2, y2, nwin)

    assert np.allclose(result, pure_python_result)


def test_brute_force_interior_points():
    """
    """

    n1, n2, nwin = 101, 31, 11
    nhalfwin = nwin/2
    x = np.linspace(0, 1, n1)
    with NumpyRNGContext(fixed_seed):
        y = np.random.uniform(0, 1, n1)
    ranks_sample1 = cython_sliding_rank(x, y, nwin)

    x2 = np.linspace(0, 1, n2)
    with NumpyRNGContext(fixed_seed):
        y2 = np.random.uniform(-4, -3, n2)
    ranks_sample2 = cython_sliding_rank(x2, y2, nwin)

    pure_python_result = pure_python_rank_matching(x, ranks_sample1,
            x2, ranks_sample2, y2, nwin)

    cython_result = bin_free_conditional_abunmatch(x, y, x2, y2, nwin)

    assert np.allclose(pure_python_result[nwin:-nwin], cython_result[nwin:-nwin])


def test_brute_force_endpoints():
    """
    """

    n1, n2, nwin = 101, 31, 11
    nhalfwin = nwin/2
    x = np.linspace(0, 1, n1)
    with NumpyRNGContext(fixed_seed):
        y = np.random.uniform(0, 1, n1)
    ranks_sample1 = cython_sliding_rank(x, y, nwin)

    x2 = np.linspace(0, 1, n2)
    with NumpyRNGContext(fixed_seed):
        y2 = np.random.uniform(-4, -3, n2)
    ranks_sample2 = cython_sliding_rank(x2, y2, nwin)

    pure_python_result = pure_python_rank_matching(x, ranks_sample1,
            x2, ranks_sample2, y2, nwin)

    cython_result = bin_free_conditional_abunmatch(x, y, x2, y2, nwin)

    #  Test left edge
    assert np.allclose(pure_python_result[:nhalfwin], cython_result[:nhalfwin])

    #  Test right edge
    assert np.allclose(pure_python_result[-nhalfwin:], cython_result[-nhalfwin:])


def test_hard_coded_case1():
    nwin = 3

    x = np.array([0.1,  0.36, 0.5, 0.74, 0.83])
    x2 = np.copy(x)

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.03, 0.54, 0.67, 0.73, 0.86])

    correct_result = [0.03, 0.54, 0.67, 0.73, 0.86]
    result = bin_free_conditional_abunmatch(x, y, x2, y2, nwin)

    assert np.allclose(result, correct_result)

def test_hard_coded_case2():
    nwin = 3

    x = np.array([0.1,  0.36, 0.36, 0.74, 0.83])
    x2 = np.array([0.54, 0.54, 0.55, 0.56, 0.57])

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.03, 0.54, 0.67, 0.73, 0.86])

    correct_result = [0.03, 0.54, 0.54, 0.73, 0.86]
    result = bin_free_conditional_abunmatch(x, y, x2, y2, nwin)

    assert np.allclose(result, correct_result)


def test_hard_coded_case3():
    nwin = 3

    x = np.array([0.1,  0.36, 0.5, 0.74, 0.83])
    x2 = np.copy(x)

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.3, 0.04, 0.6, 10., 5.])

    correct_result = [0.04, 0.3, 0.6, 5., 10.]
    result = bin_free_conditional_abunmatch(x, y, x2, y2, nwin)

    assert np.allclose(result, correct_result)


def test_hard_coded_case4():
    nwin = 3

    x = np.array((0., 0., 0., 0., 0.))
    x2 = np.array([0.1,  0.36, 0.5, 0.74, 0.83])

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.3, 0.04, 0.6, 10., 5.])

    correct_result = [0.04, 0.3, 0.3, 0.3, 0.6]
    result = bin_free_conditional_abunmatch(x, y, x2, y2, nwin)

    assert np.allclose(result, correct_result)


def test_hard_coded_case6():
    """
    """

    x = [0.15, 0.31, 0.48, 0.64, 0.81, 0.97, 1.14, 1.3]
    x2 = [0.15, 0.38, 0.61, 0.84, 1.07, 1.3]

    y = [0.22, 0.87, 0.21, 0.92, 0.49, 0.61, 0.77, 0.52]
    y2 = [-3.78, -3.13, -3.79, -3.08, -3.51, -3.39]

    nwin = 5

    ranks_sample1 = cython_sliding_rank(x, y, nwin)
    ranks_sample2 = cython_sliding_rank(x2, y2, nwin)

    pure_python_result = pure_python_rank_matching(x, ranks_sample1,
            x2, ranks_sample2, y2, nwin)

    result = bin_free_conditional_abunmatch(x, y, x2, y2, nwin)

    assert np.allclose(result, pure_python_result)
