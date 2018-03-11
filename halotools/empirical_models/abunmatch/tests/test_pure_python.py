"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from ....utils.conditional_percentile import cython_sliding_rank


fixed_seed = 43


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


def test_pure_python1():
    """
    """
    n1, n2, nwin = 5001, 1001, 11
    nhalfwin = nwin/2
    x_sample1 = np.linspace(0, 1, n1)
    with NumpyRNGContext(fixed_seed):
        y_sample1 = np.random.uniform(0, 1, n1)
    ranks_sample1 = cython_sliding_rank(x_sample1, y_sample1, nwin)

    x_sample2 = np.linspace(0, 1, n2)
    with NumpyRNGContext(fixed_seed):
        y_sample2 = np.random.uniform(-4, -3, n2)
    ranks_sample2 = cython_sliding_rank(x_sample2, y_sample2, nwin)

    result = pure_python_rank_matching(x_sample1, ranks_sample1,
            x_sample2, ranks_sample2, y_sample2, nwin)

    for ix1 in range(2*nwin, n1-2*nwin):

        x1 = x_sample1[ix1]
        rank1 = ranks_sample1[ix1]
        low, high = sample2_window_indices(x1, x_sample2, nwin)

        sorted_window2 = np.sort(y_sample2[low:high])
        assert len(sorted_window2) == nwin

        correct_result_ix1 = sorted_window2[rank1]

        assert correct_result_ix1 == result[ix1]


def test_hard_coded_case1():
    nwin = 3

    x = np.array([0.1,  0.36, 0.5, 0.74, 0.83])
    x2 = np.copy(x)

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.03, 0.54, 0.67, 0.73, 0.86])

    ranks_sample1 = cython_sliding_rank(x, y, nwin)
    ranks_sample2 = cython_sliding_rank(x2, y2, nwin)

    pure_python_result = pure_python_rank_matching(x, ranks_sample1,
            x2, ranks_sample2, y2, nwin)

    correct_result = [0.03, 0.54, 0.67, 0.73, 0.86]

    assert np.allclose(pure_python_result, correct_result)


def test_hard_coded_case2():
    nwin = 3

    x = np.array([0.1,  0.36, 0.36, 0.74, 0.83])
    x2 = np.array([0.54, 0.54, 0.55, 0.56, 0.57])

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.03, 0.54, 0.67, 0.73, 0.86])

    ranks_sample1 = cython_sliding_rank(x, y, nwin)
    ranks_sample2 = cython_sliding_rank(x2, y2, nwin)

    pure_python_result = pure_python_rank_matching(x, ranks_sample1,
            x2, ranks_sample2, y2, nwin)

    correct_result = [0.03, 0.54, 0.54, 0.73, 0.86]

    assert np.allclose(pure_python_result, correct_result)


def test_hard_coded_case3():
    nwin = 3

    x = np.array([0.1,  0.36, 0.5, 0.74, 0.83])
    x2 = np.copy(x)

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.3, 0.04, 0.6, 10., 5.])

    ranks_sample1 = cython_sliding_rank(x, y, nwin)
    ranks_sample2 = cython_sliding_rank(x2, y2, nwin)

    pure_python_result = pure_python_rank_matching(x, ranks_sample1,
            x2, ranks_sample2, y2, nwin)

    correct_result = [0.04, 0.3, 0.6, 5., 10.]

    assert np.allclose(pure_python_result, correct_result)


def test_hard_coded_case4():
    nwin = 3

    x = np.array((0., 0., 0., 0., 0.))
    x2 = np.array([0.1,  0.36, 0.5, 0.74, 0.83])

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.3, 0.04, 0.6, 10., 5.])

    ranks_sample1 = cython_sliding_rank(x, y, nwin)
    ranks_sample2 = cython_sliding_rank(x2, y2, nwin)

    pure_python_result = pure_python_rank_matching(x, ranks_sample1,
            x2, ranks_sample2, y2, nwin)

    correct_result = [0.04, 0.3, 0.3, 0.3, 0.6]

    assert np.allclose(pure_python_result, correct_result)


def test_hard_coded_case5():
    nwin = 3

    x = np.array((1., 1., 1, 1, 1))
    x2 = np.array([0.1,  0.36, 0.5, 0.74, 0.83])

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.3, 0.04, 0.6, 10., 5.])

    ranks_sample1 = cython_sliding_rank(x, y, nwin)
    ranks_sample2 = cython_sliding_rank(x2, y2, nwin)

    pure_python_result = pure_python_rank_matching(x, ranks_sample1,
            x2, ranks_sample2, y2, nwin)

    correct_result = [0.04, 5, 5, 5, 10]

    assert np.allclose(pure_python_result, correct_result)
