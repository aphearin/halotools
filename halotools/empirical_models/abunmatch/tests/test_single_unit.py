"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from ..bin_free_cam import bin_free_conditional_abunmatch
from ....utils.conditional_percentile import cython_sliding_rank, rank_order_function
from .test_pure_python import pure_python_rank_matching


fixed_seed = 43


def test4():

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

    print("x = {0}".format(x))
    print("x2 = {0}\n".format(x2))
    print("y = {0}".format(y))
    print("y2 = {0}".format(y2))

    print("result = {0}".format(result))

    assert np.allclose(result, pure_python_result)
