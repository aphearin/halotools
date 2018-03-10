"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from ..bin_free_cam import bin_free_conditional_abunmatch
from ....utils.conditional_percentile import cython_sliding_rank, rank_order_function
from .test_pure_python import pure_python_rank_matching


fixed_seed = 5


def test5():
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


