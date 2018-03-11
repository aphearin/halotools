"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from ..bin_free_cam import bin_free_conditional_abunmatch
from ....utils.conditional_percentile import cython_sliding_rank, rank_order_function
from .naive_python_cam import pure_python_rank_matching


fixed_seed = 5


def test_hard_coded_case5():
    nwin = 3

    x = np.array((1., 1., 1, 1, 1))
    x2 = np.array([0.1,  0.36, 0.5, 0.74, 0.83])

    y = np.array([0.12, 0.13, 0.24, 0.33, 0.61])
    y2 = np.array([0.3, 0.04, 0.6, 10., 5.])

    correct_result = [0.6, 5., 5., 5., 10.]
    result = bin_free_conditional_abunmatch(x, y, x2, y2, nwin)

    print("\n\ncorrect result = {0}".format(correct_result))
    print("cython result  = {0}\n".format(result))

    assert np.allclose(result, correct_result)
