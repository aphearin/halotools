"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from ..bin_free_cam import bin_free_conditional_abunmatch
from ....utils.conditional_percentile import cython_sliding_rank

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
    print("y2    = {0}\n".format(y2))

    assert np.all(result == y2)
