"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from ..bin_free_cam import bin_free_conditional_abunmatch


fixed_seed = 43


def test1():
    nwin = 3

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    x2 = x

    y = np.arange(1, len(x)+1)
    y2 = y*10.

    i2_matched = np.searchsorted(x2, x)
    i2_matched = np.where(i2_matched >= len(y2), len(y2)-1, i2_matched)

    result = bin_free_conditional_abunmatch(x, y, x2, y2, nwin)

    print("i  = {0}\n".format(i2_matched.astype('i4')))

    print("y  = {0}".format(y))
    print("y2 = {0}\n".format(y2))
    print("ynew  = {0}".format(result.astype('i4')))

    assert 4 == 5
