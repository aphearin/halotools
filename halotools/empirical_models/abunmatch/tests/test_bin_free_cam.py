"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from ..bin_free_cam import bin_free_conditional_abunmatch


fixed_seed = 43


def test1():
    nwin = 3

    x = [1, 2, 3, 4, 5, 6, 7]
    x2 = [2.5, 3.5, 4.5, 5.5, 6.5]

    y = np.arange(0, len(x))
    y2 = np.arange(5, len(x2)+5)

    i2_matched = np.searchsorted(x2, x)

    result = bin_free_conditional_abunmatch(x, y, x2, y2, nwin)

    print("i  = {0}\n".format(i2_matched.astype('f4')))

    print("y  = {0}".format(y))
    print("y2 = {0}\n".format(y2))
    print("r  = {0}".format(result))

    assert 4 == 5
