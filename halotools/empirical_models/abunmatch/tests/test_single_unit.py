"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from ..bin_free_cam import bin_free_conditional_abunmatch


fixed_seed = 5


def test_subgrid_noise1():
    n1, n2 = int(5e4), int(5e3)

    with NumpyRNGContext(fixed_seed):
        x = np.sort(np.random.uniform(0, 10, n1))
        y = np.random.uniform(0, 1, n1)

    with NumpyRNGContext(fixed_seed):
        x2 = np.sort(np.random.uniform(0, 10, n2))
        y2 = np.random.uniform(-4, -3, n2)

    nwin1 = 201
    result = bin_free_conditional_abunmatch(x, y, x2, y2, nwin1)
    result2 = bin_free_conditional_abunmatch(x, y, x2, y2, nwin1, add_subgrid_noise=True)
    assert np.allclose(result, result2, atol=0.1)
    assert not np.allclose(result, result2, atol=0.02)

    nwin2 = 1001
    result = bin_free_conditional_abunmatch(x, y, x2, y2, nwin2)
    result2 = bin_free_conditional_abunmatch(x, y, x2, y2, nwin2, add_subgrid_noise=True)
    assert np.allclose(result, result2, atol=0.02)
