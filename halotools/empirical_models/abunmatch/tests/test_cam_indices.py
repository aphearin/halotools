"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from ..cam import conditional_abunmatch
from ..cam_indices import conditional_abunmatch_indices


fixed_seed = 43


def test_cam_indices_equals_cam():
    """
    """
    nwin = 101
    npts = int(1e4)
    with NumpyRNGContext(fixed_seed):
        x = np.random.uniform(0, 4, npts)
        x2 = np.random.uniform(-1, 10, npts)
        y = np.random.normal(loc=0, scale=4, size=npts)
        y2 = np.random.normal(loc=-2, scale=0.4, size=npts)
    ynew1 = conditional_abunmatch(x, y, x2, y2, nwin, add_subgrid_noise=False)
    ynew2 = conditional_abunmatch_indices(x, y, x2, y2, nwin, add_subgrid_noise=False)

    assert np.allclose(ynew1, ynew2, rtol=0.05)
