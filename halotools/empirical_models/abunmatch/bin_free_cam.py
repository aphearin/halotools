"""
"""
import numpy as np
from ...utils import unsorting_indices
from .engines import cython_bin_free_cam_kernel


def bin_free_conditional_abunmatch(x, y, x2, y2, nwin,
            assume_x_is_sorted=False, assume_x2_is_sorted=False):
    """
    Examples
    --------
    >>> npts1, npts2 = 5000, 3000
    >>> x = np.linspace(0, 1, npts1)
    >>> y = np.random.uniform(-1, 1, npts1)
    >>> x2 = np.linspace(0.5, 0.6, npts2)
    >>> y2 = np.random.uniform(-5, 3, npts2)
    >>> nwin = 51
    >>> result = bin_free_conditional_abunmatch(x, y, x2, y2, nwin)
    """
    if assume_x_is_sorted:
        x_sorted = x
        y_sorted = y
    else:
        idx_x_sorted = np.argsort(x)
        x_sorted = x[idx_x_sorted]
        y_sorted = y[idx_x_sorted]

    if assume_x2_is_sorted:
        x2_sorted = x2
        y2_sorted = y2
    else:
        idx_x2_sorted = np.argsort(x2)
        x2_sorted = x2[idx_x2_sorted]
        y2_sorted = y2[idx_x2_sorted]

    i2_matched = np.searchsorted(x2_sorted, x_sorted).astype('i4')

    result = np.array(cython_bin_free_cam_kernel(
        y_sorted, i2_matched, x2_sorted, y2_sorted, nwin))

    if assume_x_is_sorted:
        return result
    else:
        return result[unsorting_indices(idx_x_sorted)]
