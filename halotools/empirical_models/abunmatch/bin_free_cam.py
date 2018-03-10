"""
"""
import numpy as np
from ...utils import unsorting_indices
from ...utils.conditional_percentile import _check_xyn_bounds, rank_order_function
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
    """
    x = np.atleast_1d(x).astype('f8')
    y = np.atleast_1d(y).astype('f8')
    x2 = np.atleast_1d(x2).astype('f8')
    y2 = np.atleast_1d(y2).astype('f8')
    nwin = int(nwin)
    nhalfwin = int(nwin/2)

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
    print("initial i2_matched = {0}".format(i2_matched))

    result = np.array(cython_bin_free_cam_kernel(
        y_sorted, y2_sorted, i2_matched, nwin))

    leftmost_window_x = x_sorted[:nwin]
    leftmost_window_x2 = x2_sorted[:nwin]
    leftmost_window_i2 = np.searchsorted(leftmost_window_x2, leftmost_window_x).astype('i4')
    leftmost_window_i2 = np.where(leftmost_window_i2 >= nwin, nwin-1, leftmost_window_i2)
    leftmost_sorted_window_y2 = np.sort(y2_sorted[:nwin])

    leftmost_window_ranks = rank_order_function(y_sorted[:nwin])
    leftmost_window_y = leftmost_sorted_window_y2[leftmost_window_ranks[leftmost_window_i2]]
    result[:nhalfwin] = leftmost_window_y[:nhalfwin]

    rightmost_window_x = x_sorted[-nwin:]
    rightmost_window_x2 = x2_sorted[-nwin:]
    rightmost_window_i2 = np.searchsorted(rightmost_window_x2, rightmost_window_x).astype('i4')
    rightmost_window_i2 = np.where(rightmost_window_i2 >= nwin, nwin-1, rightmost_window_i2)
    rightmost_sorted_window_y2 = y2_sorted[-nwin:]
    rightmost_window_ranks = rank_order_function(y_sorted[-nwin:])
    rightmost_window_y = rightmost_sorted_window_y2[rightmost_window_ranks[rightmost_window_i2]]
    result[-nhalfwin:] = rightmost_window_y[-nhalfwin:]

    if assume_x_is_sorted:
        return result
    else:
        return result[unsorting_indices(idx_x_sorted)]
