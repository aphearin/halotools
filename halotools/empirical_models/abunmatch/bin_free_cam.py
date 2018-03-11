"""
"""
import numpy as np
from ...utils import unsorting_indices
from ...utils.conditional_percentile import _check_xyn_bounds, rank_order_function
from .engines import cython_bin_free_cam_kernel
from .tests.naive_python_cam import sample2_window_indices


def conditional_abunmatch_bin_free(x, y, x2, y2, nwin, add_subgrid_noise=True,
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
    >>> new_y = conditional_abunmatch_bin_free(x, y, x2, y2, nwin)
    """
    x, y, nwin = _check_xyn_bounds(x, y, nwin)
    x2, y2, nwin = _check_xyn_bounds(x2, y2, nwin)
    nhalfwin = int(nwin/2)
    npts1 = len(x)

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
        y_sorted, y2_sorted, i2_matched, nwin, int(add_subgrid_noise)))

    #  Finish the leftmost points in pure python
    iw = 0
    for ix1 in range(0, nhalfwin):
        iy2_low, iy2_high = sample2_window_indices(ix1, x_sorted, x2_sorted, nwin)
        leftmost_sorted_window_y2 = np.sort(y2_sorted[iy2_low:iy2_high])
        leftmost_window_ranks = rank_order_function(y_sorted[:nwin])
        result[ix1] = leftmost_sorted_window_y2[leftmost_window_ranks[iw]]
        iw += 1

    #  Finish the rightmost points in pure python
    iw = nhalfwin + 1
    for ix1 in range(npts1-nhalfwin, npts1):
        iy2_low, iy2_high = sample2_window_indices(ix1, x_sorted, x2_sorted, nwin)
        rightmost_sorted_window_y2 = np.sort(y2_sorted[iy2_low:iy2_high])
        rightmost_window_ranks = rank_order_function(y_sorted[-nwin:])
        result[ix1] = rightmost_sorted_window_y2[rightmost_window_ranks[iw]]
        iw += 1

    if assume_x_is_sorted:
        return result
    else:
        return result[unsorting_indices(idx_x_sorted)]
