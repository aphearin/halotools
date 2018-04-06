"""
"""
import numpy as np
from ...utils import unsorting_indices
from ...utils.conditional_percentile import _check_xyn_bounds, rank_order_function
from .engines import cython_bin_free_cam_kernel
from .tests.naive_python_cam import sample2_window_indices


__all__ = ('conditional_abunmatch_indices', )


def conditional_abunmatch_indices(x, y, x2, y2, nwin, add_subgrid_noise=True,
            assume_x_is_sorted=False, assume_x2_is_sorted=False):
    r"""
    Given a set of input points with primary property `x` and secondary property `y`,
    use conditional abundance matching to map new values `ynew` onto the input points
    such that :math:`P(<y_{\rm new} | x) = P(<y_2 | x)`, and also that
    `y` and `ynew` are in monotonic correspondence at fixed `x`.

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (n1, ) storing the primary property of the input points.

    y : ndarray
        Numpy array of shape (n1, ) storing the secondary property of the input points.

    x2 : ndarray
        Numpy array of shape (n2, ) storing the primary property of the desired distribution.

    y2 : ndarray
        Numpy array of shape (n2, ) storing the secondary property of the desired distribution.

    nwin : int
        Odd integer specifying the size of the window
        used to estimate :math:`P(<y | x)`. See Notes.

    add_subgrid_noise : bool, optional
        Flag determines whether random uniform noise will be added to fill in
        the gaps at the sub-grid level determined by `nwin`. This argument
        can be important for eliminating artificial discreteness effects.
        Default is True.

    assume_x_is_sorted : bool, optional
        Performance enhancement flag that can be used for cases where input `x` and `y`
        have been pre-sorted so that `x` is monotonically increasing. Default is False.

    assume_x2_is_sorted : bool, optional
        Performance enhancement flag that can be used for cases where input `x2` and `y2`
        have been pre-sorted so that `x2` is monotonically increasing. Default is False.

    Returns
    -------
    ynew : ndarray
        Numpy array of shape (n1, ) storing the new values of
        the secondary property for the input points.

    Examples
    --------
    >>> npts1, npts2 = 5000, 3000
    >>> x = np.linspace(0, 1, npts1)
    >>> y = np.random.uniform(-1, 1, npts1)
    >>> x2 = np.linspace(0.5, 0.6, npts2)
    >>> y2 = np.random.uniform(-5, 3, npts2)
    >>> nwin = 51
    >>> idx_new_y = conditional_abunmatch_indices(x, y, x2, y2, nwin)

    Notes
    -----
    The ``nwin`` argument controls the precision of the calculation,
    and also the performance. For estimations of Prob(< y | x) with sub-percent accuracy,
    values of ``window_length`` must exceed 100. Values more tha a few hundred are
    likely overkill when using the (recommended) sub-grid noise option.

    See :ref:`cam_tutorial` demonstrating how to use this
    function in galaxy-halo modeling with several worked examples.

    With the release of Halotools v0.7, this function replaced a previous function
    of the same name. The old function is now called
    `~halotools.empirical_models.conditional_abunmatch_bin_based`.

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
