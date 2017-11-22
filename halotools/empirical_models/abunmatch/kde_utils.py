"""
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
from scipy.stats import gaussian_kde


__all__ = ('kde_cdf', 'kde_conditional_percentile', 'percentile_curves')


def kde_cdf(data, y, ymin=-np.inf, ymax=np.inf):
    r""" Estimate the cumulative distribution function P(< y) defined by an input data sample.

    Parameters
    ----------
    data : ndarray
        Numpy array of shape (num_sample, )

    y : ndarray
        Numpy array of shape (num_pts, )

    ymin : float, optional
        Minimum value for the data. Default is -np.inf

    ymax : float, optional
        Maximum value for the data. Default is np.inf

    Returns
    -------
    cdf : ndarray
        Numpy array of shape (num_pts, )

    Eyamples
    --------
    >>> data = np.random.normal(loc=0, scale=1, size=int(1e4))
    >>> y = np.linspace(-4, 4, 100)
    >>> cdf = kde_cdf(data, y)
    """
    y = np.atleast_1d(y)

    kde = gaussian_kde(data)
    cdf = np.fromiter((kde.integrate_box_1d(ymin, high) for high in y), dtype='f4')

    #  Over-write CDF values below ymin
    lowmask = y < ymin
    highmask = y > ymax

    if ymin > -np.inf:
        cdf[lowmask] = 0.
        lowprob = kde.integrate_box_1d(-np.inf, ymin)
    else:
        lowprob = 0.0

    #  Over-write CDF values above ymax
    if ymax < np.inf:
        cdf[highmask] = 1.
        highprob = kde.integrate_box_1d(ymax, np.inf)
    else:
        highprob = 0.0

    #  Add back in the incorrectly missed CDF outside the boundaries
    if (ymax < np.inf) | (ymin > -np.inf):
        cdf[~lowmask & ~highmask] = cdf[~lowmask & ~highmask] + lowprob + highprob

    return cdf


def kde_cdf_interpol(data, y, npts_interpol=None, npts_sample=None,
            ymin=-np.inf, ymax=np.inf):
    r""" Estimate the cumulative distribution function P(< y) defined by an input data sample,
    optionally interpolating from a downsampling of the data
    to improve performance at the cost of precision.

    Parameters
    ----------
    data : ndarray
        Numpy array of shape (num_sample, )

    y : ndarray
        Numpy array of shape (num_pts, )

    npts_interpol : int, optional
        Number of points used to build a lookup table in lieu of evaluating the CDF
        at every point in y. Should be smaller than the number of points in y.
        Default behavior is to evaluate the CDF eyactly at each value of y.

    npts_sample : int, optional
        Size of downsampled data to use for construction of the KDE.
        Default behavior is not to downsample at all.
        Should be smaller than the number of points in the input data sample.

    Returns
    -------
    cdf : ndarray
        Numpy array of shape (num_pts, )

    Eyamples
    --------
    >>> data = np.random.normal(loc=0, scale=1, size=int(1e4))
    >>> y = np.linspace(-4, 4, 100)
    >>> cdf = kde_cdf_interpol(data, y)
    >>> cdf = kde_cdf_interpol(data, y, npts_interpol=500)
    >>> cdf = kde_cdf_interpol(data, y, npts_sample=1000)
    """
    if npts_sample is not None:
        msg = "npts_sample ={0} cannot exceed number of len(sample) = {1}"
        assert npts_sample <= len(data), msg.format(npts_sample, len(data))
        data = np.random.choice(data, npts_sample, replace=False)

    y = np.atleast_1d(y)
    if npts_interpol is not None:
        y_table = np.linspace(data.min(), data.max(), npts_interpol)
        cdf_table = kde_cdf(data, y_table, ymin=ymin, ymax=ymax)

        return np.interp(y, y_table, cdf_table)
    else:
        return kde_cdf(data, y, ymin=ymin, ymax=ymax)


def kde_conditional_percentile(y, x, x_bins, npts_interpol=2000, npts_sample=5000,
            ymin=-np.inf, ymax=np.inf):
    r""" Estimate P(< y | x) for each input point (x, y) by using Gaussian kernel density
    estimation within each bin defined by x_bins.

    Points outside the bounds of x_bins
    will be treated as if they are members of the outer bins.

    Parameters
    ----------
    y : ndarray
        Numpy array of shape (num_sample, )

    x : ndarray
        Numpy array of shape (num_sample, )

    x_bins : ndarray
        Numpy array of shape (num_bins, )

    Examples
    --------
    >>> npts = int(1e4)
    >>> x = np.linspace(0, 10, npts)
    >>> y = np.random.normal(loc=x)
    >>> nbins = 15
    >>> x_bins = np.linspace(0, 10, nbins)
    >>> conditional_percentile = kde_conditional_percentile(y, x, x_bins)
    """
    result = np.zeros_like(y) + np.nan

    #  Overwrite values of x that lie outside the outermost bins
    x = np.where(x < x_bins[0], x_bins[0], x)
    dx = (x_bins[-1] - x_bins[-2])/2.
    x = np.where(x >= x_bins[-1], x_bins[-1]-dx, x)

    for low, high in zip(x_bins[:-1], x_bins[1:]):
        mask = (x >= low) & (x < high)
        npts_bin = np.count_nonzero(mask)
        if npts_bin > 0:
            data = y[mask]
            result[mask] = kde_cdf_interpol(data, data,
                npts_interpol=min(npts_interpol, npts_bin),
                npts_sample=min(npts_sample, npts_bin),
                ymin=ymin, ymax=ymax)

    return result


def percentile_curves(p, x, y, x_bins, **kwargs):
    r""" Calculate the p-percentile curves of y as a function of x.

    A common use of this function is to compute, for example,
    the 95-percentile region enveloping :math:`\langle y \vert x \rangle`.

    Parameters
    ----------
    p : float or sequence
        Float or ndarray of shape (num_p, ) storing the value of the percentiles
        that define the percentile curves.

    x : ndarray
        Numpy array of shape (num_pts, ) storing the x-value used to bin the data

    y : ndarray
        Numpy array of shape (num_pts, ) storing the y-value used to compute P( > y | x)

    x_bins : ndarray
        Numpy array of shape (num_bins, ) storing the bin edges used to bin the data
        Care must be taken so that each bin contains a sufficient number of points
        to robustly calculate each percentile (e.g., at least 50 points if p=0.02 is desired)

    Returns
    -------
    x_mids : ndarray
        Numpy array shape of (num_bins-1, ) storing the midpoint of each bin

    pcurves : ndarray
        Numpy array of shape (num_p, num_bins-1) storing the percentile curves

    Examples
    --------
    Here we generate some fake data to show how to calculate
    the median value of y as a function of x,
    and also the 90-percentile region enveloping this median relation.

    >>> npts = int(1e4)
    >>> p = (0.05, 0.5, 0.95)
    >>> x = np.linspace(5, 100, npts)
    >>> y = np.random.rand(npts)
    >>> x_bins = np.linspace(5, 100, 10)
    >>> x_mids, pcurves = percentile_curves(p, x, y, x_bins)
    """
    p = np.atleast_1d(p)

    y_percentile = kde_conditional_percentile(y, x, x_bins, **kwargs)

    xmids = 0.5*(x_bins[:-1] + x_bins[1:])
    result = np.zeros((len(p), len(xmids)))

    for j, bounds in enumerate(zip(x_bins[:-1], x_bins[1:])):
        low, high = bounds
        mask = (x >= low) & (x < high)
        sample_percentile = y_percentile[mask]
        sample_y = y[mask]
        for i, pval in enumerate(p):
            result[i, j] = sample_y[np.argmin(np.abs(sample_percentile-pval))]

    return xmids, result
