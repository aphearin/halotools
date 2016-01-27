# -*- coding: utf-8 -*-

"""
functions to calculate empirical galaxy/halo abundnaces
"""

from __future__ import (division, print_function, absolute_import, unicode_literals)
import numpy as np
from ...utils.array_utils import custom_len, array_is_monotonic
from ...custom_exceptions import HalotoolsError

__all__ = ['empirical_cum_ndensity', 'empirical_diff_ndensity']


def empirical_cum_ndensity(x, volume, xbins = None, weights = None, 
                           nd_increases_wtih_x = False):
    """
    Caclulate cumulative number density of galaxies/halos given a property ``x``.
    
    Parameters
    ----------
    x : array_like
        array of galaxy or halo properties
    
    volume : float
        effective volume
    
    xbins : array_like, optional
        value of ``x`` for which to return the cumulative number densities.  If set to 
        None, return for every ``x``
    
    weights : array_like, optional
        weight to give every ``x``.  If set to None, equal weight of 1 given to each ``x``
    
    nd_increases_wtih_x : boolean, optional
        Boolean indicating that the number density increases with increasing ``x``.
        Default is False.
    
    Returns
    -------
    cumu_x : numpy.array
        cumulative number desntiy at values ``x_val``
    
    x_val : numpy.array
        values of ``x`` for which cumulative abundances are returned
    
    """
    
    x = np.array(x)
    Nx = custom_len(x)
    
    if weights is None:
        weights = np.ones(Nx)
    
    if not nd_increases_wtih_x:
        sorted_inds=np.argsort(x)[::-1]
        sorted_weights = weights[sorted_inds]
    else:
        sorted_inds=np.argsort(x)
        sorted_weights = weights[sorted_inds]
    
    sorted_x = x[sorted_inds]
    cumu_x = np.ones(Nx)*sorted_weights
    cumu_x = np.cumsum(cumu_x)/volume
    
    if xbins is not None:
        if not nd_increases_wtih_x:
            xbins = np.sort(xbins)[::-1]
            inds = np.searchsorted(sorted_x[::-1], xbins)*-1
        else:
            xbins = np.sort(xbins)
            inds = np.searchsorted(sorted_x, xbins)
        in_range = (np.abs(inds)!=Nx) & (inds!=0)
        cumu_x = cumu_x[inds[in_range]]
        x_centers = xbins[in_range]
    else:
        x_centers = sorted_x
    
    return cumu_x, x_centers


def empirical_diff_ndensity(x, volume, xbins, weights = None):
    """
    Caclulate differential number density of galaxies/halos given a property ``x``.
    
    Parameters
    ----------
    x : array_like
        array of galaxy or halo properties
    
    volume : float
        effective volume
    
    xbins : array_like
        bins of ``x`` for which to return the differential number densities.
    
    weights : array_like, optional
        weight to give every ``x``.  If set to None, equal weight of 1 given to each ``x``
    
    Returns
    -------
    diff_x : numpy.array
        differential number density at values ``x_centers``
    
    x_centers : numpy.array
        values of ``x`` for which differential abundances are returned
    """
    
    x = np.array(x)
    Nx = custom_len(x)
    
    x_centers = (xbins[:-1]+xbins[1:])/2.0
    
    if weights is None:
        weights = np.ones(Nx)
    
    effective_weights = (1.0/volume)*weights
    
    diff_x = np.histogram(x, bins=xbins, weights=effective_weights)[0]
    
    return diff_x, x_centers


