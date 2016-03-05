# -*- coding: utf-8 -*-

"""
funcitons to measure void statistics
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
####import modules########################################################################
import numpy as np
from .pair_counters.double_tree_per_object_pairs import *
from ..custom_exceptions import *
from warnings import warn

from ..utils import convert_to_ndarray
##########################################################################################


__all__=['void_prob_func', 'underdensity_prob_func']
__author__ = ['Duncan Campbell', 'Andrew Hearin']


np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero in e.g. DD/RR


def void_prob_func(sample1, rbins, n_ran, period=None, num_threads=1,
                   approx_cell1_size=None, approx_cellran_size=None):
    """
    Calculate the void probability function (VPF), :math:`P_0(r)`.
    
    :math:`P_0(r)` is defined as the probability that randomly placed sphere of size 
    :math:`r` contains zero points.
    
    See the :ref:`mock_obs_pos_formatting` documentation page for 
    instructions on how to transform your coordinate position arrays into the 
    format accepted by the ``sample1`` argument.   

    See also :ref:`galaxy_catalog_analysis_tutorial8`

    Parameters
    ----------
    sample1 : array_like
        Npts x 3 numpy array containing 3-D positions of points.
    
    rbins : float
        size of spheres to search for neighbors
    
    n_ran : int
        integer number of randoms to use to seeach for voids
    
    period : array_like, optional 
        length 3 array defining axis-aligned periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*3).
        If set to None, PBCs are set to infinity. Even in this case, it is still necessary 
        to drop down randomly placed spheres in order to compute the VPF. To do so, 
        the spheres will be dropped inside a cubical box whose sides are defined by  
        the smallest/largest coordinate distance of the input ``sample1``. 
    
    num_threads : int, optional
        number of 'threads' to use in the pair counting.  if set to 'max', use all 
        available cores.  num_threads=0 is the default.
    
    approx_cell1_size : array_like, optional 
        Length-3 array serving as a guess for the optimal manner by which 
        the `~halotools.mock_observables.pair_counters.FlatRectanguloidDoubleTree` 
        will apportion the ``sample1`` points into subvolumes of the simulation box. 
        The optimum choice unavoidably depends on the specs of your machine. 
        Default choice is to use *max(rbins)* in each dimension, 
        which will return reasonable result performance for most use-cases. 
        Performance can vary sensitively with this parameter, so it is highly 
        recommended that you experiment with this parameter when carrying out  
        performance-critical calculations. 
    
    approx_cellran_size : array_like, optional 
        Analogous to ``approx_cell1_size``, but for used for randoms.  See comments for 
        ``approx_cell1_size`` for details.
    
    Returns
    -------
    vpf : numpy.array
        *len(rbins)* length array containing the void probability function 
        :math:`P_0(r)` computed for each :math:`r` defined by input ``rbins``.
    
    Notes
    -----
    This function requires the calculation of the number of pairs per randomly placed 
    sphere, and thus storage of an array of shape(n_ran,len(rbins)).  This can be a 
    memory intensive process as this array becomes large.
    
    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a 
    periodic unit cube. 
    
    >>> Npts = 10000
    >>> Lbox = 1.0
    >>> period = np.array([Lbox,Lbox,Lbox])
    
    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)
    
    We transform our *x, y, z* points into the array shape used by the pair-counter by 
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation 
    is used throughout the `~halotools.mock_observables` sub-package:
    
    >>> coords = np.vstack((x,y,z)).T
    
    >>> rbins = np.logspace(-2,-1,20)
    >>> n_ran = 1000
    >>> vpf = void_prob_func(coords, rbins, n_ran, period)

    See also 
    ----------
    :ref:`galaxy_catalog_analysis_tutorial8`

    """
    if period is None:
        pass
    else:
        period = convert_to_ndarray(period)
        if len(period) == 1:
            period = np.array([period, period, period])
        elif len(period) == 3:
            pass
        else:
            msg = ("\nInput ``period`` must either be a float or length-3 sequence")
            raise HalotoolsError(msg)

    #process input
    if type(n_ran) is not int:
        msg = ("\n `n_ran` must be a positive integer.")
        raise HalotoolsError(msg)
    elif n_ran<1:
        raise HalotoolsError(msg)
    
    #create random sphere centers
    if period is None:
        pmin, pmax = np.min(sample1), np.max(sample1)
        randoms = np.random.uniform(pmin, pmax, (n_ran, 3))
    else:
        xran = np.random.uniform(0, period[0], n_ran)
        yran = np.random.uniform(0, period[1], n_ran)
        zran = np.random.uniform(0, period[2], n_ran)
        randoms = np.vstack([xran, yran, zran]).T
    
    result = per_object_npairs(randoms, sample1, rbins, period = period,\
                              num_threads = num_threads,\
                              approx_cell1_size = approx_cell1_size,\
                              approx_cell2_size = approx_cellran_size)
    
    mask = (result > 1)
    result = np.sum(mask, axis=0)
    
    return (n_ran - result)/n_ran


def underdensity_prob_func(sample1, rbins, n_ran, period=None, 
    sample_volume = None, u=0.2, num_threads=1,
    approx_cell1_size=None, approx_cellran_size=None):
    """
    Calculate the underdensity probability function (UPF), :math:`P_U(r)`.
    
    :math:`P_U(r)` is defined as the probability that a randomly placed sphere of size 
    :math:`r` encompases a volume with less than a specified density.

    See the :ref:`mock_obs_pos_formatting` documentation page for 
    instructions on how to transform your coordinate position arrays into the 
    format accepted by the ``sample1`` argument.   

    See also :ref:`galaxy_catalog_analysis_tutorial8`. 

    Parameters
    ----------
    sample1 : array_like
        Npts x 3 numpy array containing 3-D positions of points.
    
    rbins : float
        size of spheres to search for neighbors
    
    n_ran : int
        integer number of randoms to use to seeach for voids
    
    period : array_like, optional 
        length 3 array defining axis-aligned periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*3).
        If set to None, PBCs are set to infinity. Even in this case, it is still necessary 
        to drop down randomly placed spheres in order to compute the VPF. To do so, 
        the spheres will be dropped inside a cubical box whose sides are defined by  
        the smallest/largest coordinate distance of the input ``sample1``. 
    
    sample_volume : float, optional 
        If period is set to None, you must specify the effective volume of the sample. 

    u : float, optional
        density threshold in units of the mean object density
    
    num_threads : int, optional
        number of 'threads' to use in the pair counting.  if set to 'max', use all 
        available cores.  num_threads=0 is the default.
    
    approx_cell1_size : array_like, optional 
        Length-3 array serving as a guess for the optimal manner by which 
        the `~halotools.mock_observables.pair_counters.FlatRectanguloidDoubleTree` 
        will apportion the ``sample1`` points into subvolumes of the simulation box. 
        The optimum choice unavoidably depends on the specs of your machine. 
        Default choice is to use *max(rbins)* in each dimension, 
        which will return reasonable result performance for most use-cases. 
        Performance can vary sensitively with this parameter, so it is highly 
        recommended that you experiment with this parameter when carrying out  
        performance-critical calculations. 
    
    approx_cellran_size : array_like, optional 
        Analogous to ``approx_cell1_size``, but for used for randoms.  See comments for 
        ``approx_cell1_size`` for details.
    
    Returns
    -------
    upf : numpy.array
        *len(rbins)* length array containing the underdensity probability function 
        :math:`P_U(r)` computed for each :math:`r` defined by input ``rbins``.
    
    Notes
    -----
    This function requires the calculation of the number of pairs per randomly placed 
    sphere, and thus storage of an array of shape(n_ran,len(rbins)).  This can be a 
    memory intensive process as this array becomes large.
    
    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a 
    periodic unit cube. 
    
    >>> Npts = 10000
    >>> Lbox = 1.0
    >>> period = np.array([Lbox,Lbox,Lbox])
    
    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)
    
    We transform our *x, y, z* points into the array shape used by the pair-counter by 
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation 
    is used throughout the `~halotools.mock_observables` sub-package:
    
    >>> coords = np.vstack((x,y,z)).T
    
    >>> rbins = np.logspace(-2,-1,20)
    >>> n_ran = 1000
    >>> upf = underdensity_prob_func(coords, rbins, n_ran, period, u=0.2)

    See also 
    ----------
    :ref:`galaxy_catalog_analysis_tutorial8`
    """
    if period is None:
        if sample_volume is None:
            msg = ("If period is set to None, "
                "you must specify the effective sample_volume of the sample.")
            raise HalotoolsError(msg)
    else:
        period = convert_to_ndarray(period)
        if len(period) == 1:
            period = np.array([period, period, period])
        elif len(period) == 3:
            pass
        else:
            msg = ("\nInput ``period`` must either be a float or length-3 sequence")
            raise HalotoolsError(msg)
        sample_volume = period.prod()
    
    #process input
    if type(n_ran) is not int:
        msg = ("\n `n_ran` must be a positive integer.")
        raise HalotoolsError(msg)
    elif n_ran<1:
        raise HalotoolsError(msg)
    
    u = float(u)
    
    #create random sphere centers
    if period is None:
        pmin, pmax = np.min(sample1), np.max(sample1)
        randoms = np.random.uniform(pmin, pmax, (n_ran, 3))
    else:
        xran = np.random.uniform(0, period[0], n_ran)
        yran = np.random.uniform(0, period[1], n_ran)
        zran = np.random.uniform(0, period[2], n_ran)
        randoms = np.vstack([xran, yran, zran]).T
    
    result = per_object_npairs(randoms, sample1, rbins, period = period,\
                               num_threads = num_threads,\
                               approx_cell1_size = approx_cell1_size,\
                               approx_cell2_size = approx_cellran_size)
    
    # calculate the number of galaxies as a
    # function of r that corresponds to the
    # specified under-density
    mean_rho = len(sample1)/sample_volume
    vol = (4.0/3.0)* np.pi * rbins**3
    N_max = mean_rho*vol*u
    mask = (result > N_max)
    
    result = np.sum(mask, axis=0)
    
    return (n_ran - result)/n_ran



