# -*- coding: utf-8 -*-

"""
Module containing classes used to model the mapping between a galaxy property, e.g. 
stellar mass, and a halo property, e.g. virial mass. 
"""

from __future__ import (division, print_function, absolute_import, unicode_literals)

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import optimize
from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty
from astropy import cosmology
from warnings import warn
from functools import partial

from ..smhm_models import PrimGalpropModel, ConstantLogNormalScatter
from ..model_defaults import *
from ..model_helpers import *

from .deconvolution import abunmatch_deconvolution

from .abundance_function import *

from ...utils.array_utils import custom_len
from ...sim_manager import sim_defaults, CachedHaloCatalog
from ...custom_exceptions import HalotoolsError

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


__all__ = ('AbundanceMatching', )

class AbundanceMatching(PrimGalpropModel):
    """ 
    """
    
    def __init__(self, galprop_name, prim_haloprop_key, galaxy_abundance_function, 
        galprop_sample_values, haloprop_sample_values, 
        scatter_level = 0., **kwargs):
        """
        Parameters
        ----------
        galprop_name : string
        
        prim_haloprop_key : string
        
        galaxy_abundance_function : AbundanceFunction object
        
        halo_abundance_function : AbundanceFunction object
        
        galprop_sample_values : array_like
        
        haloprop_sample_values : array_like
        
        scatter_level : float, optional
        """
        self.prim_haloprop_key = prim_haloprop_key
        self.galaxy_abundance_function = galaxy_abundance_function
        self.galprop_sample_values = galprop_sample_values
        self.haloprop_sample_values = haloprop_sample_values
        
        #a halo abundance function is needed--process input to build one.
        try:
            halo_abundance_function = kwargs['halo_abundance_function']
        except KeyError:
            try:
                complete_halo_catalog10 = kwargs['complete_subhalo_catalog10']
                Lbox = kwargs['Lbox']
                sim_volume = Lbox**3.
            except KeyError:
                msg = ("\n If you do not pass in a ``halo_abundance_function`` \n"
                        "keyword argument, you must pass in ``complete_subhalo_catalog10``\n"
                        "and ``Lbox`` keyword arguments.\n")
                raise HalotoolsError(msg)
            else:
                try:
                    prim_haloprop_array = complete_halo_catalog10[prim_haloprop_key]
                except KeyError:
                    msg = ("\n You passed in a ``complete_halo_catalog10`` argument.\n"
                           "This catalog10 does not have a column corresponding to the \n"
                           "input ``prim_haloprop_key`` = " + prim_haloprop_key)
                    raise HalotoolsError(msg)
                else: #build halo abundance function
                    halo_abundance_array, sorted_prim_haloprop = \
                        empirical_cum_ndensity(prim_haloprop_array, sim_volume)
                    halo_abundance_function = (
                        AbundanceFunctionFromTabulated(
                            n = halo_abundance_array, 
                            x = sorted_prim_haloprop,
                            use_log10 = True, 
                            type = 'cumulative', 
                            n_increases_with_x = False)
                        )
        self.halo_abundance_function = halo_abundance_function
        
        new_method_name = 'mean_' + galprop_name
        new_method_behavior = self._galprop_from_haloprop
        setattr(self, new_method_name, new_method_behavior)
        
        PrimGalpropModel.__init__(self, galprop_name, prim_haloprop_key=prim_haloprop_key, 
            scatter_model = ConstantLogNormalScatter, scatter_level = scatter_level, **kwargs)
        
        self._match_params = None
        
        self.publications = ['arXiv:1001.0015']
    
    
    def _galprop_from_haloprop(self, **kwargs):
        """
        Return the mean ``gal_prop`` as a function of ``prim_halo_prop``.
        
        Parameters
        ----------
        table : astropy.table
        
        prim_haloprop : arry_like
        
        Returns
        -------
        gal_prop : numpy.array
        
        """
        # Retrieve the array storing the mass-like variable
        if 'table' in kwargs.keys():
            halo_mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in kwargs.keys():
            halo_mass = kwargs['prim_haloprop']
        else:
            msg = ("Must pass one of the following keyword arguments to mean_occupation:\n"
                   "``table`` or ``prim_haloprop``.")
            raise KeyError(msg)
        
        params = {'n1':self.galaxy_abundance_function,
                  'n2':self.halo_abundance_function,
                  'x1':self.galprop_sample_values,
                  'x2':self.haloprop_sample_values,
                  'N_sample': 1000
                  }
        
        #check to see if it is necessary to call the match routine
        if params==self._match_params:
            pass
        else:
            self._match_func = self.match(**params)
        
        return self._match_func(halo_mass)
    
    def match(self, n1, n2, x1, x2, N_sample=1000):
        """
        Given abundance functions, n1 and n2, return x1(x2), such that the cumulative
        abundances are equal, e.g. return the first momemnt of the stellar mass halo mass 
        relation given the stellar mass and halo mass abundance functions.
        
        Parameters
        ----------
        n1 : `~halotools.empirical_models.abundance_matching.AbundanceFunction` object
            primary abundance function, e.g. stellar mass function
        
        n2: `~halotools.empirical_models.abundance_matching.AbundanceFunction` object
            secondary abundance function, e.g. halo mass function
        
        x1 : array_like
            abscissa for ``n1``, e.g. stellar mass. The cumulative abundance of 
            ``x1`` must span the range of ``x2``.
        
        x2 : array_like
            abscissa for ``n2``, e.g. halo mass. This defines the range over which 
            abundances are matched.
        
        N_sample : int
            integer number indicating the number of samples to take of ``n1`` during 
            inversion.
        
        Returns
        -------
        x1(x2) : function
            callable
        """
        
        self._match_params = {'n1':n1,
                              'n2':n2,
                              'x1':x1,
                              'x2':x2,
                              'N_sample':N_sample
                              }
        
        #sort input halo and galaxy abscissa
        x1 = np.sort(x1)
        x2 = np.sort(x2)
        if n1.n_increases_with_x:
            x1 = x1[::-1]
        if n2.n_increases_with_x:
            x2 = x2[::-1]
        
        #get the range of x2 cumulative abundances
        ns = n2.n(x2)
        n_low = np.min(ns) #minimum abundance
        n_high = np.max(ns)#maximum abundance
        
        #sample the n1 cumulative abundance function using the x1 abscissa
        n = n1.n(x1)
        
        #check to see that at least two values fall within the n2.n(x2) range
        N_in_range = np.sum((n>n_low) & (n<n_high))
        if N_in_range<2:
            msg = ("``n1`` is under-sampled, and could not be inverted. \n"
                   "Increase the sampeling using ``x1`` abscissa parameter.")
            raise HalotoolsError(msg)
        
        #check to see that the values span the full range of n2.n(x2)
        N_high = np.sum((n>n_high))
        N_low = np.sum((n>n_high))
        if (N_high<1) | (N_low<1):
            msg = ("``x1`` must sample a cumulative abundance range \n"
                   "which spans that of secondary abundance function, ``n2``, \n"
                   "sampled at abscissa ``x2``.")
            raise HalotoolsError(msg)
        
        #find the values of x1 that straddle the extremes of n2.n(x2)
        inds = np.abs(np.searchsorted(ns[::-1],n[::-1])[::-1]-len(ns)) #bam!
        #find the values of x1 that straddle the low abundance end
        l1 = np.sum(inds<1)-1 #index of last 0
        x1_l1 = x1[l1]
        r1 = l1+1
        x1_r1 = x1[r1]
        #find the values of x1 that straddle the high abundance end
        r2 = np.sum(inds<len(ns)) #index of first len(ns)-1
        x1_r2 = x1[r2]
        l2 = r2-1
        x1_l2 = x1[l2]
        #solve for values on the boundaries (root finding)
        f = lambda xx: n1.n(xx)-n_high
        min_x1 = optimize.brentq(f, x1_l1, x1_r1, maxiter=100)
        f = lambda xx: n1.n(xx)-n_low
        max_x1 = optimize.brentq(f, x1_l2, x1_r2, maxiter=100)
        
        #get new x1 abscissa that span the abundance range of n2.n(x2)
        if n1._use_log10_x:
            x1 = np.logspace(np.log10(min_x1), np.log10(max_x1), N_sample)
        else: 
            x1 = np.linspace(min_x1, max_x1, N_sample)
        
        #get the cumulative abundances associate with the new x1 abscissa
        n = n1.n(x1)
        
        #invert the secondary abundance function at each x2
        sort_inds = np.argsort(n2.n(x2)) #must be monotonically increasing
        if n2._use_log10_x:
            log10_inverted_n2 = InterpolatedUnivariateSpline(np.log10(n2.n(x2)[sort_inds]),np.log10(x2[sort_inds]), k=1)
            inverted_n2 = lambda x: 10**log10_inverted_n2(x) 
        else:
            inverted_n2 = InterpolatedUnivariateSpline(np.log10(n2.n(x2)[sort_inds]),x2[sort_inds],k=1)
        
        #calculate the value of x2 at the cumulative abundances of the sampled x1
        x2n = inverted_n2(np.log10(n))
        
        #get x1 as a function of x2
        if n1._use_log10_x: x1 = np.log10(x1)
        if n2._use_log10_x: x2n = np.log10(x2n)
        x1x2 = InterpolatedUnivariateSpline(x2n,x1,k=1)
        
        #define a function and return
        def x1x2_func(x):
            """
            Return the x1 which has equal abundance to n2(x).
            """
            if n2._use_log10_x: x = np.log10(x)
            
            #throw a warning if it is beyond the interpolated range
            if np.any(x>x2):
                msg = ("``x`` is beyond the range x1x2() hase been solved for.")
                warn(msg)
            elif np.any(x<x2):
                msg = ("``x`` is beyond the range x1x2() hase been solved for.")
                warn(msg)
            
            result = x1x2(x)
            
            if n1._use_log10_x: result=10**result
            return result
        
        return x1x2_func
    
    def compute_deconvolved_galaxy_abundance_function(self, 
        galaxy_abundance_function, scatter, **kwargs):
        """ Should call abunmatch_deconvolution_wrapper.pyx 
        """
        return galaxy_abundance_function.compute_deconvolved_galaxy_abundance_function(scatter, **kwargs)

