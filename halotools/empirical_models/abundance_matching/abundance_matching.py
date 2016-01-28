# -*- coding: utf-8 -*-

"""
Module containing classes used to model the mapping between a galaxy property, e.g. 
stellar mass, and a halo property, e.g. virial mass. 
"""

from __future__ import (division, print_function, absolute_import, unicode_literals)

import numpy as np
from scipy.interpolate import UnivariateSpline
from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty
from astropy import cosmology
from warnings import warn
from functools import partial

from ..smhm_models import PrimGalpropModel, ConstantLogNormalScatter
from ..model_defaults import *
from ..model_helpers import *
from .abunmatch_deconvolution_solver import AbunMatchSolver

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
                complete_halo_catalog = kwargs['complete_subhalo_catalog']
                Lbox = kwargs['Lbox']
                sim_volume = Lbox**3.
            except KeyError:
                msg = ("\n If you do not pass in a ``halo_abundance_function`` \n"
                        "keyword argument, you must pass in ``complete_subhalo_catalog``\n"
                        "and ``Lbox`` keyword arguments.\n")
                raise HalotoolsError(msg)
            else:
                try:
                    prim_haloprop_array = complete_halo_catalog[prim_haloprop_key]
                except KeyError:
                    msg = ("\n You passed in a ``complete_halo_catalog`` argument.\n"
                           "This catalog does not have a column corresponding to the \n"
                           "input ``prim_haloprop_key`` = " + prim_haloprop_key)
                    raise HalotoolsError(msg)
                else: #build halo abundance function
                    halo_abundance_array, sorted_prim_haloprop = \
                        empirical_cum_ndensity(prim_haloprop_array, sim_volume)
                    halo_abundance_function = (
                        AbundanceFunctionFromTabulated(
                            n = halo_abundance_array, 
                            x = sorted_prim_haloprop,
                            use_log = True, 
                            type = 'cumulative', 
                            n_increases_with_x = False)
                        )
        self.halo_abundance_function = halo_abundance_function
        
        new_method_name = 'mean_' + galprop_name
        new_method_behavior = self._galprop_from_haloprop
        setattr(self, new_method_name, new_method_behavior)
        
        PrimGalpropModel.__init__(self, galprop_name, prim_haloprop_key=prim_haloprop_key, 
            scatter_model = ConstantLogNormalScatter, scatter_level = scatter_level, **kwargs)
        
        self.publications = ['arXiv:1001.0015']
    
    
    def _galprop_from_haloprop(self, **kwargs):
        """
        
        """
        # Retrieve the array storing the mass-like variable
        if 'table' in kwargs.keys():
            halo_mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in kwargs.keys():
            halo_mass = kwargs['prim_haloprop']
        else:
            raise KeyError("Must pass one of the following keyword arguments to mean_occupation:\n"
                "``table`` or ``prim_haloprop``")
        
        # downsample the halo_mass array for the spline table
        num_sample_pts = min(1000, len(halo_mass))
        
        func = self.match(
            self.galaxy_abundance_function, self.halo_abundance_function, 
            self.galprop_sample_values, self.haloprop_sample_values)
        
        return func(halo_mass)
    
    def match(self, n1, n2, x1, x2):
        """
        Given two cumulative abundnace functions, n1 and n2, return x1(x2).
        e.g. return the stellar mass halo mass relation given the stellar 
        mass and halo mass abundnace functions.
        
        Parameters
        ----------
        n1 : function
            primary cumulative abundance function, e.g. stellar mass function
        
        n2: function
            secondary cumulative abundance function, e.g. halo mass function
        
        x1 : array_like
            galaxy/halo property to sample result, e.g. stellar mass
        
        x2 : array_like
            galaxy/halo property thats samples relavent abundance range, e.g. halo mass
        
        Returns
        -------
        x1(x2) : function
            callable
        """
        
        x1 = np.sort(x1)
        x2 = np.sort(x2)
        
        #get range of x2 abundances
        ns = n2.n(x2)
        n_low = np.min(ns)
        n_high = np.max(ns)
        
        #sample the abundance function of x1
        n = n1.n(x1)
        
        #what is the range of x1 for x1 whose abundances are within that for x2
        in_range = (n <= n_high) & (n >= n_low)
        if np.sum(in_range)<2:
            raise ValueError("x1 under-sampled--could not invert the x1 abundance function")
        x1s = x1[in_range]
        
        #re-sample the x1 range more finely in the appropriate range
        N_x1_samples = 1000
        if n1._use_log_x:
            x1 = np.logspace(np.log10(np.min(x1s)), np.log10(np.max(x1s)), N_x1_samples)
        else: 
            x1 = np.linspace(np.min(x1s), np.max(x1s), N_x1_samples)
        
        #get the abundances associate with the sampled x1 range
        n = n1.n(x1)
        
        #invert the secondary abundance function at each x2
        sort_inds = np.argsort(n2.n(x2)) #needs to be monotonically increasing for the interpolation function
        if n2._use_log_x:
            log_inverted_n2 = interp1d(np.log10(n2.n(x2)[sort_inds]),np.log10(x2[sort_inds]))
            inverted_n2 = lambda x: 10**log_inverted_n2(x) 
        else:
            inverted_n2 = interp1d(np.log10(n2.n(x2)[sort_inds]),x2[sort_inds])
        
        #calculate the value of x2 at the abundances of the sampled x1
        x2n = inverted_n2(np.log10(n))
        
        #get x1 as a function of x2
        if n1._use_log_x: x1 = np.log10(x1)
        if n2._use_log_x: x2n = np.log10(x2n)
        x1x2 = interp1d(x2n,x1)
        
        #extrapolate beyond the interpolation range using a linear fit
        def fitting_func(x, a, b):
            return a*x+b
        
        #use the first three and last three tabulated points to fit extrapolation
        r_slice = slice(-3,None,None)
        l_slice = slice(3,None,None)
        
        #guess initial right and left extrapolation parameters
        init_r_params = [0,0]
        init_l_params = [0,0]
        
        #fit the left and right sides
        right_ext_p = curve_fit(fitting_func,x2n[r_slice],x1x2(x2n[r_slice]), p0=init_r_params)[0]
        right_ext = lambda x: fitting_func(x,*right_ext_p)
        
        left_ext_p = curve_fit(fitting_func,x2n[l_slice],x1x2(x2n[l_slice]), p0=init_l_params)[0]
        left_ext = lambda x: fitting_func(x,*left_ext_p)
        
        def x1x2_func(x):
            """
            given a value of x2, return matched x1
            
            use the interpolated x1x2 func if x is in the range of tabulated x1,
            otherwise, use the extrapolations
            """
            
            if n2._use_log_x: x = np.log10(x)
            
            mask_high = (x>np.max(x2n))
            mask_low = (x<np.min(x2n))
            mask_in_range = (x>=np.min(x2n)) & (x<=np.max(x2n))
            
            #initialize the result
            result = np.zeros(len(x))
            result[mask_in_range] = x1x2(x[mask_in_range])
            
            result[mask_high] = right_ext(x[mask_high])
            result[mask_low] = left_ext(x[mask_low])
            
            if n1._use_log_x: result=10**result
            return result
        
        return x1x2_func
    
    def deconvolve_galaxy_halo_mapping(self):
        pass

























    
