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

from .abundance import AbundanceFunctionFromTabulated, empirical_cum_ndensity

from ...utils.array_utils import custom_len
from .abundance import AbundanceFunction
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

        """
        self.prim_haloprop_key = prim_haloprop_key
        self.galaxy_abundance_function = galaxy_abundance_function
        self.galprop_sample_values = galprop_sample_values
        self.haloprop_sample_values = haloprop_sample_values

        try:
            halo_abundance_function = kwargs['halo_abundance_function']
        except KeyError:
            try:
                complete_halo_catalog = kwargs['complete_subhalo_catalog']
                Lbox = kwargs['Lbox']
                sim_volume = Lbox**3.
            except KeyError:
                msg = ("\nIf you do not pass in a ``halo_abundance_function`` keyword argument \n"
                    "you must pass in ``complete_subhalo_catalog`` and ``Lbox`` keyword arguments.\n")
                raise HalotoolsError(msg)
            else:
                try:
                    prim_haloprop_array = complete_halo_catalog[prim_haloprop_key]
                except KeyError:
                    msg = ("\nYou passed in a ``complete_halo_catalog`` argument.\n"
                        "This catalog does not have a column corresponding to the input \n"
                        "``prim_haloprop_key`` = " + prim_haloprop_key)
                    raise HalotoolsError(msg)
                else:
                    halo_abundance_array, sorted_prim_haloprop = empirical_cum_ndensity(prim_haloprop_array, sim_volume)
                    print(np.shape(halo_abundance_array), np.shape(sorted_prim_haloprop))
                    halo_abundance_function = (
                        AbundanceFunctionFromTabulated(
                            n = halo_abundance_array, 
                            x = sorted_prim_haloprop, 
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
        
        #calculate abundances for x1
        n = n1.n(x1)
        
        #invert the secondary abundance function at each x2
        sort_inds = np.argsort(n2.n(x2))
        inverted_n2 = interp1d(n2.n(x2)[sort_inds],x2[sort_inds])
        
        #calculate the value of x2 at the abundances of x1
        x2n = inverted_n2(n)
        
        #get x1 as a function of x2
        x1x2 = interp1d(x1,x2n)
        
        #extrapolate beyond x1 range using a linear function
        def fitting_func(x, a, b):
            return a*x+b
        
        #use the first 3 and last three tabulated points to fit extrapolation
        r_slice = slice(-3,None,None)
        l_slice = slice(3,None,None)
        
        #guess initial right and left extrapolation parameters
        init_r_params = [0,0]
        init_l_params = [0,0]
        
        #fit the left and right sides
        right_ext = curve_fit(fitting_func,x1[r_slice],x1x2[r_slice], p0=init_r_params)
        left_ext = curve_fit(fitting_func,x1[l_slice],x1x2[l_slice], p0=init_l_params)
        
        def x1x2_func(x):
            """
            given a value of x1, return matched x2
            
            use the interpolated x1x2 func if x is in the range of tabulated x1,
            otherwise, use the extrapolations
            """
            mask_high = (x>x1)
            mask_low = (x<x1)
            mask_in_range = (x>=x1) & (x<=x1)
            
            #initialize the result
            result = np.zeros(len(x))
            result[mask_in_range] = x1x2(x)
            
            result[mask_high] = right_ext(x)
            result[mask_low] = left_ext(x)
            
            return result
        
        return x1x2_func
    
    def deconvolve_galaxy_halo_mapping(self):
        pass

























    
