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

from ..smhm_models import PrimGalpropModel, ConstantLogNormalScatterModel
from ..model_defaults import *
from ..model_helpers import *
from .abunmatch_deconvolution_solver import AbunMatchSolver

from ...utils.array_utils import custom_len
from ...utils.abundance import AbundanceFunction
from ...sim_manager import sim_defaults, HaloCatalog
from ...custom_exceptions import *


__all__ = ['AbunMatchSmHm']

class AbunMatchSmHm(PrimGalpropModel):
    """ 
    SMHM relation based on Behroozi et al. (2010), arXiv:1205.5807.  While this is based
    on modeling the stellar mass-halo mass relaton, the galaxy property and halo property
    could be other properties than those.  This class provides a general framework for
    abudance matching to solve for the galaxy-halo connection.
    """
    
    def __init__(self, gal_abund_func, halo_abund_func, galprop_name,
                 prim_haloprop_key, **kwargs):
        """
        Parameters 
        ----------
        gal_abund_func : AbundanceFunction object
            galaxy abundance function
        
        gal_prop_range : tuple
            a tuple containing the minimum and maximum galaxy property for which the 
            abundance funciton is valid.
        
        halo_abund_func : AbundanceFunction object
            halo abundance function
        
        halo_prop_range : tuple
            a tuple containing the minimum and maximum halo property for which the 
            galaxy-halo is to be defined.
        
        galprop_name : string  
            String giving the name of the primary galaxy property the model determines 
        
        prim_haloprop_key : string  
            String giving the column name of the primary halo property governing
            ``galprop_name``
        
        scatter_level : float, optional  
            The amount of scatter in dex. Default behavior will result in constant scatter 
            at a level set in the `~halotools.empirical_models.model_defaults` module.
        
        redshift : float, optional 
            Redshift of the snapshot of the simulation you will be using. 
        
        new_haloprop_func_dict : function object, optional  
            Dictionary of function objects used to create additional halo properties 
            that may be needed by the model component. 
            Used strictly by the `MockFactory` during call to the `process_halo_catalog` method. 
            Each dict key of ``new_haloprop_func_dict`` will 
            be the name of a new column of the halo catalog; each dict value is a function 
            object that returns a length-N numpy array when passed a length-N Astropy table 
            via the ``halo_table`` keyword argument. 
            The input ``model`` model object has its own new_haloprop_func_dict; 
            if the keyword argument ``new_haloprop_func_dict`` passed to `MockFactory` 
            contains a key that already appears in the ``new_haloprop_func_dict`` bound to 
            ``model``, and exception will be raised. 
        
        """
        
        setattr(self, 'mean_'+galprop_name, self._galprop_first_moment)
        
        super(AbunMatchSmHm, self).__init__(prim_haloprop_key = prim_haloprop_key, 
            galprop_name = galprop_name, scatter_model = ConstantLogNormalScatterModel,
            **kwargs)
        
        self.gal_abund_func = gal_abund_func
        self.halo_abund_func = halo_abund_func
        self.scatter_level = scatter_level
        
        #intialize abundance matching solver
        self._abunmatch_solver = 
            AbunMatchSolver(self.gal_abund_func, self.halo_abund_func,
            self.halo_prop_range)
        
        #solve for SMHM
        self._gal_prop_first_moment = self._abunmatch_solver.solve(
                                          self.scatter_level, xprop_step=0.05)
        
        
    def _galprop_first_moment(self, **kwargs):
        """
        Return the mean galaxy property as a function of the input table.
        
        Parameters 
        ----------
        prim_haloprop : array, optional 
            Array of mass-like variable upon which occupation statistics are based. 
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed. 
        
        table : object, optional 
            Data table storing halo catalog. 
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed. 
        
        Returns 
        -------
        mean_gal_props : array_like 
            Array containing the mean value of the galaxy property occupying haloes with
            in ``table``.
        """
        redshift = safely_retrieve_redshift(self, 'mean_stellar_mass', **kwargs)

        # Retrieve the array storing the mass-like variable
        if 'table' in kwargs.keys():
            halo_props = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in kwargs.keys():
            halo_props= kwargs['prim_haloprop']
        else:
            raise KeyError("Must pass one of the following keyword arguments to \n"
                           "mean_"+galprop_name ":``table`` or ``prim_haloprop``")
        
        mean_galaxy_props = self._gal_prop_first_moment(halo_props)
        
        return mean_galaxy_props
    
    
        