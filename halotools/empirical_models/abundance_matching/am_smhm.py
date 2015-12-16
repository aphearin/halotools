# -*- coding: utf-8 -*-

"""
Module containing classes used to model the mapping between a galaxy property, e.g. 
stellar mass, and a halo property, e.g. virial mass using abundnace matching.
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
#from .abunmatch_deconvolution_solver import AbunMatchSolver

from ...utils.array_utils import custom_len
from .abundance import AbundanceFunction
from ...sim_manager import sim_defaults, HaloCatalog
from ...custom_exceptions import *


__all__ = ['AbunMatchSmHm']

class AbunMatchSmHm(PrimGalpropModel):
    """ 
    This class provides a general framework for abudance matching to solve for 
    the galaxy-halo connection.
    
    SMHM relation based on Behroozi et al. (2010), arXiv:1205.5807.  While this is based
    on modeling the stellar mass-halo mass relaton, the galaxy property and halo property
    could be other properties than those.
    """
    
    def __init__(self, gal_abund_func, halo_abund_func, galprop_name,
                 prim_haloprop_key, halo_prop_range, **kwargs):
        """
        Parameters 
        ----------
        gal_abund_func : AbundanceFunction object
            galaxy abundance function
        
        halo_abund_func : AbundanceFunction object
            halo abundance function
        
        galprop_name : string  
            String giving the name of the primary galaxy property the model determines 
        
        prim_haloprop_key : string  
            String giving the column name of the primary halo property governing
            ``galprop_name``
        
        halo_prop_range : tuple
            a tuple containing the minimum and maximum halo property for which the 
            galaxy-halo connection is to be defined.
        
        scatter_level : float, optional
            The amount of scatter in dex. Default behavior will result in constant scatter 
            at a level set in the `~halotools.empirical_models.model_defaults` module.
        """
        
        
        setattr(self, 'mean_'+galprop_name, self._galprop_first_moment)
        
        super(AbunMatchSmHm, self).__init__(prim_haloprop_key = prim_haloprop_key, 
            galprop_name = galprop_name, scatter_model = ConstantLogNormalScatter,
            **kwargs)
        
        self.gal_abund_func = gal_abund_func
        self.halo_abund_func = halo_abund_func
        
        self.scatter_level = super(AbunMatchSmHm, self).param_dict['scatter_model_param1']
        
        print(self.scatter_level)
    
    