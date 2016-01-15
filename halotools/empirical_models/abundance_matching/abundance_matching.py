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

from ...utils.array_utils import custom_len
from .abundance import AbundanceFunction
from ...sim_manager import sim_defaults, CachedHaloCatalog
from ...custom_exceptions import *


__all__ = ('AbundanceMatching', )

class AbundanceMatching(PrimGalpropModel):
    """ 
    """
    
    def __init__(self, galprop_name, prim_haloprop_key, galaxy_abundance_function, 
        scatter = 0.2, **kwargs):
        """

        """

        new_method_name = 'mean_' + galprop_name
        new_method_behavior = self._galprop_from_haloprop
        setattr(self, new_method_name, new_method_behavior)

        PrimGalpropModel.__init__(self, galprop_name, prim_haloprop_key=prim_haloprop_key, 
            scatter_model = ConstantLogNormalScatter, **kwargs)

        self.galaxy_abundance_function = galaxy_abundance_function


        self.publications = ['arXiv:1001.0015']


    def _galprop_from_haloprop(self, haloprop):
        return galprop

    def match(self):
        pass

    def deconvolve_galaxy_halo_mapping(self):
        pass

























    
