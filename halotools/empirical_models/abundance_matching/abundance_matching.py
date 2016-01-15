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


__all__ = ['AbundanceMatching']

class AbundanceMatching(PrimGalpropModel):
    """ 
    """
    
    def __init__(self, abundance_function, prim_haloprop_key, scatter = 0.2):
    	"""

    	"""

    	pass


    def match(self):
    	pass

    def deconvolve_galaxy_halo_mapping(self):
    	pass























    
