# -*- coding: utf-8 -*-

"""
Module for fitting parameterized models to galaxy/halo abundances
"""

from __future__ import (division, print_function, absolute_import, unicode_literals)

import numpy as np
from astropy.modeling.fitting import (_validate_model,
                                      _fitter_to_model_params,
                                      _model_to_fit_params, Fitter,
                                      _convert_input)
from astropy.modeling.optimizers import Simplex

__all__ = ['AbundanceFitter']

def chi2(measured_vals, updated_model, cov, x):
        """
        Chi^2 statistic for fitting an abundance function with covariance in the 
        measured abundances
        
        Parameters
        ----------
        measured_vals : array_like
            array of measured abundances
        
        updated_model : `~astropy.modeling.ParametricModel`
            model with parameters set by the current iteration of the optimizer
        
        cov : numpy.matrix
            covaraince matrix of measured_values
        
        x : array_like
            input coordinates
        
        """
        model_vals = updated_model(x)
        
        Y = np.matrix(model_vals - measured_vals)
        
        inv_cov = cov.I
        
        X2 = Y*inv_cov*Y.T
        
        return float(X2)

class AbundanceFitter(Fitter):
    """
    Fit a paramterized abundance function to differential abundances
    
    Parameters
    ----------
    optimizer : class or callable
        one of the classes in optimizers.py (default: Simplex)
    """
    
    def __init__(self, optimizer=Simplex):
        self.statistic = chi2
        super(AbundanceFitter, self).__init__(optimizer, statistic=self.statistic)
    
    def __call__(self, model, x, y, cov, **kwargs):
        """
        Fit model to this data.
        
        Parameters
        ----------
        model : `~astropy.modeling.core.ParametricModel`
            model to fit to x, y
        
        x : array
            input coordinates
        
        y : array
            input coordinates
        
        cov : numpy.matrix
            covaraince matrix of y.  
            If a 1-D array, it is assumed to the diagonal, 
            and all other elemetns are set to 0.0
        
        kwargs : dict
            optional keyword arguments to be passed to the optimizer
        
        Returns
        ------
        model_copy : `~astropy.modeling.core.ParametricModel`
            a copy of the input model with parameters set by the fitter
        
        """
        model_copy = _validate_model(model, self._opt_method.supported_constraints)
        
        N = len(x)
        
        #process cov parameter.
        if np.shape(cov) == (N,):
            cov = np.matrix(np.diag(cov))
        elif np.shape(cov) == (N,N):
            cov = np.matrix(cov)
        else:
            msg = ("``cov`` parameter must be a square matrix \n"
                   "of shape len(``y``) x len(``y``), or a len(``y``) vector.")
            raise ValueError(msg)
        
        farg = _convert_input(x, y)
        farg = (model_copy, cov) + farg
        p0, _ = _model_to_fit_params(model_copy)
        
        fitparams, self.fit_info = self._opt_method(
            self.objective_function, p0, farg, **kwargs)
        _fitter_to_model_params(model_copy, fitparams)
        
        #degrees of freedom
        self.dof = N - len(model.parameters)-1.0
        
        #final chi^2 value
        self.chi2 = self.fit_info['final_func_val']
        
        #reduced chi^2 value
        self.red_chi2 = self.chi2/self.dof
        
        return model_copy
