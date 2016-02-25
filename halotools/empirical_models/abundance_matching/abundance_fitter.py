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
from copy import deepcopy
from warnings import warn

__all__ = ['AbundanceFitter']

def chi2(measured_vals, updated_model, cov, x):
        """
        Chi^2 statistic for fitting an abundance function with covariance in the 
        measured abundances.
        
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
        """
        Parameters
        ----------
        optimizer : `~astropy.modeling.optimizers` object, optional
            optimizer to use when fitting model.
        
        """
        
        self.statistic = chi2
        super(AbundanceFitter, self).__init__(optimizer, statistic=self.statistic)
    
    def __call__(self, model, x, y, cov, estimate_param_cov=False, N_realizations=100,  **kwargs):
        """
        Fit model to this data.
        
        Parameters
        ----------
        model : `~astropy.modeling.core.ParametricModel`
            model to fit to ``x``, ``y``
        
        x : array
            input coordinates
        
        y : array
            input coordinates
        
        cov : numpy.matrix
            covaraince matrix of ``y``.  
            If a 1-D array, it is assumed to the diagonal, 
            and all other elemetns are set to 0.0
        
        estimate_param_cov : boolean, optional
            estimate the parameter covariance assuming multivariate normal errors 
            on the data given the input covariance matrix, ``cov``.  Default is False.
        
        N_realizations : int
            integer indicating the number of realizations to use for the parameter 
            covariance if ``estimate_param_cov`` is set to True.
        
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
        
        #estimate parameter covariance
        if estimate_param_cov:
            self.pcov = _estimate_parameter_covariance(self, model_copy, x, y,
                                                       cov, N=N_realizations, **kwargs)
        else:
            self.pcov = None
        
        return model_copy


def _estimate_parameter_covariance(fitter, model, x, y, cov, N=100, **kwargs):
    """
    Estimate the parameter covariance of a model by simulating the data and fitting it
    ``N`` times assuming multivariate normal errors on the data specified by ``cov``.
    
    Parameters
    ----------
    model : `~astropy.modeling.core.ParametricModel`
            model to fit to ``x``, ``y``
        
    x : array
        input coordinates
        
    y : array
        input coordinates
        
    cov : numpy.matrix
        covaraince matrix of ``y``.  
    
    N : int, optional
        integer indicating the number of realizations to use for the parameter 
        covariance if ``estimate_param_cov`` is set to True.
        
    kwargs : dict
        optional keyword arguments to be passed to the optimizer
    
    Returns
    -------
    pcov : np.ndarray
        convariance matrix of the parmeters
    """
    
    #dont use the fitted model, but instead use a copy
    model_copy = deepcopy(model)
    
    #get best fit parameters
    p0, _ = _model_to_fit_params(model_copy)
    
    #simulate data
    yy = np.random.multivariate_normal(y, cov, size=N)
    
    #initialize array to store new fit parameters
    new_fitparams = np.zeros((N,len(model.parameters)))
    fit_sucessful = np.array([True]*N)
    
    kwargs['disp'] = 0
    
    #refit the data using each set of simulated data
    for i in range(0, N):
        print(kwargs)
        farg = _convert_input(x, yy[i,:]) #choose one of the simulations
        farg = (model_copy, cov) + farg
        result_holder, fit_info = fitter._opt_method(
            fitter.objective_function, p0, farg, **kwargs)
        
        #check to see if fit was a success
        if fit_info['exit_mode']==0:
            new_fitparams[i,:] = result_holder
        else:
            fit_sucessful[i] = False
    
    #check to see if fits were a success!
    percent_sucessful = np.sum(fit_sucessful)/N
    if percent_sucessful!=1.0:
        msg = ("During parameter covariance estimation,\n"
               "only {0}% of fits were sucessful.\n"
               "Only sucssful fits are used for the \n"
               "estimate of the covariance matrix.".format(percent_sucessful*100))
        warn(msg)
    
    #remove unsuccessful fits
    new_fitparams = new_fitparams[fit_sucessful,:]
    
    #calculate the covariance in the parameters
    pcov = np.cov(new_fitparams,rowvar=0)
    
    return pcov
    
    
