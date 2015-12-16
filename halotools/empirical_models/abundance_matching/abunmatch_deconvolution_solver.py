# -*- coding: utf-8 -*-

"""
provides a class to caclulate the first moment of the galaxy-halo connection for 
generalized abundance matching.
"""

from __future__ import (division, print_function, absolute_import, unicode_literals)
import numpy as np
from .deconvolution import abunmatch_deconvolution
from ...custom_exceptions import *
from ...utils.abundance import *


__all__ = ['AbunmatchSolver']


class AbunMatchSolver():
    """
    A class that provides methods to solve for the first moment of SMHM relation.
    """
    
    def __init__(self, gal_abund_func, halo_abund_func, gal_prop_range, halo_prop_range):
        """
        Parameters
        ----------
        gal_abund_func : AbundanceFunction object
            galaxy abundance function
        
        gal_prop_range : 
        """
        
        self.gal_abund_func = gal_abund_func
        self.gal_prop_range = gal_prop_range
        self.halo_abund_func = halo_abund_func
        self.halo_prop_range = halo_prop_range
        
        #initialize dictionary to store deconvolution results, and remainders
        self._gal_prop_deconv = {}
    
    
    def solve(self, scatter=0, n_gal_prop_steps=100):
        """
        Solve for the first moment of the SMHM given fixed scatter.
        
        Parameters
        ----------
        scatter : float
            fixed log-normal scatter in galprop given haloprop
        
        Notes
        -----
        If the deconvolution has been solved for already for all the parameters, use the 
        previous result and set the first moment.
        
        If it has not been solved for with the parameters, call the deconvolution method, 
        and then set the first moment with the result.
        """
        
        #this should be a function that returns the galaxy property given the halo property.
        #self.gal_prop_first_moment = 
    
    
    def _deconvolute(self, scatter, repeat=10, sm_step=0.005, return_remainder=True):
        """
        Deconvolute the abundance function with a given scatter (assuming Gaussian)
        This function uses Peter Behroozi's 'fiducial_deconvolute' in c code.
        You must first compile fiducial_deconvolute to use this function.

        Parameters
        ----------
        scatter : float
            Standard deviation (sigma) of the Gaussian, in the unit of x.
        repeat : int, optional
            Number of times to repeat fiducial deconvolute process.
            This value can change the result significantly. 
            *Always* check a reasonable value is used.
        sm_step : float, optional
            Some parameter used in fiducial_deconvolute.
            Using 0.01 or 0.005 is fine.
        return_remainder : bool, optional
            If True, calculate the remainder of this deconvolution.
            *Always* check the reminder is reasonable before 
            doing abundance matching.

        Returns
        -------
        remainder : array_like
            Returned only if `return_remainder` is True.
        """
        if not _has_fiducial_deconvolute:
            raise NotImplementedError('Make sure you compliled fiducial_deconvolute.')

        af_key = np.empty(len(self._x), float)
        af_val = np.empty_like(af_key)
        af_key[::-1] = self._x
        if not self._x_flipped:
            af_key *= -1.0
        af_val[::-1] = self._phi_log
        af_val /= np.log(10.0)

        smm = np.empty_like(af_key)
        mf = np.empty_like(af_key)
        smm[::-1] = self._x
        mf[::-1] = np.gradient(np.exp(self._nd_log))
        if not self._x_flipped:
            smm *= -1.0
        smm = fiducial_deconvolute(af_key, af_val, smm, mf, scatter, repeat, sm_step)
        if not self._x_flipped:
            smm *= -1.0
        smm = smm[::-1]
        self._x_deconv[float(scatter)] = smm

        if return_remainder:
            nd = np.exp(np.interp(self._x, smm[self._s], self._nd_log[self._s]))
            dx = np.fabs((self._x[-1] - self._x[0])/float(len(self._x)-1))
            nd_conv = _convolve_gaussian(nd, float(scatter)/dx)
            return nd_conv - np.exp(self._nd_log)
        
        #stores result of deconvolution and remainder
        self._gal_prop_deconv[float(scatter)] = smm
    
    def _match(self, nd, scatter=0, do_add_scatter=True, do_rematch=True):
        """
        Abundance matching: match number density to x, i.e. return x(nd).
    
        Parameters
        ----------
        nd : array_like
            Number densities.
        scatter : float, optional
            If not zero, it uses an abundance function that has been 
            deconvoluted with this amount of scatter. 
            Must run `deconvolute` before calling this function.
        do_add_scatter : bool, optional
            Add scatter to the final catalog.
        do_rematch : bool, optional
            Rematch the final catalog to the abundance function.

        Returns
        -------
        catalog : array_like
            The abundance proxies (e.g. magnitude or log(stellar mass))
            at the given number densities.
        """
        scatter = float(scatter)
        if scatter > 0:
            try:
                xp = self._x_deconv[scatter]
            except (KeyError):
                raise ValueError('Please run deconvolute first!')
        else:
            xp = self._x
        x = np.interp(np.log(nd), self._nd_log, xp, np.nan, np.nan)

        if scatter > 0:
            if do_add_scatter:
                x = add_scatter(x, scatter, True)
                if do_rematch:
                    x2 = np.interp(np.log(nd), self._nd_log, self._x, np.nan, np.nan)
                    x = rematch(x, x2, self._x_flipped)
        return x
        
        
