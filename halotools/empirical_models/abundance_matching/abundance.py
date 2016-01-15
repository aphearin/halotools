# -*- coding: utf-8 -*-

"""
functions and classes used to create galaxy and halo abundance functions
"""

from __future__ import (division, print_function, absolute_import, unicode_literals)
import numpy as np
from ...utils.array_utils import custom_len, array_is_monotonic
from ...custom_exceptions import HalotoolsError
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import six 
from abc import ABCMeta, abstractmethod

__all__ = ['AbundanceFunction', 'AbundanceFunctionFromTabulated',
           'empirical_cum_ndensity','empirical_diff_ndensity']

@six.add_metaclass(ABCMeta)
class AbundanceFunction(object):
    """
    object that has callable galaxy/halo differential and cumulative abundance functions
    """
    
    @abstractmethod
    def n(self,x):
        """
        Return the cumulative number density of galaxies/halos
        
        Parameters 
        ----------
        x : numpy.array
            galaxy or halo properties, e.g. stellar mass
        
        Returns 
        -------
        n : numpy.array
            cumulative number desnity of galaixes/halos with property x
        """
        msg = "All subclasses of AbundanceFunction must include a `n` method."
        raise NotImplementedError(msg)
    
    @abstractmethod
    def dn(self,x):
        """
        Return the differential number density of galaxies/halos
        
        Parameters 
        ----------
        x : numpy.array
            galaxy or halo properties
        
        Returns 
        -------
        dn : numpy.array
            differential number desnity of galaixes/halos with property x
        """
        msg = "All subclasses of AbundanceFunction must include a `dn` method."
        raise NotImplementedError(msg)
    
    #check for required attributes
    def __init__(self):
        required_attrs = ["n_increases_with_x"]
        for attr in required_attrs:
            try:
                assert hasattr(self, attr)
            except AssertionError:
                raise HalotoolsError("All subclasses of Parent must have the following attributes: 'n_increases_with_x'")


class AbundanceFunctionFromTabulated(AbundanceFunction):
    """
    Create a galaxy/halo abundance function from tabulated data.
    """
    def __init__(self, **kwargs):
        """
        n : array_like
            tabulated number densities (cumulative or differential).
        
        x : array_like
            galaxy/halo property.
            *log10(n)* should roughly be linear in ``x``
        
        type : string
            'cumulative' or 'differential'
        
        n_increases_with_x : boolean, optional
            boolean indicating if abundance increases with increasing x.  
            The default is False.
        
        num_points_to_fit_high_abundance : int, optional
            The number of input tabulated points to use when fitting the high abundance 
            end of the abundnace function for extrapolation.  The deault is 4.
        
        num_points_to_fit_low_abundance : int, optional
            The number of input tabulated points to use when fitting the low abundance 
            end of the abundnace function for extrapolation.  The deault is 4.
        """
        
        #process input arguments
        if 'n_increases_with_x' not in kwargs.keys():
            self.n_increases_with_x = False
        else:
            if type(kwargs['n_increases_with_x']) is not bool:
                msg = "`n_increases_with_x` parameter must of type bool."
                raise ValueError(msg)
            self.n_increases_with_x = kwargs['n_increases_with_x']
        if 'num_points_to_fit_high_abundance' not in kwargs.keys():
            self._num_points_to_fit_high_abundance = 4
        else:
            self._num_points_to_fit_high_abundance = kwargs['num_points_to_fit_high_abundance']
        if 'num_points_to_fit_low_abundance' not in kwargs.keys():
            self._num_points_to_fit_low_abundance = 4
        else:
            self._num_points_to_fit_low_abundance = kwargs['num_points_to_fit_low_abundance']
        
        self._s_low = slice(-self._num_points_to_fit_low_abundance,None,None)
        self._s_high = slice(self._num_points_to_fit_low_abundance)
        
        self._n = kwargs['n']
        self._x = kwargs['x']
        if not array_is_monotonic(self._x):
            msg = "input `x` must be a monotonic array"
            raise ValueError(msg)
        if not array_is_monotonic(self._n):
            msg = "input `n` must be a monotonic array"
            raise ValueError(msg)
        if custom_len(self._x)!=custom_len(self._n):
            msg = "input `n` and `x` must be the same length"
            raise ValueError(msg)
        if (custom_len(self._x)<4) | (custom_len(self._n)<4):
            msg = "input `n` and `x` must be longer than 4 elements"
            raise ValueError(msg)
        
        if kwargs['type']=='cumulative':
            self._n = np.sort(self._n)[::-1]
            self._log_n = np.log10(self._n)
        elif kwargs['type']=='differential':
            self._dn = np.sort(self._n)[::-1]
            self._log_dn = np.log10(self._dn)
        
        if not self.n_increases_with_x is False:
            self._x = np.sort(self._x)[::-1]
        else:
            self._x = np.sort(self._x)
        self._min_x = np.min(self._x)
        self._max_x = np.max(self._x)
        
        #build tabulated cumulative or differential depending on what is missing.
        if kwargs['type']=='cumulative':
           self. _diff_cum_n()
        else:
            self._integrate_diff_n()
        
        #build callable abundance functions
        self._spline_dn()
        self._spline_n()
        self._extrapolate_dn()
        self._extrapolate_n()
        
        AbundanceFunction.__init__(self)
    
    def _spline_dn(self):
        """
        spline the tabulated differential abundance function
        """
        
        if self.n_increases_with_x:
            self._log_dn_func = interp1d(self._x[::-1], self._log_dn[::-1], kind='linear')
        else:
            self._log_dn_func = interp1d(self._x, self._log_dn, kind='linear')
    
    def _spline_n(self):
        """
        spline the tabulated cumulative abundance function
        """
        
        if self.n_increases_with_x:
            self._log_n_func = interp1d(self._x[::-1], self._log_n[::-1], kind='linear')
        else:
            self._log_n_func = interp1d(self._x, self._log_n, kind='linear')
            
    def _extrapolate_dn(self):
        """
        Fit the upper and lower bounds of the tabulated differential abundance 
        function to create extrapolation.  The high abndnace end is fit with a 
        linear function, the low-abundnace end is fit with a linear + exponetial 
        drop-off.
        """
        
        #fit low abundance end
        def func_l(x, a, b, c, d):
            return -1.0*np.exp(a*x+b) + c*x + d
            
        a0 = -1.0 if self.n_increases_with_x else 1.0
        popt_l = curve_fit(func_l, self._x[self._s_low],
            self._log_dn[self._s_low], [a0, 0, 0, 0], maxfev=100000)[0]
        self._ext_log_dn_func_l = lambda x: func_l(x, *popt_l)
        
        #fit high abundance end
        func_h = lambda x, a, b: a*x+b
        
        popt_h = curve_fit(func_h, self._x[self._s_high],
            self._log_dn[self._s_high], [0, 0], maxfev=100000)[0]
        self._ext_log_dn_func_h = lambda x: func_h(x, *popt_h)
    
    def _extrapolate_n(self):
        """
        Fit the upper and lower bounds of the tabulated cumulative abundance 
        function to create extrapolation.  The high abndnace end is fit with a 
        linear function, the low-abundnace end is fit with a linear + exponetial 
        drop-off.
        """
        
        #fit low abundance end
        def func_l(x, a, b, c, d):
            return -1.0*np.exp(a*x+b) + c*x + d
            
        a0 = 1.0 if self.n_increases_with_x else -1.0
        popt_l = curve_fit(func_l, self._x[:-1][self._s_low],
            self._log_n[:-1][self._s_low], [a0, 0, 0, 0], maxfev=1000000)[0]
        self._ext_log_n_func_l = lambda x: func_l(x, *popt_l)
        
        #fit high abundance end
        func_h = lambda x, a, b: a*x+b
        
        popt_h = curve_fit(func_h, self._x[self._s_high],
            self._log_n[self._s_high], [0, 0], maxfev=100000)[0]
        self._ext_log_n_func_h = lambda x: func_h(x, *popt_h)
    
    def _integrate_diff_n(self):
        """
        integrate a differential number density to get the cumulative number 
        density.
        """
        
        #set the initial value.  This is somewhat arbitrary.
        init_value = self._dn[-1]*np.fabs(self._x[-1]-self._x[-2])
        
        self._n = np.fabs(integrate.cumtrapz(self._dn[::-1], self._x[::-1],
            initial=init_value))
        self._n = self._n[::-1]
        self._log_n = np.log10(self._n)
    
    def _diff_cum_n(self):
        """
        differential the cumulative number density to get the differential number 
        density
        """
        self._dn = np.fabs(np.diff(self._n))
        self._log_dn = np.log10(self._dn)
    
    def dn(self, x):
        """
        return the differential abundance
        """
        
        #determine if the galaxies/halos are inside the tabulated range
        mask_high = (x>self._max_x)
        mask_low = (x<self._min_x)
        mask_in_range = (x>=self._min_x) & (x<=self._max_x)
        
        #initialize the result
        result = np.zeros(len(x))
        result[mask_in_range] = 10**self._log_dn_func(x[mask_in_range])
        
        #call the interpolation functions if necessary
        if self.n_increases_with_x:
            result[mask_high] = 10**self._ext_log_dn_func_h(x[mask_high])
            result[mask_low] = 10**self._ext_log_dn_func_l(x[mask_low])
        else:
            result[mask_high] = 10**self._ext_log_dn_func_l(x[mask_high])
            result[mask_low] = 10**self._ext_log_dn_func_h(x[mask_low])
        
        return result
    
    def n(self, x):
        """
        return the cumulative abundance
        """
        
        #determine if the galaxies/halos are inside the tabulated range
        mask_high = (x>self._max_x)
        mask_low = (x<self._min_x)
        mask_in_range = (x>=self._min_x) & (x<=self._max_x)
        
        #initialize the result
        result = np.zeros(len(x))
        result[mask_in_range] = 10**self._log_n_func(x[mask_in_range])
        
        #call the interpolation functions if necessary
        if self.n_increases_with_x:
            result[mask_high] = 10**self._ext_log_n_func_h(x[mask_high])
            result[mask_low] = 10**self._ext_log_n_func_l(x[mask_low])
        else:
            result[mask_high] = 10**self._ext_log_n_func_l(x[mask_high])
            result[mask_low] = 10**self._ext_log_n_func_h(x[mask_low])
        
        return result


def empirical_cum_ndensity(x, volume, xbins = None, weights = None, 
                           nd_increases_wtih_x = False):
    """
    Caclulate cumulative number density of galaxies/halos given a property ``x``.
    
    Parameters
    ----------
    x : array_like
        array of galaxy or halo properties
    
    volume : float
        effective volume
    
    xbins : array_like, optional
        value of ``x`` for which to return the cumulative number densities.  If set to 
        None, return for every ``x``
    
    weights : array_like, optional
        weight to give every ``x``.  If set to None, equal weight of 1 given to each ``x``
    
    nd_increases_wtih_x : boolean, optional
        Boolean indicating that the number density increases with increasing ``x``.
        Default is False.
    
    Returns
    -------
    cumu_x : numpy.array
        cumulative number desntiy at values ``x_val``
    
    x_val : numpy.array
        values of ``x`` for which cumulative abundances are returned
    
    """
    
    Nx = custom_len(x)
    
    if weights is None:
        weights = np.ones(Nx)
    
    sorted_inds=np.argsort(x)
    sorted_weights = weights[sorted_inds]
    
    if xbins is None:
        sorted_x = x[sorted_inds]
        cumu_x = np.ones(Nx)*sorted_weights
        cumu_x = np.cumsum(cumu_x)/volume
        x_centers = sorted_x
    else:
        Nx_weighed = np.sum(weights)
        inds = np.searchsorted(x, xbins, sorter=sorted_inds)
        cumu_x = np.zeros(len(xbins))
        #for i,ind in enumerate(inds):
        #    cumu_x[i] = Nx_weighed - np.sum(sorted_weights[:inds])
        x_centers = xbins
    
    return cumu_x, x_centers


def empirical_diff_ndensity(x, volume, xbins, weights = None):
    """
    Caclulate differential number density of galaxies/halos given a property ``x``.
    
    Parameters
    ----------
    x : array_like
        array of galaxy or halo properties
    
    volume : float
        effective volume
    
    xbins : array_like
        bins of ``x`` for which to return the differential number densities.
    
    weights : array_like, optional
        weight to give every ``x``.  If set to None, equal weight of 1 given to each ``x``
    
    Returns
    -------
    diff_x : numpy.array
        differential number density at values ``x_centers``
    
    x_centers : numpy.array
        values of ``x`` for which differential abundances are returned
    """
    
    Nx = custom_len(x)
    
    x_centers = (xbins[:-1]+xbins[1:])/2.0
    
    if weights is None:
        weights = np.ones(Nx)
    
    effective_weights = (1.0/volume)*weigths
    
    diff_x = np.histogram(x, bins=xbins, weights=effective_weights)
    
    return diff_x, x_centers
    
    