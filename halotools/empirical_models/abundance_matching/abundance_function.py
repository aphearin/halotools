# -*- coding: utf-8 -*-

"""
classes used to create galaxy and halo abundance functions used in the 
abundance matching module.
"""

from __future__ import (division, print_function, absolute_import, unicode_literals)
import numpy as np
from ...utils.array_utils import custom_len, array_is_monotonic
from ...custom_exceptions import HalotoolsError
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline
import six 
from abc import ABCMeta, abstractmethod
from warnings import warn

__all__ = ['AbundanceFunction',
           'AbundanceFunctionFromTabulated',
           'AbundanceFunctionFromCallable']
__author__=['Duncan Campbell', 'Andrew Hearin']

@six.add_metaclass(ABCMeta)
class AbundanceFunction(object):
    """
    abstract class that has callable galaxy/halo differential and cumulative abundance functions.
    """
    
    @abstractmethod
    def n(self,x):
        """
        Return the cumulative number density of galaxies/halos, n(<x).
        
        Parameters 
        ----------
        x : numpy.array
            galaxy or halo properties, e.g. stellar mass or halo mass
        
        Returns 
        -------
        n : numpy.array
            cumulative number density of galaxies/halos with property x
        """
        msg = "All subclasses of AbundanceFunction must include a `n` method."
        raise NotImplementedError(msg)
    
    @abstractmethod
    def dn(self,x):
        """
        Return the differential number density of galaxies/halos, dn(x)/dx.
        
        Parameters 
        ----------
        x : numpy.array
            galaxy or halo properties, e.g. stellar mass or halo mass
        
        Returns 
        -------
        dn : numpy.array
            differential number density of galaixes/halos with property x
        """
        msg = "All subclasses of AbundanceFunction must include a `dn` method."
        raise NotImplementedError(msg)
    
    #check for required attributes
    def __init__(self):
        required_attrs = ["n_increases_with_x","x_abscissa","_use_log_x"]
        for attr in required_attrs:
            try:
                assert hasattr(self, attr)
            except AssertionError:
                msg = ("All subclasses of Parent must have the following \n"
                       "attributes: 'n_increases_with_x', 'x_abscissa','_use_log_x'")
                raise HalotoolsError(msg)


class AbundanceFunctionFromTabulated(AbundanceFunction):
    """
    Galaxy/halo abundance function object from tabulated data.
    """
    def __init__(self, **kwargs):
        """
        n : array_like
            tabulated number densities (cumulative or differential).
        
        x : array_like
            tabulated galaxy/halo property.
        
        use_log : boolean
           bool indicating whether to use the log10(``x``).  Note that *log10(n)* should 
           roughly be linear in log10(``x``). The default is to True.
        
        type : string
            'cumulative' or 'differential'
        
        n_increases_with_x : boolean, optional
            boolean indicating if abundance increases with increasing x.  
            The default is False.
        
        num_points_to_fit_high_abundance : int, optional
            The number of input tabulated points to use when fitting the high abundance 
            end of the abundnace function for extrapolation.  The default is 4.
        
        num_points_to_fit_low_abundance : int, optional
            The number of input tabulated points to use when fitting the low abundance 
            end of the abundnace function for extrapolation.  The default is 4.
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
            if self._num_points_to_fit_high_abundance<4:
                msg = ("''num_points_to_fit_high_abundance'' must be >= 4")
                raise ValueError(msg)
        
        if 'num_points_to_fit_low_abundance' not in kwargs.keys():
            self._num_points_to_fit_low_abundance = 4
        else:
            self._num_points_to_fit_low_abundance = kwargs['num_points_to_fit_low_abundance']
            if self._num_points_to_fit_low_abundance<4:
                msg = ("''num_points_to_fit_low_abundance'' must be >= 4")
                raise ValueError(msg)
        
        #last three elements
        self._s_low = slice(-self._num_points_to_fit_low_abundance,None,None)
        #forst three three elements
        self._s_high = slice(self._num_points_to_fit_high_abundance)
        
        if not array_is_monotonic(kwargs['x']):
            msg = "input `x` must be a monotonic array"
            raise ValueError(msg)
        if not array_is_monotonic(kwargs['n']):
            msg = "input `n` must be a monotonic array"
            raise ValueError(msg)
        if custom_len(kwargs['x'])!=custom_len(kwargs['n']):
            msg = "input `n` and `x` must be the same length"
            raise ValueError(msg)
        if (custom_len(kwargs['x'])<4) | (custom_len(kwargs['n'])<4):
            msg = "input `n` and `x` must be longer than 4 elements"
            raise ValueError(msg)
        
        #Note that we store the tabulated abundance functions from high to low abundance,
        #as an astronomer would plot them!
        n = np.array(kwargs['n'])
        sort_inds = np.argsort(n)[::-1]
        x = kwargs['x']
        n = n[sort_inds]
        x = x[sort_inds]
        
        self.x_abscissa = x
        
        #use self._x to store x's associated with cumulative
        #use self._xx to store x's associated with differential
        
        if kwargs['type']=='cumulative':
            self._n = n
            self._log_n = np.log10(self._n)
            self._x = x
            #remove duplicate x's
            self._x, uniq_inds = np.unique(self._x, return_index=True)
            self._n = np.copy(self._n[uniq_inds])
            self._log_n = np.copy(self._log_n[uniq_inds])
            self._xx = np.copy(self._x)
        elif kwargs['type']=='differential':
            self._dn = n
            self._log_dn = np.log10(self._dn)
            self._xx = x
            #remove duplicate x's
            self._xx, uniq_inds = np.unique(self._xx, return_index=True)
            self._dn = np.copy(self._dn[uniq_inds])
            self._log_dn = np.copy(self._log_dn[uniq_inds])
            self._x = np.copy(self._xx)
       
        #set whether the log10 of x should be used
        if 'use_log' not in kwargs.keys():
            self._use_log_x = True
        else:
            self._use_log_x = kwargs['use_log']
        if self._use_log_x:
            self._x = np.log10(self._x)
            self._xx = np.log10(self._xx)
        
        if not self.n_increases_with_x is False:
            self._x = np.sort(self._x)[::-1]
            self._xx = np.sort(self._x)[::-1]
        else:
            self._x = np.sort(self._x)
            self._xx = np.sort(self._xx)
        
        self._min_x = np.min(self._x)
        self._max_x = np.max(self._x)
        self._min_xx = np.min(self._xx)
        self._max_xx = np.max(self._xx)
        
        #build tabulated cumulative or differential depending on what is missing.
        if kwargs['type']=='cumulative':
            self._spline_n()
            self._diff_cum_n()
            self._spline_dn()
        elif kwargs['type']=='differential':
            self._spline_dn()
            self._integrate_diff_n()
            self._spline_n()
        else:
            msg = ("abundnace type keyword must be 'cumulative' or 'differential'.")
            raise ValueError()
        
        #extrapolate beyond tabulated values
        self._extrapolate_dn()
        self._extrapolate_n()
        
        AbundanceFunction.__init__(self)
    
    def _spline_dn(self):
        """
        spline the tabulated differential abundance function
        """
        
        if self.n_increases_with_x:
            #self._log_dn_func = interp1d(self._xx[::-1], self._log_dn[::-1], kind='linear')
            self._log_dn_func = InterpolatedUnivariateSpline(self._xx[::-1], self._log_dn[::-1], k=1)
        else:
            #self._log_dn_func = interp1d(self._xx, self._log_dn, kind='linear')
            self._log_dn_func = InterpolatedUnivariateSpline(self._xx, self._log_dn, k=1)
            
    def _spline_n(self):
        """
        spline the tabulated cumulative abundance function
        """
        
        if self.n_increases_with_x:
            #self._log_n_func = interp1d(self._x[::-1], self._log_n[::-1], kind='linear')
            self._log_n_func = InterpolatedUnivariateSpline(self._x[::-1], self._log_n[::-1], k=1)
        else:
            #self._log_n_func = interp1d(self._x, self._log_n, kind='linear')
            self._log_n_func = InterpolatedUnivariateSpline(self._x, self._log_n, k=1)
            
    def _extrapolate_dn(self):
        """
        Fit the upper and lower bounds of the tabulated differential abundance 
        function to create extrapolation.  The high abundnace end is fit with a 
        linear function, the low-abundnace end is fit with a linear + exponetial 
        drop-off.
        """
        
        #fit low abundance end
        def func_l(x, a, b, c, d):
            return -1.0*np.exp(a*x+b) + c*x + d
            
        a0 = 1.0 if self.n_increases_with_x else -1.0
        popt_l = curve_fit(func_l, self._xx[self._s_low],
            self._log_dn[self._s_low], [a0, 0.0, 0.0, 0.0], maxfev=100000)[0]
        self._ext_log_dn_func_l = lambda x: func_l(x, *popt_l)
        
        #fit high abundance end
        func_h = lambda x, a, b: a*x+b
        
        popt_h = curve_fit(func_h, self._xx[self._s_high],
            self._log_dn[self._s_high], [0.0, 0.0], maxfev=100000)[0]
        self._ext_log_dn_func_h = lambda x: func_h(x, *popt_h)
    
    def _extrapolate_n(self):
        """
        Fit the upper and lower bounds of the tabulated cumulative abundance 
        function to create extrapolation.  The high abundnace end is fit with a 
        linear function, the low-abundnace end is fit with a linear + exponetial 
        drop-off.
        """
        
        #fit low abundance end
        def func_l(x, a, b, c, d):
            return -1.0*np.exp(a*x+b) + c*x + d
            
        a0 = 1.0 if self.n_increases_with_x else -1.0
        popt_l = curve_fit(func_l, self._x[self._s_low],
            self._log_n[self._s_low], [a0, 0.0, 0.0, 0.0], maxfev=100000)[0]
        self._ext_log_n_func_l = lambda x: func_l(x, *popt_l)
        
        #fit high abundance end
        func_h = lambda x, a, b: a*x+b
        
        popt_h = curve_fit(func_h, self._x[self._s_high],
            self._log_n[self._s_high], [0.0, 0.0], maxfev=100000)[0]
        self._ext_log_n_func_h = lambda x: func_h(x, *popt_h)
        
    def _integrate_diff_n(self):
        """
        integrate a differential number density to get the cumulative number 
        density.
        """
        
        #set the initial value. This is somewhat arbitrary.
        init_value = self._dn[-1]*np.fabs(self._x[-1]-self._x[-2])
        
        self._n = np.fabs(integrate.cumtrapz(self._dn[::-1], self._x[::-1],
            initial=init_value))
        self._n = self._n[::-1]
        self._log_n = np.log10(self._n)
    
    def _diff_cum_n(self):
        """
        differentiate the cumulative number density to get the differential number 
        density
        """
        
        #sample the cumulative function finely
        self._xx = np.linspace(self._x[0],self._x[-2],1000)
        n = 10**(self._log_n_func(self._xx))
        
        #get the run
        dx = np.fabs(np.diff(self._xx))
        
        #calculate derivative
        self._dn = np.diff(n[::-1])[::-1]/dx
        self._xx = self._xx[:-1]
        
        #taking the diff shortens arrays by 1
        self._min_xx = np.min(self._xx)
        self._max_xx = np.max(self._xx)
        
        #calculate log10 of dn
        self._log_dn = np.log10(self._dn)
        
    def dn(self, x):
        """
        return the differential abundance
        """
        
        if self._use_log_x:
            x = np.log10(x)
        
        #determine if the galaxies/halos are inside the tabulated range
        mask_high = (x>self._max_xx)
        mask_low = (x<self._min_xx)
        mask_in_range = (x>=self._min_xx) & (x<=self._max_xx)
        
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
        
        if self._use_log_x:
            x = np.log10(x)
        
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


class AbundanceFunctionFromCallable(AbundanceFunction):
    """
    Galaxy/halo abundance function object from a callable function
    """
    def __init__(self, **kwargs):
        """
        n : callable
            callabel function returning number densities (cumulative or differential) 
            given a galaxy/halo property, ``x``.
        
        x : array_like
            abscissa sampling the relevant galaxy/halo property range with an appropriate 
            density.
        
        use_log : boolean
           bool indicating whether to use the log10(``x``).  Note that *log10(n)* should 
           roughly be linear in either ``x`` or log10(``x``). The default is to True.
        
        type : string
            string indicating: 'cumulative' or 'differential', corresponding to the input
            callable, ``n``
        
        n_increases_with_x : boolean, optional
            boolean indicating if abundance increases with increasing x.  
            The default is False.
        """
        
        if 'n_increases_with_x' not in kwargs.keys():
            self.n_increases_with_x = False
        else:
            if type(kwargs['n_increases_with_x']) is not bool:
                msg = "`n_increases_with_x` parameter must of type bool."
                raise ValueError(msg)
            self.n_increases_with_x = kwargs['n_increases_with_x']
        
        #tabulated abundance functions are stored from high to low, i.e. the way 
        #astronomers would plot it.
        
        if self.n_increases_with_x:
            self._x = np.sort(kwargs['x'])[::-1]
        else:
            self._x = np.sort(kwargs['x'])
        
        #set whether the log10 of x should be used
        if 'use_log' not in kwargs.keys():
            self._use_log_x = True
        else:
            if type(kwargs['use_log']) is not bool:
                msg = "`use_log` parameter must of type bool."
                raise ValueError(msg)
            self._use_log_x = kwargs['use_log']
        
        if kwargs['type']=='cumulative':
            self._type = 'cumulative'
            self._n_func = kwargs['n']
            self._log_n_func = lambda x: np.log10(self._n_func(x))
        elif kwargs['type']=='differential':
            self._type = 'differential'
            self._dn_func = kwargs['n']
            self._log_dn_func = lambda x: np.log10(self._dn_func(x))
        else:
            msg = ("abundance type keyword must be 'cumulative' or 'differential'.")
            raise ValueError(msg)
        
        #depending on input, calculate either the differential or cumulative functions
        if kwargs['type']=='cumulative':
            self._diff_cum_n()
        else:
            self._integrate_diff_n()
        
        #remove the last point because dn (n) is not known at that point when 
        #the passed in callable is cumulative (differential).
        self.x_abscissa = np.copy(self._x[:-1])
        
        AbundanceFunction.__init__(self)
    
    def _integrate_diff_n(self):
        """
        integrate a differential number density to get the cumulative number 
        density, n(<x)
        """
        
        #integrate from low to high density
        n = integrate.cumtrapz(self._dn_func(self._x)[::-1],self._x[::-1])
        if not self.n_increases_with_x:
            n = -1.0*n
        x = np.copy(self._x[:-1])
        log_n = np.log10(n)[::-1]
        
        #used for bounds checking when calling the interpolation
        self._min_x = np.min(x)
        self._max_x = np.max(x)
        
        if self._use_log_x:
            x = np.log10(x)
        
        #x must be monotonically increasing for the interpolation routine
        if self.n_increases_with_x:
            self._log_n_func = InterpolatedUnivariateSpline(x[::-1], log_n[::-1], k=1)
        else:
            self._log_n_func = InterpolatedUnivariateSpline(x, log_n, k=1)
        
        #use log10(x) as argument is appropriate
        if self._use_log_x:
            self._n_func = lambda x: 10**self._log_n_func(np.log10(x))
        else:
            self._n_func = lambda x: 10**self._log_n_func(x)
    
    def _diff_cum_n(self):
        """
        differentiate the cumulative number density to get the differential number 
        density, dn(x)/dx
        """
        
        dn = np.diff(self._dn_func(self._x)[::-1])[::-1]
        dx = np.fabs(np.diff(self._x))
        dndx = dn/dx
        
        x = self._x[:-1]
        
        #used for bounds checking when calling the interpolation
        self._min_x = np.min(x)
        self._max_x = np.max(x)
        
        if self._use_log_x:
            x = np.log10(x)
        
        if self.n_increases_with_x:
            self._log_dn_func = InterpolatedUnivariateSpline(x[::-1], dndx[::-1], k=1)
        else:
            self._log_dn_func = InterpolatedUnivariateSpline(x, dndx, k=1)
        
        if self._use_log_x:
            self._dn_func = lambda x: 10**self._log_n_func(np.log10(x))
        else:
            self._dn_func = lambda x: 10**self._log_n_func(x)
    
    def dn(self, x):
        """
        Return the differential number density of galaxies/halos, dn(x)/dx.
        
        Parameters 
        ----------
        x : numpy.array
            galaxy or halo properties, e.g. stellar mass or halo mass
        
        Returns 
        -------
        dn : numpy.array
            differential number density of galaxies/halos with property ``x``
        """
        
        if self._type == 'cumulative':
            out_of_abcissa_bounds = np.any(((x<self._min_x) | (x>self._max_x)))
            if out_of_abcissa_bounds:
                msg = ("Input out of interpolated abundance range. \n"
                       "Reinstantiate abundance function object with an \n"
                       "increased range in the `x` parameter which serves as \n"
                       "abcissa for the interpolation, or use a callable \n"
                       "differential function for the `n` parameter.")
                warn(msg)
        
        return self._dn_func(x)
    
    def n(self, x):
        """
        Return the cumulative number density of galaxies/halos, n(<x).
        
        Parameters 
        ----------
        x : numpy.array
            galaxy or halo properties, e.g. stellar mass or halo mass
        
        Returns 
        -------
        n : numpy.array
            cumulative number density of galaxies/halos with property ``x``
        """
        
        if self._type == 'differential':
            out_of_abcissa_bounds = np.any(((x<self._min_x) | (x>self._max_x)))
            if out_of_abcissa_bounds:
                msg = ("Input out of interpolated abundance range. \n"
                       "reinstantiate abundance function object with \n"
                       "increased range in the `x` parameter which serves as \n"
                       "abcissa for the interpolation, or use a callable \n"
                       "cumulative function for the `n` parameter.")
                warn(msg)
        
        return self._n_func(x)
        
        
