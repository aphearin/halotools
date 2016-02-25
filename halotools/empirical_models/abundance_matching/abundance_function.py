# -*- coding: utf-8 -*-

"""
classes used to create galaxy and halo abundance functions used in the 
abundance matching module.
"""

from __future__ import (division, print_function, absolute_import, unicode_literals)
import numpy as np
from ...utils.array_utils import custom_len, array_is_monotonic, convert_to_ndarray
from ...custom_exceptions import HalotoolsError
from scipy import integrate
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.interpolate import InterpolatedUnivariateSpline
import six 
from abc import ABCMeta, abstractmethod
from warnings import warn, catch_warnings, simplefilter
from copy import deepcopy, copy 

from .deconvolution import abunmatch_deconvolution

__all__ = ['AbundanceFunction',
           'AbundanceFunctionFromTabulated',
           'AbundanceFunctionFromCallable']
__author__=['Duncan Campbell', 'Andrew Hearin']

def _convolve_gaussian(y, sigma, truncate=4):
    sd = float(sigma)
    size = int(np.ceil(truncate * sd))
    weights = np.zeros(size*2+1)
    i = np.arange(size+1)
    weights[size:] = np.exp(-(i*i)/(2.0*sd*sd))
    weights[:size] = weights[:size:-1]
    weights /= weights.sum()
    y_full = np.concatenate((np.zeros(size), y, np.ones(size)*y[-1]))
    return np.convolve(y_full, weights, 'valid')

@six.add_metaclass(ABCMeta)
class AbundanceFunction(object):
    """
    abstract class that has callable galaxy/halo differential and cumulative abundance functions.
    """
    
    @abstractmethod
    def n(self,x):
        """
        Return the cumulative number density of galaxies/halos, i.e. n(<x) 
        
        Parameters 
        ----------
        x : numpy.array
            galaxy or halo properties, e.g. stellar mass or halo mass
        
        Returns 
        -------
        n : numpy.array
            cumulative number density of galaxies/halos with property x
        
        Notes
        -----
        Note that this could instead be n(>x) if we are dealing with a preverse 
        quantity like magnitudes.
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
        
        Notes
        -----
        This is not the same as dn(x)/dx * dx, the number density of objects 
        with property between [x,x+dx], which is often the quanity that is plotted.
        """
        
        msg = "All subclasses of AbundanceFunction must include a `dn` method."
        raise NotImplementedError(msg)
    
    #check for required attributes
    def __init__(self, **kwargs):
        self._constructor_kwargs = kwargs
        
        required_attrs = ["n_increases_with_x","x_abscissa","_use_log10_x"]
        for attr in required_attrs:
            try:
                assert hasattr(self, attr)
            except AssertionError:
                msg = ("All subclasses of Parent must have the following \n"
                       "attributes: 'n_increases_with_x', 'x_abscissa','_use_log10_x'")
                raise HalotoolsError(msg)
    
    def compute_deconvolved_galaxy_abundance_function(self, scatter, x_range, x_pad,
                                                      repeat = 40,
                                                      x_step = 0.01,
                                                      remainder_tol = 0.5,
                                                      **kwargs):
        """
        Deconvolve the input ``scatter`` from the `AbundanceFunction` instance.
        
        Parameters
        ----------
        scatter : float 
            Level of scatter in dex
        
        x_range : tuple
            The range, minimum and maximum galaxy/halo proerty, over which to complete
            the deconvolution.
        
        x_pad : tuple
            extended range used for padding in the deconvolution.
            This should encompass ``x_range``.
        
        repeat : int, optional 
            Number of iterations in the deconvolution algorithm. Default is 40. 
        
        x_step : float, optional 
            Step size in the galaxy/halo property to use in the deconvolution algorithm. 
            Default is 0.01.  If ``use_log10`` was set to True for 
            the `AbundanceFunction` instance, then log spacing is used.
        
        remainder_tol : float, optional 
            Tolerance for the fractional difference between the input SMF 
            and the deconvolved SMF with the appropriate scatter added back in. 
            Used as a sanity check on the Richardson-Lucy deconvolution. 
            Default is 0.5.
        
        kwargs : dictionary
            additional keyword arguments to pass to AbundanceFunctionFromTabulated init
        
        Returns 
        --------
        deconvolved_galaxy_abundance_function : object 
            Instance of the `AbundanceFunction` class in which the input scatter 
            has been deconvolved. Thus when the input ``scatter`` is applied 
            to the returned ``deconvolved_galaxy_abundance_function``, 
            you get the original `AbundanceFunction` back. 
        
        Notes 
        -------
        The `compute_deconvolved_galaxy_abundance_function` method is just a 
        wrapper around Peter Behroozi's C-implementation of the Richardson-Lucy 
        deconvolution algorithm.
        
        It is not always possible to sucessfully deconvolve the ``scatter``.
        The user should always examine the result.
        """
        
        #if scatter is 0, do nothing.
        if scatter == 0:
            return self.__class__(**self._constructor_kwargs)
        #otherwise, after some processing, call the deconvolution wrapper.
        else:
            ######################################################################
            # We want to calculate a de-convolved abundance function that is valid
            # over a certain range indicated by 'x_range'. 
            # To do this, we need to pad the function on either end to minimize 
            # edge effects in the deconvolution. This done using the range 
            # indicated by x_pad.  At the end, we will trim the padded region.
            
            x_pad = convert_to_ndarray(x_pad, dt=np.float)
            if len(x_pad)!=2:
                msg = ("``x_pad`` parameter must have lenght 2.")
                raise ValueError(msg)
            x_range = convert_to_ndarray(x_range, dt=np.float)
            if len(x_range)!=2:
                msg = ("``x_range`` parameter must have lenght 2.")
                raise ValueError(msg)
            
            #calculate minimums and maximums
            min_x_pad = min(x_pad)
            max_x_pad = max(x_pad)
            min_x_range = min(x_range)
            max_x_range = max(x_range)
            
            if min_x_pad > min_x_range:
               msg = ("``x_pad`` must encompass ``x_range``.")
            if max_x_pad < max_x_range:
               msg = ("``x_pad`` must encompass ``x_range``.")
            
            #calculate x abscissa to use for deconvolution.
            if self._use_log10_x:
                x_abscissa = np.arange(np.log10(min_x_pad),np.log10(max_x_pad),x_step)
                x_abscissa = 10**x_abscissa
            else:
                x_abscissa = np.arange(min_x_pad,man_x_pad,x_step)
            if self.n_increases_with_x:
                x_abscissa = x_abscissa[::-1]
            
            log10_x_abscissa = np.log10(x_abscissa)
            ######################################################################
            
            ######################################################################
            # Now we will one by one initialize the following four arrays 
            # so that they store the variables required by the C code:
            # 1. af_key, 2. af_val, 3. smm, 4. mf
            
            # C code convention 1
            # All logarithmic quantities are in base-10 
            
            # C code convention 2
            # Rare, high-mass entries must be stored at the end of the array. 
            
            # C code convention 3
            # The values stored in the abscissa array must be monotonically increasing
            # For the case of abundance matching on absolute magnitudes, 
            # this assumption is incompatible with Convention 2 
            # To work around this, we use the following hack: 
            # if using magnitudes, we manually multiply the abscissa values by -1, 
            # call the C code, and then manually mutiply the returned values by -1
            
            # C code convention 4
            # The array returned by fiducial_deconvolute preserves input label. 
            # So if log10 --> in, then log10 --> out
            # mf just stores differential number density, that's it
            
            ###############
            # 1. af_key
            af_key = np.copy(log10_x_abscissa)
            if self.n_increases_with_x is True: af_key *= -1.0
            
            ###############
            # 2. af_val 
            dn_dx_abscissa = self.dn(x_abscissa)
            dn_dlog10x_abscissa = x_abscissa*np.log(10)*dn_dx_abscissa
            af_val = np.log10(dn_dlog10x_abscissa) 
            
            ###############
            # 3. smm
            smm = np.copy(log10_x_abscissa)
            if self.n_increases_with_x is True: smm *= -1.0
            
            ###############
            # 4. mf
            mf = dn_dlog10x_abscissa*abs(log10_x_abscissa[1]-log10_x_abscissa[0])
            ######################################################################
            
            ######################################################################
            # Now we will call the C-wrapper with the above defined arrays
            
            deconvolved_log10_x_abscissa = abunmatch_deconvolution(
                af_key, af_val, smm, mf, scatter, repeat = repeat)
            if self.n_increases_with_x is True: deconvolved_log10_x_abscissa *= -1.0
            
            #calculate the remainder
            nd = 10.**np.interp(np.log10(x_abscissa), 
                deconvolved_log10_x_abscissa, np.log10(dn_dx_abscissa))
            dlog10x = (np.fabs((log10_x_abscissa[-1] - log10_x_abscissa[0])/
                float(len(log10_x_abscissa)-1)))
            nd_conv = _convolve_gaussian(nd, float(scatter)/dlog10x)
            frac_remainder = abs((nd_conv - dn_dx_abscissa)/dn_dx_abscissa)
            
            #convert back to non-log units
            deconvolved_x_abscissa = 10**deconvolved_log10_x_abscissa
            
            #extract solution within the desired range indicated by 'x_range' parameter
            keep = (deconvolved_x_abscissa >= min_x_range) & (deconvolved_x_abscissa <= max_x_range)
            deconvolved_x_abscissa = deconvolved_x_abscissa[keep]
            frac_remainder = frac_remainder[keep]
            deconvolved_dn = dn_dx_abscissa[keep]
            
            #check to see if deconvolution was succesful over the desired range
            mask = (frac_remainder < remainder_tol)
            if np.sum(mask)<len(mask):
                msg = ("Deconvolution resulted in a solution that is not within the \n"
                       "tolerance over some range of the ``x_range``. \n"
                       "The resulting AbundanceFunction instance should be examined. \n"
                       "Only the abscissa within the tolerance will be used to \n"
                       "construct the AbundanceFunction instance.")
                warn(msg)
             
            #only use the successful range to build the abundance function
            deconvolved_x_abscissa = deconvolved_x_abscissa[mask]
            deconvolved_dn = deconvolved_dn[mask]
            
            #build from the tabulated data
            args = {'n':deconvolved_dn,
                    'x':deconvolved_x_abscissa,
                    'use_log10':self._use_log10_x,
                    'abundance_type':'differential',
                    'n_increases_with_x':self.n_increases_with_x}
            
            return AbundanceFunctionFromTabulated(**args)


class AbundanceFunctionFromTabulated(AbundanceFunction):
    """
    Galaxy/halo abundance function object from tabulated data.
    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        n : array_like
            tabulated number densities (cumulative or differential).
        
        x : array_like
            tabulated galaxy/halo property.
        
        abundance_type : string
            'cumulative' or 'differential'
        
        use_log10 : boolean, optional
           bool indicating whether to use the log10(``x``).  Note that *log10(n)* should 
           roughly be linear in log10(``x``). The default is to True.
        
        n_increases_with_x : boolean, optional
            boolean indicating if abundance increases with increasing x.  
            The default is False.  This is True for e.g. magnitudes.
        
        num_points_to_fit_high_abundance : int, optional
            The number of input tabulated points to use when fitting the high abundance 
            end of the abundnace function for extrapolation.  The default is 3.
        
        num_points_to_fit_low_abundance : int, optional
            The number of input tabulated points to use when fitting the low abundance 
            end of the abundnace function for extrapolation.  The default is 7.
        
        Notes
        -----
        The tabulated abundances are fit with a linear spline.  Beyond the tabulated
        range, extrapolations are used: linear for the low-abundance end, and 
        linear+exponential drop-off for the high-abundnace end.
        """
        
        #process input arguments, and apply defaults
        if 'n_increases_with_x' not in kwargs.keys():
            self.n_increases_with_x = False
        else:
            if type(kwargs['n_increases_with_x']) is not bool:
                msg = "`n_increases_with_x` parameter must of type bool."
                raise ValueError(msg)
            self.n_increases_with_x = kwargs['n_increases_with_x']
        
        if 'num_points_to_fit_high_abundance' not in kwargs.keys():
            self._num_points_to_fit_high_abundance = 3
        else:
            self._num_points_to_fit_high_abundance = kwargs['num_points_to_fit_high_abundance']
            if self._num_points_to_fit_high_abundance<3:
                msg = ("''num_points_to_fit_high_abundance'' must be >= 3")
                raise ValueError(msg)
        
        if 'num_points_to_fit_low_abundance' not in kwargs.keys():
            self._num_points_to_fit_low_abundance = 7
        else:
            self._num_points_to_fit_low_abundance = kwargs['num_points_to_fit_low_abundance']
            if self._num_points_to_fit_low_abundance<7:
                msg = ("''num_points_to_fit_low_abundance'' must be >= 7")
                raise ValueError(msg)
        
        #define slice arrays that access points to use for fitting extrapolation functions
        #last N elements
        self._s_low = slice(-self._num_points_to_fit_low_abundance,None,None)
        #first N elements
        self._s_high = slice(self._num_points_to_fit_high_abundance)
        
        #check the 'x' and 'n' arguments
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
        
        # We always store the tabulated abundance functions from high to low abundance,
        # the same ordering as they are always plotted
        n = np.array(kwargs['n'])
        sort_inds = np.argsort(n)[::-1]
        x = np.array(kwargs['x'])
        n = n[sort_inds]
        x = x[sort_inds]
        
        self._x = copy(x)
        
        if kwargs['abundance_type']=='cumulative':
            self._n = copy(n)
            self._log10_n = np.log10(self._n)
        elif kwargs['abundance_type']=='differential':
            self._dn = copy(n)
            self._log10_dn = np.log10(self._dn)
        
        #put x in the correct order
        if self.n_increases_with_x:
            self._x = np.sort(self._x)[::-1]
        
        #store the values as x_abscissa
        self.x_abscissa = copy(self._x)
        
        #set whether the log10 of x should be used
        if 'use_log10' not in kwargs.keys():
            self._use_log10_x = True
        else:
            self._use_log10_x = kwargs['use_log10']
        
        #store minimum and maximum values
        self._min_x = np.min(self._x)
        self._max_x = np.max(self._x)
        
        #calculate cumulative or differential abundance depending on what was passed.
        #also calculate splines.
        if kwargs['abundance_type']=='cumulative':
            self._spline_n()
            self._diff_cum_n()
            self._spline_dn()
        elif kwargs['abundance_type']=='differential':
            self._spline_dn()
            self._integrate_diff_n()
            self._spline_n()
        else:
            msg = ("abundance type keyword must be 'cumulative' or 'differential'.")
            raise ValueError(msg)
        
        #extrapolate beyond tabulated values
        self._extrapolate_dn()
        self._extrapolate_n()
        
        AbundanceFunction.__init__(self, **kwargs)
    
    def _spline_dn(self):
        """
        spline the tabulated differential abundance function
        """
        
        x = copy(self._x)
        if self._use_log10_x:
            x = np.log10(x)
        
        if self.n_increases_with_x:
            self._log10_dn_func = InterpolatedUnivariateSpline(x[::-1], self._log10_dn[::-1], k=1)
        else:
            self._log10_dn_func = InterpolatedUnivariateSpline(x, self._log10_dn, k=1)
        
        if self._use_log10_x:
            self._dn_func = lambda x: 10**self._log10_dn_func(np.log10(x))
        else:
            self._dn_func = lambda x: 10**self._log10_dn_func(x)
        
    def _spline_n(self):
        """
        spline the tabulated cumulative abundance function
        """
        
        x = copy(self._x)
        if self._use_log10_x:
            x = np.log10(x)
        
        if self.n_increases_with_x:
            self._log10_n_func = InterpolatedUnivariateSpline(x[::-1], self._log10_n[::-1], k=1)
        else:
            self._log10_n_func = InterpolatedUnivariateSpline(x, self._log10_n, k=1)
        
        if self._use_log10_x:
            self._n_func = lambda x: 10**self._log10_n_func(np.log10(x))
        else:
            self._n_func = lambda x: 10**self._log10_n_func(x)
        
    def _extrapolate_dn(self):
        """
        Fit the upper and lower bounds of the tabulated differential abundance 
        function to create extrapolation.  The high abundnace end is fit with a 
        linear function, the low-abundnace end is fit with a linear + exponetial 
        drop-off.
        """
        
        x = copy(self._x)
        if self._use_log10_x:
            x = np.log10(x)
        
        #check for fitting issues
        with catch_warnings():
            simplefilter("error", OptimizeWarning)
            
            #fit low abundance end
            def func_l(x, a, b, c, d):
                return -1.0*np.exp(a*x+b) + c*x + d
            
            a0 = -1.0 if self.n_increases_with_x else 1.0
            try:
                popt_l = curve_fit(func_l, x[self._s_low],
                    self._log10_dn[self._s_low], [a0, 0.0, 0.0, 0.0], maxfev=100000)[0]
            except OptimizeWarning:
                msg = ("extrapolation of high abundance end of the differential abundance function may have failed.")
                warn(msg)
            self._ext_log10_dn_func_l = lambda x: func_l(x, *popt_l)
            
            #fit high abundance end
            func_h = lambda x, a, b: a*x+b
            
            try:
                popt_h = curve_fit(func_h, x[self._s_high],
                    self._log10_dn[self._s_high], [0.0, 0.0], maxfev=100000)[0]
            except OptimizeWarning:
                msg = ("extrapolation of high abundance end of the differential abundance function may have failed.")
                warn(msg)
            self._ext_log10_dn_func_h = lambda x: func_h(x, *popt_h)
        
        if self._use_log10_x:
            self._ext_dn_func_h = lambda x: 10**self._ext_log10_dn_func_h(np.log10(x))
            self._ext_dn_func_l = lambda x: 10**self._ext_log10_dn_func_l(np.log10(x))
        else:
            self._ext_dn_func_h = lambda x: 10**self._ext_log10_dn_func_h(x)
            self._ext_dn_func_l = lambda x: 10**self._ext_log10_dn_func_l(x)
    
    def _extrapolate_n(self):
        """
        Fit the upper and lower bounds of the tabulated cumulative abundance 
        function to create extrapolation.  The high abundnace end is fit with a 
        linear function, the low-abundnace end is fit with a linear + exponetial 
        drop-off.
        """
        
        x = copy(self._x)
        if self._use_log10_x:
            x = np.log10(x)
        
        #check for fitting issues
        with catch_warnings():
            simplefilter("error", OptimizeWarning)
            
            #fit low abundance end
            def func_l(x, a, b, c, d):
                return -1.0*np.exp(a*x+b) + c*x + d
            
            a0 = -1.0 if self.n_increases_with_x else 1.0
            try:
                popt_l = curve_fit(func_l, x[self._s_low],
                    self._log10_n[self._s_low], [a0, 0.0, 0.0, 0.0], maxfev=100000)[0]
            except OptimizeWarning:
                msg = ("Extrapolation of high abundance end of the cumulative abundance function may have failed.")
                warn(msg)
            self._ext_log10_n_func_l = lambda x: func_l(x, *popt_l)
            
            #fit high abundance end
            func_h = lambda x, a, b: a*x+b
            
            try:
                popt_h = curve_fit(func_h, x[self._s_high],
                    self._log10_n[self._s_high], [0.0, 0.0], maxfev=100000)[0]
            except OptimizeWarning:
                msg = ("Extrapolation of low abundance end of the cumulative abundance function may have failed.")
                warn(msg)
            self._ext_log10_n_func_h = lambda x: func_h(x, *popt_h)
        
        if self._use_log10_x:
            self._ext_n_func_h = lambda x: 10**self._ext_log10_n_func_h(np.log10(x))
            self._ext_n_func_l = lambda x: 10**self._ext_log10_n_func_l(np.log10(x))
        else:
            self._ext_n_func_h = lambda x: 10**self._ext_log10_n_func_h(x)
            self._ext_n_func_l = lambda x: 10**self._ext_log10_n_func_l(x)
        
    def _integrate_diff_n(self):
        """
        integrate a differential number density to get the cumulative number 
        density.
        """
        
        #set the initial value. This is somewhat arbitrary and annoying.
        init_value = self._dn[-1]*(self._x[-2]-self._x[-1])
        
        self._n = integrate.cumtrapz(self._dn[::-1], self._x[::-1], initial=init_value)
        
        if not self.n_increases_with_x:
            self._n = -1.0*self._n
        
        self._n = self._n[::-1]
        self._log10_n = np.log10(self._n)
    
    def _diff_cum_n(self):
        """
        differentiate the cumulative number density to get the differential number 
        density
        """
        
        n = copy(self._n_func(self._x))
        
        dx = np.gradient(self._x)
        dn = np.gradient(n[::-1])[::-1]
        
        #calculate derivative
        self._dn = dn/dx
        
        #calculate log10 of dn
        self._log10_dn = np.log10(self._dn)
        
    def dn(self, x):
        """
        return the differential abundance
        """
        
        x = convert_to_ndarray(x)
        
        #determine if the galaxies/halos are inside the tabulated range
        mask_high = (x>self._max_x)
        mask_low = (x<self._min_x)
        mask_in_range = (x>=self._min_x) & (x<=self._max_x)
        
        #initialize the result
        result = np.zeros(len(x))
        result[mask_in_range] = self._dn_func(x[mask_in_range])
        
        #call the interpolation functions if necessary
        if self.n_increases_with_x:
            result[mask_high] = self._ext_dn_func_h(x[mask_high])
            result[mask_low] = self._ext_dn_func_l(x[mask_low])
        else:
            result[mask_high] = self._ext_dn_func_l(x[mask_high])
            result[mask_low] = self._ext_dn_func_h(x[mask_low])
        
        return result
    
    def n(self, x):
        """
        return the cumulative abundance
        """
        
        x = convert_to_ndarray(x)
        
        #determine if the galaxies/halos are inside the tabulated range
        mask_high = (x>self._max_x)
        mask_low = (x<self._min_x)
        mask_in_range = (x>=self._min_x) & (x<=self._max_x)
        
        #initialize the result
        result = np.zeros(len(x))
        result[mask_in_range] = self._n_func(x[mask_in_range])
        
        #call the interpolation functions if necessary
        if self.n_increases_with_x:
            result[mask_high] = self._ext_n_func_h(x[mask_high])
            result[mask_low] = self._ext_n_func_l(x[mask_low])
        else:
            result[mask_high] = self._ext_n_func_l(x[mask_high])
            result[mask_low] = self._ext_n_func_h(x[mask_low])
        
        return result


class AbundanceFunctionFromCallable(AbundanceFunction):
    """
    Galaxy/halo abundance function object from a callable function
    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        n : callable
            callable function returning number densities (cumulative or differential) 
            given a galaxy/halo property, ``x``.
        
        x : array_like
            abscissa sampling the relevant galaxy/halo property range with an appropriate 
            density.
        
        abundance_type : string
            'cumulative' or 'differential'
        
        use_log10 : boolean, optional
           bool indicating whether to use the log10(``x``).  Note that log10(``n``) 
           should roughly be linear in either ``x`` or log10(``x``).
           The default is to True.
        
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
        if 'use_log10' not in kwargs.keys():
            self._use_log10_x = True
        else:
            if type(kwargs['use_log10']) is not bool:
                msg = "`use_log10` parameter must of type bool."
                raise ValueError(msg)
            self._use_log10_x = kwargs['use_log10']
        
        if kwargs['abundance_type']=='cumulative':
            self._type = 'cumulative'
            self._n_func = kwargs['n']
            self._log10_n_func = lambda x: np.log10(self._n_func(x))
        elif kwargs['abundance_type']=='differential':
            self._type = 'differential'
            self._dn_func = kwargs['n']
            self._log10_dn_func = lambda x: np.log10(self._dn_func(x))
        else:
            msg = ("abundance type keyword must be 'cumulative' or 'differential'.")
            raise ValueError(msg)
        
        #depending on input, calculate either the differential or cumulative functions
        if kwargs['abundance_type']=='cumulative':
            self._diff_cum_n()
        else:
            self._integrate_diff_n()
        
        #remove the last point because dn (n) is not known at that point when 
        #the passed in callable is cumulative (differential).
        self.x_abscissa = np.copy(self._x[:-1])
        
        AbundanceFunction.__init__(self, **kwargs)
    
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
        log10_n = np.log10(n)[::-1]
        
        #used for bounds checking when calling the interpolation
        self._min_x = np.min(x)
        self._max_x = np.max(x)
        
        if self._use_log10_x:
            x = np.log10(x)
        
        #x must be monotonically increasing for the interpolation routine
        if self.n_increases_with_x:
            self._log10_n_func = InterpolatedUnivariateSpline(x[::-1], log10_n[::-1], k=1)
        else:
            self._log10_n_func = InterpolatedUnivariateSpline(x, log10_n, k=1)
        
        #use log10(x) as argument is appropriate
        if self._use_log10_x:
            self._n_func = lambda x: 10**self._log10_n_func(np.log10(x))
        else:
            self._n_func = lambda x: 10**self._log10_n_func(x)
    
    def _diff_cum_n(self):
        """
        differentiate the cumulative number density to get the differential number 
        density, dn(x)/dx
        """
        
        dn = np.diff(self._dn_func(self._x)[::-1])[::-1]
        dx = np.fabs(np.diff(self._x))
        dndx = dn/dx
        
        x = np.copy(self._x[:-1])
        
        #used for bounds checking when calling the interpolation
        self._min_x = np.min(x)
        self._max_x = np.max(x)
        
        if self._use_log10_x:
            x = np.log10(x)
        
        if self.n_increases_with_x:
            self._log10_dn_func = InterpolatedUnivariateSpline(x[::-1], dndx[::-1], k=1)
        else:
            self._log10_dn_func = InterpolatedUnivariateSpline(x, dndx, k=1)
        
        if self._use_log10_x:
            self._dn_func = lambda x: 10**self._log10_n_func(np.log10(x))
        else:
            self._dn_func = lambda x: 10**self._log10_n_func(x)
    
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
            out_of_abscissa_bounds = np.any(((x<self._min_x) | (x>self._max_x)))
            if out_of_abscissa_bounds:
                msg = ("Input out of interpolated abundance range. \n"
                       "Reinstantiate abundance function object with an \n"
                       "increased range in the `x` parameter which serves as \n"
                       "abscissa for the interpolation, or use a callable \n"
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
            out_of_abscissa_bounds = np.any(((x<self._min_x) | (x>self._max_x)))
            if out_of_abscissa_bounds:
                msg = ("Input out of interpolated abundance range. \n"
                       "reinstantiate abundance function object with \n"
                       "increased range in the `x` parameter which serves as \n"
                       "abscissa for the interpolation, or use a callable \n"
                       "cumulative function for the `n` parameter.")
                warn(msg)
        
        return self._n_func(x)
        
        
