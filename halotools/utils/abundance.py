# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import, unicode_literals)
import numpy as np
from .array_utils import custom_len, array_is_monotonic
from ..custom_exceptions import HalotoolsError
from scipy import integrate

__all__ = ['AbundanceTabulator', 'empirical_cumu_ndensity', 'empirical_diff_ndensity']


class AbundanceTabulator(object):
    """
    Calculate tabulated differential and cumulative abundances of galaxies/halos.
    """
    def __init__(self, **kwargs):
        """
        class can be instantiated in three ways:
        1. by passing in galaxy/halo properties from a cataloge and the effective volume.
            see ``sample_x`` and ``sample_volume``
        2. by passing in tabulated cumulative or differental number densities for 
            galaxies/halos with property x.  see ``diff_nd_array``, ``cumu_nd_array``,
            and ``x_array``.
        3. or by passing in a callable differential number density fuction and a list of
            values at which to evalulate it at.  see ``diff_abun_function`` and 
            ``x_array``.
        
        Parameters
        ----------
        sample_x : array_like, optional 
            Array storing the values of the galaxy/halo property for every object 
            in the catalog whose abundances are being tabulated. 
            If ``sample_x`` is passed, ``sample_volume`` and ``xbins`` 
            must also be passed.
        
        sample_volume : float, optional 
            Float specifying the effective volume of the sample of galaxies/halos 
            in units of Mpc/h. 
            Required keyword argument when passing ``sample_x``. 
            For samples of halos, this is typically :math:`L_{\\rm box}^{3}`. 
        
        xbins : array_like, optional 
            Array defining the boundaries into which the input ``sample_x`` will be binned. 
            Required keyword argument when passing ``sample_x``.
        
        diff_abun_function : callable function object, optional 
            Function providing a map between the galaxy/halo property and the 
            differential abundance of that property in units of :math:`h^{3}/Mpc^{3}`. 
        
        diff_nd_array : array_like, optional 
            A monotonic array storing tabulated values of the differential number density 
            for galaxies/halos with values of a property given by ``x_array``.
        
        cumu_nd_array : array_like, optional 
            A monotonic array storing tabulated values of the cumulative number density 
            for galaxies/halos with values of a property given by ``x_array``.
        
        x_array : array_like, optional 
            A monotonic array of tabulated values of the galaxy/halo property associated 
            with ``cumu_nd_array`` or ``diff_nd_array``.
            Required keyword argument when passing ``cumu_nd_array`` and/or 
            ``diff_nd_array``.
        
        nd_increases_wtih_x : boolean, optional
            The number density increases with increasing ``x``, default is False.
        
        Notes
        -----
        Tabulated abundances are stored from high abundance to low abundance order.
        """
        
        self._check_kwargs_consistency(**kwargs)
        self._process_kwargs_parameters(**kwargs)
        
        #option 1: a sample of galaxy/halo properties and the effective volume
        if 'sample_x' in kwargs:
            sample_x = kwargs['sample_x']
            sample_volume = kwargs['sample_volume']
            xbins = kwargs['xbins']
            
            self.diff_nd, self.x = (
                empirical_diff_ndensity(sample_x, sample_volume, xbins=xbins)
                )
            
            self.cumu_nd, _ = (
                empirical_cumu_ndensity(sample_x, sample_volume, xbins=self.x,
                    nd_increases_wtih_x=self.nd_increases_wtih_x)
                )
        
        #option 2: a callable abundance function
        elif 'diff_abun_function' in kwargs:
            self.x = kwargs['sample_control_points']
            self.diff_nd = diff_abun_function(self.x)
            self._integrate_diff_ndensity() #also sets self.cumu_nd
        
        #option 3: tabulated abundances supplied
        elif 'diff_nd_array' in kwargs:
            self.x = kwargs['x_array']
            self.diff_nd = kwargs['diff_nd_array']
        
            try:
                self.cumu_nd = kwargs['cumu_nd_array']
            except KeyError:
                ### Need to compute cumulative function by integration 
        elif 'cumu_nd_array' in kwargs:
            self.x = kwargs['x_array']
            self.cumu_nd_array = kwargs['cumu_nd_array']
        
            ### Need to compute differential function from cumulative one (see Yao's code) 
        
        #an unacceptable combination of parameters supplied.
        else:
            msg = ("Insufficient parameters supplied to calculate abundances. \n"
                   "Cannot instaniate an AbundanceTabulator object. See parameters.")
            raise HalotoolsError(msg)
    
    def diff_number_density(self, x):
        """
        return the differential number densities for objects with properties x.
        """
    
    def cumu_number_density(self, x):
        """
        return the cumulative number densities for objects with properties x.
        """
    
    def _integrate_diff_ndensity(self):
        """
        integrate a callable differential number density function to get the cumulative
        number density.
        """
        
        #integrate from low density to high density, remembering that our convention is
        #to store the tabulate abundances from high density to low density order.
        result = integrate.cumtrapz(self.diff_nd[::-1], self._x[::-1], initial=0.0)
        self.cumu_nd = result[::-1]
    
    def _process_kwargs_parameters(self, **kwargs):
        """ 
        Private method to process the keyword arguments that have been passed.
        """
        
        #check bool indicating which direction the number density increaases.
        if 'nd_increases_wtih_x' in kwargs:
           if type(kwargs['nd_increases_wtih_x']) is not bool:
                msg = "Input ``nd_increases_wtih_x`` must be a boolean type: True or False."
                raise HalotoolsError(msg)
            self.nd_increases_wtih_x = kwargs['nd_increases_wtih_x']
        
        #put x_array in the correct order
        if 'x_array' in kwargs:
            if not array_is_monotonic(kwargs['x_array']):
                msg = ('``x_array must be monotonic.``')
            if not self.nd_increases_wtih_x:
                kwargs['x_array'] = np.sort(kwargs['x_array'])
            else:
                kwargs['x_array'] = np.sort(kwargs['x_array'])[::-1]
        
        #put xbins in the correct order
        if 'xbins' in kwargs:
            if not array_is_monotonic(kwargs['xbins']):
                msg = ('``xbins must be monotonic.``')
            if not self.nd_increases_wtih_x:
                kwargs['xbins'] = np.sort(kwargs['xbins'])
            else:
                kwargs['xbins'] = np.sort(kwargs['xbins'])[::-1]
    
    def _check_kwargs_consistency(self, **kwargs):
        """ 
        Private method to test that a self-consistent set of keyword arguments 
        have been passed to the constructor.
        """
        
        if 'diff_abun_function' in kwargs:
            diff_abun_function = kwargs['diff_abun_function']
            try:
                assert callable(diff_abun_function)
            except AssertionError:
                msg = "Input ``diff_abun_function`` must be a callable function"
                raise HalotoolsError(msg)

            try:
                _ = kwargs['sample_control_points']
            except KeyError:
                msg = ("If passing an input ``diff_abun_function`` to the ``AbundanceTabulator``, \n"
                    "you must also pass an input ``sample_control_points``.\n")
                raise HalotoolsError(msg)

        if 'sample_x' in kwargs:
            try:
                sample_volume = kwargs['sample_volume']
                xbins = kwargs['xbins']
            except KeyError:
                raise HalotoolsError("\nIf providing the ``sample_x`` input "
                    "to ``AbundanceTabulator``,\n "
                    "you must provide a ``sample_volume`` and ``xbins`` keyword argument.\n")

        if 'diff_nd_array' in kwargs:
            try:
                x_array = kwargs['x_array']
            except KeyError:
                msg = ("If passing an input ``diff_nd_array`` to the ``AbundanceTabulator``, \n"
                    "you must also pass an input ``x_array`` keyword argument.\n")
                raise HalotoolsError(msg)






def empirical_cumu_ndensity(x, volume, xbins = None, weights = None, 
                            nd_increases_wtih_x = False):
    """
    Caclulate cumulative number density of galaxies/haloes given a property ``x``.
    
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
        the number density increases with increasing ``x``
    
    Returns
    -------
    cumu_x : numpy.array
        cumulative number desntiy at values ``x_centers``
    
    x_centers : numpy.array
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
        for i,ind in enumerate(inds):
            cumu_x[i] = Nx_weighed - np.sum(sorted_weights[:inds])
        x_centers = xbins
    
    return cumu_x, x_centers


def empirical_diff_ndensity(x, volume, xbins, weights = None):
    """
    Caclulate differential number density of galaxies/haloes given a property ``x``.
    
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
    
    