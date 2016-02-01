# -*- coding: utf-8 -*-

"""
schecter function models for galaxy/halo abundances
"""

from __future__ import (division, print_function, absolute_import, unicode_literals)
import numpy as np
from astropy.modeling.models import custom_model

__all__=['schechter','super_schechter','log10_schechter','log10_super_schechter','mag_schechter']

__author__=['Duncan Campbell']

@custom_model
def schechter(x, phi0=10**(-5), x0=10.0**12.0, alpha=-1):
    """
    Schechter function astropy.modeling.model object to model dn(x)/dx.
    
    Parameters
    ----------
    x : array_like
        value(s) of galaxy/halo properties
    
    phi0 : float, optional
        parameter of the model, normalization
    
    x0 : float, optional
        parameter of the model, characteristic value of parameter 
        of galaxy/halo property
    
    alpha : float, optional
        parameter of the model, high abundnace end slope
    
    Returns
    -------
    phi : numpy.array
        differential abundance of galaxies/haloes at input ``x`` values, dn(x)/dx
    
    Notes
    -----
    .. math::
        \\phi(x) = \\phi_0\\left(\\frac{x}{x_0}\\right)^{\\alpha}\\exp{-x/x_0}
    """
    
    x = np.asarray(x)
    x = x.astype(float)
    
    norm = phi0/x0
    val = norm * (x/x0)**(alpha) * np.exp(-x/x0)
    
    return val


@custom_model
def super_schechter(x, phi0=10**(-5), x0=10.0**12.0, alpha=-1.0, beta=0.5):
    """
    'super' Schecter function astropy.modeling.model object to model dn(x)/dx
    
    Parameters
    ----------
    x : array_like
        value(s) of galaxy/halo properties
    
    phi0 : float, optional
        parameter of the model, normalization
    
    x0 : float, optional
        parameter of the model, characteristic value of parameter 
        of galaxy/halo property
    
    alpha : float, optional
        parameter of the model, high abundnace end slope
    
    beta : float, optional
        parameter of the model, exponential power
    
    Returns
    -------
    phi : numpy.array
        abundance of galaxies/haloes at input ``x`` values
    
    Notes
    -----
    .. math::
        \\phi(x)\\mathrm{d}x = \\phi_0\\left(\\frac{x}{x_0}\\right)^{\\alpha}\\exp{-x/x_0}^{\\beta}
    """
    
    x = np.asarray(x)
    x = x.astype(float)
    norm = phi0
    val = norm * (x/x0)**(alpha) * np.exp(-(x/x0))**beta
    return val

@custom_model
def log10_schechter(x, phi0=10**(-5), x0=12.0, alpha=-1.0):
    """
    Schechter function astropy.modeling.model to model dn(log10(x))/dlog10(x).
    
    Parameters
    ----------
    x : array_like
        log10 value(s) of galaxy/halo properties
    
    phi0 : float, optional
        parameter of the model, normalization
    
    x0 : float, optional
        parameter of the model, characteristic value of parameter 
        of galaxy/halo property
    
    alpha : float, optional
        parameter of the model, high abundnace end slope
    
    Returns
    -------
    phi : numpy.array
        abundance of galaxies/haloes at input ``x`` values
    
    Notes
    -----
    .. math::
        \\phi(x) = \\log10(10)\\phi_0\\left(10^{(x-x_0)(1+\\alpha)}\\right)\\exp{-10^{x-x_0)}}
    """
    
    x = np.asarray(x)
    x = x.astype(float)
    norm = np.log10(10.0)*phi0
    val = norm*(10.0**((x-x0)*(1.0+alpha)))*np.exp(-10.0**(x-x0))
    return val

@custom_model
def log10_super_schechter(x, phi0=10**(-5), x0=12.0, alpha=-1.0, beta=0.5):
    """
    'super' Schecter function astropy.modeling.model object to model dn(log10(x))/dlog10(x).
    
    Parameters
    ----------
    x : array_like
        value(s) of galaxy/halo properties
    
    phi0 : float, optional
        parameter of the model, normalization
    
    x0 : float, optional
        parameter of the model, characteristic value of parameter 
        of galaxy/halo property
    
    alpha : float, optional
        parameter of the model, high abundnace end slope
    
    beta : float, optional
        parameter of the model, exponential power
    
    Returns
    -------
    phi : numpy.array
        abundance of galaxies/haloes at input ``x`` values
    
    Notes
    -----
    .. math::
        \\phi(x) = \\log10(10)\\phi_0\\left(\\frac{x}{x_0}\\right)^{\\alpha}\\exp{-x/x_0}^{\\beta}
    """
    
    x = np.asarray(x)
    x = x.astype(float)
    norm = np.log10(10.0)*phi0
    val = norm*(10.0**((x-x0)*(1.0+alpha)))*np.exp(-10.0**((x-x0)))*beta
    return val

@custom_model
def mag_schechter(x, phi0=10**(-5), m0=20.0, alpha=-1.0):
    """
    Schechter function astropy.modeling.model object for ``x`` in magnitudes
    
    Parameters
    ----------
    x : array_like
        magnitude value(s) of galaxy/halo properties
    
    phi0 : float, optional
        parameter of the model, normalization
    
    x0 : float, optional
        parameter of the model, characteristic value of parameter 
        of galaxy/halo property
    
    alpha : float, optional
        parameter of the model, high abundnace end slope
    
    Returns
    -------
    phi : numpy.array
        abundance of galaxies/haloes at input ``x`` values
    
    Notes
    -----
    .. math::
        \\phi(x) = \\frac{2}{5}\\phi_0\\log10(10)\\left(10^{0.4(m_0-x)}\\right)^{(\\alpha+1)}\\exp{-10^{0.4(m_0-x)}}
    """
    
    x = np.asarray(x)
    x = x.astype(float)
    norm = (2.0/5.0)*phi0*np.log10(10.0)
    val = norm*(10.0**(0.4*(m0-x)))**(alpha+1.0)*np.exp(-10.0**(0.4*(m0-x)))
    return val


