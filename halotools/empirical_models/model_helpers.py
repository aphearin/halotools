# -*- coding: utf-8 -*-
"""

This module contains general purpose helper functions 
used by many of the hod model components.  

"""

__all__ = (
    ['GalPropModel', 'solve_for_polynomial_coefficients', 'polynomial_from_table', 
    'enforce_periodicity_of_box', 'update_param_dict']
    )

import numpy as np
from copy import copy

from scipy.interpolate import InterpolatedUnivariateSpline as spline

from . import model_defaults
from ..utils.array_utils import array_like_length as custom_len

from astropy.extern import six
from abc import ABCMeta

@six.add_metaclass(ABCMeta)
class GalPropModel(object):
    """ Abstact container class for any model of any galaxy property. 
    """

    def __init__(self, galprop_key):

        # Enforce the requirement that sub-classes have been configured properly
        required_method_name = 'mc_'+galprop_key
        if not hasattr(self, required_method_name):
            raise SyntaxError("Any sub-class of GalPropModel must "
                "implement a method named %s " % required_method_name)


def solve_for_polynomial_coefficients(abcissa, ordinates):
    """ Solves for coefficients of the unique, 
    minimum-degree polynomial that passes through 
    the input abcissa and attains values equal the input ordinates.  

    Parameters
    ----------
    abcissa : array 
        Elements are the abcissa at which the desired values of the polynomial 
        have been tabulated.

    ordinates : array 
        Elements are the desired values of the polynomial when evaluated at the abcissa.

    Returns
    -------
    polynomial_coefficients : array 
        Elements are the coefficients determining the polynomial. 
        Element i of polynomial_coefficients gives the degree i polynomial coefficient.

    Notes
    --------
    Input arrays abcissa and ordinates can in principle be of any dimension Ndim, 
    and there will be Ndim output coefficients.

    The input ordinates specify the desired values of the polynomial 
    when evaluated at the Ndim inputs specified by the input abcissa.
    There exists a unique, order Ndim polynomial that returns the input 
    ordinates when the polynomial is evaluated at the input abcissa.
    The coefficients of that unique polynomial are the output of the function. 

    As an example, suppose that a model in which the quenched fraction is 
    :math:`F_{q}(logM_{\\mathrm{halo}} = 12) = 0.25` and :math:`F_{q}(logM_{\\mathrm{halo}} = 15) = 0.9`. 
    Then this function takes [12, 15] as the input abcissa, 
    [0.25, 0.9] as the input ordinates, 
    and returns the array :math:`[c_{0}, c_{1}]`. 
    The unique polynomial linear in :math:`log_{10}M`  
    that passes through the input ordinates and abcissa is given by 
    :math:`F(logM) = c_{0} + c_{1}*log_{10}logM`.
    
    """

    columns = np.ones(len(abcissa))
    for i in np.arange(len(abcissa)-1):
        columns = np.append(columns,[abcissa**(i+1)])
    quenching_model_matrix = columns.reshape(
        len(abcissa),len(abcissa)).transpose()

    polynomial_coefficients = np.linalg.solve(
        quenching_model_matrix,ordinates)

    return np.array(polynomial_coefficients)

def polynomial_from_table(table_abcissa,table_ordinates,input_abcissa):
    """ Method to evaluate an input polynomial at the input_abcissa. 
    The input polynomial is determined by `solve_for_polynomial_coefficients` 
    from table_abcissa and table_ordinates. 

    Parameters
    ----------
    table_abcissa : array 
        Elements are the abcissa determining the input polynomial. 

    table_ordinates : array 
        Elements are the desired values of the input polynomial 
        when evaluated at table_abcissa

    input_abcissa : array 
        Points at which to evaluate the input polynomial. 

    Returns 
    -------
    output_ordinates : array 
        Values of the input polynomial when evaluated at input_abscissa. 

    """
    if not isinstance(input_abcissa, np.ndarray):
        input_abcissa = np.array(input_abcissa)
    coefficient_array = solve_for_polynomial_coefficients(
        table_abcissa,table_ordinates)
    output_ordinates = np.zeros(custom_len(input_abcissa))
    # Use coefficients to compute values of the inflection function polynomial
    for n,coeff in enumerate(coefficient_array):
        output_ordinates += coeff*input_abcissa**n

    return output_ordinates

def enforce_periodicity_of_box(coords, box_length):
    """ Function used to apply periodic boundary conditions 
    of the simulation, so that mock galaxies all lie in the range [0, Lbox].

    Parameters
    ----------
    coords : array_like
        float or ndarray containing a set of points with values ranging between 
        [-box_length, 2*box_length]
        
    box_length : float
        the size of simulation box (currently hard-coded to be Mpc/h units)

    Returns
    -------
    periodic_coords : array_like
        array with values and shape equal to input coords, 
        but with periodic boundary conditions enforced

    """    
    return coords % box_length


def piecewise_heaviside(bin_midpoints, bin_width, values_inside_bins, value_outside_bins, abcissa):
    """ Piecewise heaviside function. 

    The function returns values_inside_bins  
    when evaluated at points within bin_width/2 of bin_midpoints. 
    Otherwise, the output function returns value_outside_bins. 

    Parameters 
    ----------
    bin_midpoints : array_like 
        Length-Nbins array containing the midpoint of the abcissa bins. 
        Bin boundaries may touch, but overlapping bins will raise an exception. 

    bin_width : float  
        Width of the abcissa bins. 

    values_inside_bins : array_like 
        Length-Nbins array providing values of the desired function when evaluated 
        at a point inside one of the bins.

    value_outside_bins : float 
        value of the desired function when evaluated at any point outside the bins.

    abcissa : array_like 
        Points at which to evaluate binned_heaviside

    Returns 
    -------
    output : array_like  
        Values of the function when evaluated at the input abcissa

    """

    if custom_len(abcissa) > 1:
        abcissa = np.array(abcissa)
    if custom_len(values_inside_bins) > 1:
        values_inside_bins = np.array(values_inside_bins)
        bin_midpoints = np.array(bin_midpoints)

    # If there are multiple abcissa bins, make sure they do not overlap
    if custom_len(bin_midpoints)>1:
        midpoint_differences = np.diff(bin_midpoints)
        minimum_separation = midpoint_differences.min()
        if minimum_separation < bin_width:
            raise ValueError("Abcissa bins are not permitted to overlap")

    output = np.zeros(custom_len(abcissa)) + value_outside_bins

    if custom_len(bin_midpoints)==1:
        idx_abcissa_in_bin = np.where( 
            (abcissa >= bin_midpoints - bin_width/2.) & (abcissa < bin_midpoints + bin_width/2.) )[0]
        print(idx_abcissa_in_bin)
        output[idx_abcissa_in_bin] = values_inside_bins
    else:
        for ii, x in enumerate(bin_midpoints):
            idx_abcissa_in_binii = np.where(
                (abcissa >= bin_midpoints[ii] - bin_width/2.) & 
                (abcissa < bin_midpoints[ii] + bin_width/2.)
                )[0]
            output[idx_abcissa_in_binii] = values_inside_bins[ii]

    return output


def custom_spline(table_abcissa, table_ordinates, k=0):
    """ Simple workaround to replace scipy's silly convention 
    for treating the spline_degree=0 edge case. 

    Parameters 
    ----------
    table_abcissa : array_like
        abcissa values defining the interpolation 

    table_ordinates : array_like
        ordinate values defining the interpolation 

    k : int 
        Degree of the desired spline interpolation

    Returns 
    -------
    output : object  
        Function object to use to evaluate the interpolation of 
        the input table_abcissa & table_ordinates 

    Notes 
    -----
    Only differs from the scipy.interpolate.UnivariateSpline for 
    the case where the input tables have a single element. The default behavior 
    of the scipy function is to raise an exception, which is silly: clearly 
    the desired behavior in this case is to simply return the input value 
    table_ordinates[0] for all values of the input abcissa. 

    """
    if custom_len(table_abcissa) != custom_len(table_ordinates):
        len_abcissa = custom_len(table_abcissa)
        len_ordinates = custom_len(table_ordinates)
        raise TypeError("table_abcissa and table_ordinates must have the same length \n"
            " len(table_abcissa) = %i and len(table_ordinates) = %i" % (len_abcissa, len_ordinates))

    if k >= custom_len(table_abcissa):
        len_abcissa = custom_len(table_abcissa)
        raise ValueError("Input spline degree k = %i "
            "must be less than len(abcissa) = %i" % (k, len_abcissa))

    max_scipy_spline_degree = 5
    k = np.min([k, max_scipy_spline_degree])

    if k<0:
        raise ValueError("Spline degree must be non-negative")
    elif k==0:
        if custom_len(table_ordinates) != 1:
            raise TypeError("In spline_degree=0 edge case, "
                "table_abcissa and table_abcissa must be 1-element arrays")
        return lambda x : np.zeros(custom_len(x)) + table_ordinates[0]
    else:
        spline_function = spline(table_abcissa, table_ordinates, k=k)
        return spline_function

def call_func_table(func_table, abcissa, func_indices):
    """ Returns the output of an array of functions evaluated at a set of input points 
    if the indices of required functions is known. 

    Parameters 
    ----------
    func_table : array_like 
        Length k array of function objects

    abcissa : array_like 
        Length Npts array of points at which to evaluate the functions. 

    func_indices : array_like 
        Length Npts array providing the indices to use to choose which function 
        operates on each abcissa element. Thus func_indices is an array of integers 
        ranging between 0 and k-1. 

    Returns 
    -------
    out : array_like 
        Length Npts array giving the evaluation of the appropriate function on each 
        abcissa element. 

    """
    func_argsort = func_indices.argsort()
    func_ranges = list(np.searchsorted(func_indices[func_argsort], range(len(func_table))))
    func_ranges.append(None)
    out = np.zeros_like(abcissa)
    for f, start, end in zip(func_table, func_ranges, func_ranges[1:]):
        ix = func_argsort[start:end]
        out[ix] = f(abcissa[ix])
    return out

def bind_required_kwargs(required_kwargs, obj, **kwargs):
    """ Method binds each element of ``required_kwargs`` to 
    the input object ``obj``, or raises and exception for cases 
    where a mandatory keyword argument was not passed to the 
    ``obj`` constructor.

    Used throughout the package when a required keyword argument 
    has no obvious default value. 

    Parameters 
    ----------
    required_kwargs : list 
        List of strings of the keyword arguments that are required 
        when instantiating the input ``obj``. 

    obj : object 
        The object being instantiated. 

    Notes 
    -----
    The `bind_required_kwargs` method assumes that each 
    required keyword argument should be bound to ``obj`` 
    as attribute with the same name as the keyword. 
    """
    for key in required_kwargs:
        if key in kwargs.keys():
            setattr(obj, key, kwargs[key])
        else:
            class_name = obj.__class__.__name__
            msg = (
                key + ' is a required keyword argument ' + 
                'to instantiate the '+class_name+' class'
                )
            raise KeyError(msg)

def update_param_dict(obj, **kwargs):
    """ Method used to update the ``param_dict`` attribute of the 
    input ``obj`` according to ``input_param_dict``. 

    The only items in ``obj.param_dict`` that will be updated 
    are those with a matching key in ``input_param_dict``; 
    all other keys in ``input_param_dict`` will be ignored. 

    Parameters 
    ----------
    obj : object
        Class instance whose ``param_dict`` is being updated. 

    input_param_dict : dict, optional keyword argument 
        Parameter dictionary used to update ``obj.param_dict``.
        If no ``input_param_dict`` keyword argument is passed, 
        the `update_param_dict` method does nothing. 
    """
    if 'input_param_dict' not in kwargs.keys():
        return 
    else:
        input_param_dict = kwargs['input_param_dict']

    if not hasattr(obj, 'param_dict'):
        raise AttributeError("Input ``obj`` must have a ``param_dict`` attribute")

    for key in obj.param_dict.keys():
        if key in input_param_dict.keys():
            obj.param_dict[key] = input_param_dict[key]









