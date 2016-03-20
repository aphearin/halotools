""" Numpy functions to compress some 
input data by sampling the data on a 2d grid 
and building a lookup table, the 'compression_matrix'. 
"""

import numpy as np
from .array_utils import array_is_monotonic 

def retrieve_sample(prop1, prop2, compression_prop, *args):
    """ Return the values of the input ``compression_prop`` array 
    whose indices fall within the bounds placed on the input ``prop1`` 
    and optionally ``prop2``.

    Parameters 
    ------------
    prop1 : array_like 
        First length-N array used to define cuts placed on input ``compression_prop``. 

    prop2 : array_like 
        Second length-N array used to define cuts placed on input ``compression_prop``. 

    compression_prop : array_like 
        Length-N array upon which cuts are placed. 

    prop1_low : float 
        Inclusive lower bound defining the cut pertaining to ``prop1``. 

    prop1_high : float 
        Exclusive upper bound defining the cut pertaining to ``prop1``. 

    prop2_low : float, optional 
        Inclusive lower bound defining the cut pertaining to ``prop2``. 
        Default is no cut. 

    prop2_high : float, optional 
        Exclusive upper bound defining the cut pertaining to ``prop2``. 
        Default is no cut. 

    Returns 
    --------
    result : array_like 
        Array storing all values of ``compression_prop`` for which 
        the indices of ``prop1`` meet the cuts defined by ``prop1_low`` and 
        ``prop1_high``, and optionally for which the indices also meet the 
        cut defined by ``prop2_low`` and ``prop2_high``. 

    Examples 
    ---------
    >>> prop1 = np.arange(-10, 0)
    >>> prop2 = np.arange(0, 10)
    >>> compression_prop = np.arange(10, 20)

    >>> result = retrieve_sample(prop1, prop2, compression_prop, -9, -6)
    >>> assert np.all(result == np.array([11, 12, 13]))

    >>> result = retrieve_sample(prop1, prop2, compression_prop, -9, -6, 2, 5)
    >>> assert np.all(result == np.array([12, 13]))

    """
    if len(args) == 2:
        prop1_low, prop1_high = args
        prop2_low, prop2_high = -np.inf, np.inf
    elif len(args) == 4:
        prop1_low, prop1_high, prop2_low, prop2_high = args
        
    mask = prop1 >= prop1_low
    mask *= prop1 < prop1_high
    mask *= prop2 >= prop2_low
    mask *= prop2 < prop2_high
    return compression_prop[mask]

def nan_array_interpolation(arr, abscissa):
    """ Interpolate over any possible NaN values in an input array. 

    Parameters 
    ------------
    arr : array_like 
        Input array of length N that may have NaN values 

    abscissa : array_like 
        Input array of length N used as the abscissa in the interpolation

    Returns 
    ---------
    result : array_like 
        Array of length N equal to the input ``arr``, but for which 
        `numpy.interp` is used to interpolate over any possible NaN values. 
    """
    arr = np.atleast_1d(arr)
    abscissa = np.atleast_1d(abscissa)
    try:
        assert len(arr) == len(abscissa)
    except AssertionError:
        msg = ("Input ``arr`` and ``abscissa`` must have the same length")
        raise ValueError(msg)

    mask = ~np.isnan(arr)
    return np.interp(abscissa, abscissa[mask], arr[mask])

def largest_nontrivial_row_index(m):
    """ Identify the index of the largest row of a matrix 
    that is not entirely composed of NaN.
    """
    return m.shape[0]-1-np.argmax(np.any(~np.isnan(m), axis=1)[::-1])

def smallest_nontrivial_row_index(m):
    """ Identify the index of the smallest row of a matrix 
    that is not entirely composed of NaN.
    """
    return np.argmax(np.any(~np.isnan(m), axis=1))

def _add_infinite_padding_to_compression_matrix(input_matrix):
    """
    """
    r1 = np.ones(input_matrix.shape[0], dtype=int)
    r1[0], r1[-1] = 2,2
    output_matrix = np.repeat(input_matrix, r1, axis=0)
    r2 = np.ones(output_matrix.shape[1], dtype=int)
    r2[0], r2[-1] = 2,2
    return np.repeat(output_matrix, r2, axis=1)

def _add_infinite_padding_to_abscissa_array(arr):
    arr = np.insert(arr, 0, -np.inf)
    return np.append(arr, np.inf)


def _compression_matrix_from_compression_array(arr, shape):
    """ Use `numpy.repeat` to transform a compression array 
    into a compression matrix. 
    """
    matrix = np.repeat(arr, shape[1])
    matrix.reshape(shape)
    return matrix

def build_compression_matrix_single_prop(
    prop1, prop2, compression_prop, prop1_bins, prop2_bins, 
    npts_requirement = 100, summary_stat = np.mean):
    """
    """
    prop1_bins_midpoints = (prop1_bins[:-1] + prop1_bins[1:])/2.
    prop2_bins_midpoints = (prop2_bins[:-1] + prop2_bins[1:])/2.

    compression_array = np.zeros_like(prop1_bins_midpoints)

    # Sample the input data in bins of prop1
    for ibin1, prop1_low, prop1_high in zip(
        xrange(len(prop1_bins_midpoints)), prop1_bins[:-1], prop1_bins[1:]):

        binned_compression_prop = retrieve_sample(
            prop1, prop2, compression_prop, prop1_low, prop1_high)
        if len(binned_compression_prop) > npts_requirement:
            compression_array[ibin1] = summary_stat(binned_compression_prop)
        else:
            compression_array[ibin1] = np.nan

    compression_array = nan_array_interpolation(compression_array, prop1_bins_midpoints)

    output_shape = (len(prop1_bins_midpoints), len(prop2_bins_midpoints))
    compression_matrix = _compression_matrix_from_compression_array(compression_array, output_shape)

    compression_matrix = (
        _add_infinite_padding_to_compression_matrix(compression_matrix))

    padded_prop1_bins = _add_infinite_padding_to_abscissa_array(prop1_bins)
    padded_prop2_bins = _add_infinite_padding_to_abscissa_array(prop2_bins)
    
    return compression_matrix, padded_prop1_bins, padded_prop2_bins

def fill_nan_matrix_rows(compression_matrix):
    """ If any of the first or final rows of the input ``compression_matrix`` 
    are composed entirely of NaN, fill in these rows with the smallest (largest) 
    nontrival rows. If any trivial rows remain, raise an exception.  

    Parameters 
    -----------
    compression_matrix : array_like 
        N x M matrix which may have rows entirely composed of NaN

    Returns 
    --------
    result : array_like 
        N x M matrix with outer rows filled in with the closest nontrivial inner rows 
    """
    # The final rows of the compression matrix may be entirely composed of NaN
    # If that is the case, replace each such row with the largest non-trivial row
    largest_nontrivial_row = largest_nontrivial_row_index(compression_matrix)
    if largest_nontrivial_row < compression_matrix.shape[0]-1:
        compression_matrix[largest_nontrivial_row+1:,:] = compression_matrix[largest_nontrivial_row,:]

    # The first rows of the compression matrix may be entirely composed of NaN
    # If that is the case, replace each such row with the smallest non-trivial row
    smallest_nontrivial_row = smallest_nontrivial_row_index(compression_matrix)
    compression_matrix[:smallest_nontrivial_row,:] = compression_matrix[smallest_nontrivial_row,:]

    return compression_matrix

def _check_for_remaining_nan_rows(compression_matrix, prop1_bins, npts_requirement):
    # No NaN rows should remain
    for irow in xrange(compression_matrix.shape[0]):
        try:
            assert not np.all(np.isnan(compression_matrix[irow,:]))
        except AssertionError:
            msg = ("Row %i of the compression_matrix is entirely composed of NaN. \n"
                "This row corresponds to the following cut on your data:\n"
                "%f <= prop1 < %f \n"
                "There are no 2d cells with %i data points passing this cut.\n"
                "You should either broaden your bins or relax the ``npts_requirement``\n")
            prop1_low, prop1_high = prop1_bins[irow], prop1_bins[irow+1]
            raise ValueError(msg % (irow, prop1_low, prop1_high, npts_requirement))


def build_compression_matrix_double_prop(
    prop1, prop2, compression_prop, prop1_bins, prop2_bins, 
    npts_requirement = 100, summary_stat = np.mean):
    """
    """
    prop1_bins_midpoints = (prop1_bins[:-1] + prop1_bins[1:])/2.
    prop2_bins_midpoints = (prop2_bins[:-1] + prop2_bins[1:])/2.

    compression_matrix = np.zeros((len(prop1_bins_midpoints), len(prop2_bins_midpoints)))

    # Sample the input data in bins of prop1 and prop2
    for ibin1, prop1_low, prop1_high in zip(
        xrange(len(prop1_bins_midpoints)), prop1_bins[:-1], prop1_bins[1:]):

        for ibin2, prop2_low, prop2_high in zip(
            xrange(len(prop2_bins_midpoints)), prop2_bins[:-1], prop2_bins[1:]):

            binned_compression_prop = retrieve_sample(prop1, prop2, compression_prop, 
                prop1_low, prop1_high, prop2_low, prop2_high)

            if len(binned_compression_prop) > npts_requirement:
                compression_matrix[ibin1, ibin2] = summary_stat(binned_compression_prop)
            else:
                compression_matrix[ibin1, ibin2] = np.nan

    compression_matrix = fill_nan_matrix_rows(compression_matrix)
    _check_for_remaining_nan_rows(compression_matrix, prop1_bins, npts_requirement)

    compression_matrix = (
        _add_infinite_padding_to_compression_matrix(compression_matrix))

    padded_prop1_bins = _add_infinite_padding_to_abscissa_array(prop1_bins)
    padded_prop2_bins = _add_infinite_padding_to_abscissa_array(prop2_bins)

    return compression_matrix, padded_prop1_bins, padded_prop2_bins









