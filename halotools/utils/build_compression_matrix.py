""" Numpy functions to compress some 
input data by sampling the data on a 2d grid 
and building a lookup table, the 'compression_matrix'. 
"""

import numpy as np

def retrieve_sample(prop1, prop2, compression_prop, *args):
    """
    """
    if len(args) == 2:
        prop1_bins_low, prop1_bins_high = args
        prop2_bins_low, prop2_bins_high = -np.inf, np.inf
    elif len(args) == 4:
        prop1_bins_low, prop1_bins_high, prop2_bins_low, prop2_bins_high = args
        
    mask = prop1 >= prop1_bins_low
    mask *= prop1 < prop1_bins_high
    mask *= prop2 >= prop2_bins_low
    mask *= prop2 < prop2_bins_high
    return compression_prop[mask]

def largest_nontrivial_row_index(m):
    return m.shape[0]-1-np.argmax(np.any(~np.isnan(m), axis=1)[::-1])

def smallest_nontrivial_row_index(m):
    return np.argmax(np.any(~np.isnan(m), axis=1))

def add_infinite_padding_to_compression_matrix(input_matrix):
    """
    """
    r1 = np.ones(input_matrix.shape[0], dtype=int)
    r1[0], r1[-1] = 2,2
    output_matrix = np.repeat(input_matrix, r1, axis=0)
    r2 = np.ones(output_matrix.shape[1], dtype=int)
    r2[0], r2[-1] = 2,2
    return np.repeat(output_matrix, r2, axis=1)

def add_infinite_padding_to_abscissa_array(arr):
    arr = np.insert(arr, 0, -np.inf)
    return np.append(arr, np.inf)

def nan_array_interpolation(arr, abscissa):
    """
    """
    mask = ~np.isnan(arr)
    return np.interp(abscissa, abscissa[mask], arr[mask])

def compression_matrix_from_array(arr, shape):
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
    for ibin1, prop1_bin_low, prop1_bin_high in zip(
        xrange(len(prop1_bins_midpoints)), prop1_bins[:-1], prop1_bins[1:]):

        binned_compression_prop = retrieve_sample(
            prop1, prop2, compression_prop, prop1_bin_low, prop1_bin_high)
        if len(binned_compression_prop) > npts_requirement:
            compression_array[ibin1] = summary_stat(binned_compression_prop)
        else:
            compression_array[ibin1] = np.nan

    compression_array = nan_array_interpolation(compression_array, prop1_bins_midpoints)

    output_shape = (len(prop1_bins_midpoints), len(prop2_bins_midpoints))
    compression_matrix = compression_matrix_from_array(compression_array, output_shape)

    compression_matrix = (
        add_infinite_padding_to_compression_matrix(compression_matrix))

    padded_prop1_bins = add_infinite_padding_to_abscissa_array(prop1_bins)
    padded_prop2_bins = add_infinite_padding_to_abscissa_array(prop2_bins)
    
    return compression_matrix, padded_prop1_bins, padded_prop2_bins

# def build_compression_matrix_double_prop(
#     prop1, prop2, compression_prop, prop1_bins, prop2_bins, 
#     npts_requirement = 100, summary_stat = np.mean):
#     """
#     """
#     prop1_bins_midpoints = (prop1_bins[:-1] + prop1_bins[1:])/2.
#     prop2_bins_midpoints = (prop2_bins[:-1] + prop2_bins[1:])/2.

#     compression_matrix = np.zeros((len(prop1_bins_midpoints), len(prop2_bins_midpoints)))

#     for ism, sm_low, sm_high in zip(xrange(len(sm_midpoints)), sm_bins[:-1], sm_bins[1:]):
#         for ilogm, logm_low, logm_high in zip(xrange(len(logm_midpoints)), logm_bins[:-1], logm_bins[1:]):
#             sat_sample = retrieve_sample(sats, sm_low, sm_high, logm_low, logm_high)
#             if len(sat_sample) > ngals_requirement:
#                 compression_matrix[ism, ilogm] = np.mean(sat_sample['iquench'])
#             else:
#                 compression_matrix[ism, ilogm] = np.nan












