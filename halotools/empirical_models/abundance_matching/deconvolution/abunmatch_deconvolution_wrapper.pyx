# cython: profile=False

"""
wrapper for ./src/abunmatch_deconvolution.c
"""

cimport cython
import numpy as np
cimport numpy as np

from abunmatch_deconvolution_declarations cimport convolved_fit

__all__ = ['abunmatch_deconvolution']

def abunmatch_deconvolution(np.ndarray[np.float64_t, ndim=1] af_key, 
                            np.ndarray[np.float64_t, ndim=1] af_val, 
                            np.ndarray[np.float64_t, ndim=1] smm, 
                            np.ndarray[np.float64_t, ndim=1] mf, 
                            np.float64_t scatter, np.int_t repeat=40,
                            np.float64_t sm_step=0.01):
    """
    python wrapper call ./src/abunmatch_deconvolution.c
    
    Parameters
    ----------
    af_key : 
    
    af_val : 
    
    smm : 
    
    mf :
    
    scatter : 
    
    repeat : 
    
    sm_step : 
    
    Returns
    -------
    smm : 
    
    """
    
    cdef int num_af_points = len(af_key)
    cdef int num_mass_bins = len(mf)
    
    if len(smm) != len(mf):
        raise ValueError('`smf` and `mf` must have the same size!')
    
    sm_step = np.fabs(float(sm_step))
    sm_min = min(af_key.min(), smm.min())
    
    if sm_min <= 0:
        offset = sm_step-sm_min
        af_key += offset
        smm += offset
    
    convolved_fit(&af_key[0], &af_val[0], num_af_points, &smm[0], &mf[0], num_mass_bins,
                  scatter, repeat, sm_step)
    
    if sm_min <= 0:
        smm -= offset
        af_key -= offset
    
    return smm

