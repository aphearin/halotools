""" 
Cython declarations for abunmatch_deconvolution. 
"""

import numpy as np
cimport numpy as np

cdef extern from "include/abunmatch_deconvolution.h":
    void convolved_fit(np.float64_t* af_key, np.float64_t* af_val, np.int_t num_af_points, 
                       np.float64_t* smm, np.float64_t* mf, np.int_t MASS_BINS, np.float64_t scatter, 
                       np.int_t repeat, np.float64_t sm_step);    
