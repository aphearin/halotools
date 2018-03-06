"""
"""
import numpy as np
from .engines import cython_bin_free_cam_kernel


def bin_free_conditional_abunmatch(x, y, x2, y2, nwin):
    """
    Examples
    --------
    >>> x = np.linspace(0, 1, 100)
    >>> result = bin_free_conditional_abunmatch(x, x, x, x, 15)
    """
    return cython_bin_free_cam_kernel(y, nwin)
