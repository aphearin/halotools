"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from ..bin_free_cam import bin_free_conditional_abunmatch
from ....utils.conditional_percentile import cython_sliding_rank, rank_order_function
from .naive_python_cam import pure_python_rank_matching


fixed_seed = 5


def test():
    pass
