#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import numpy as np 
from astropy.tests.helper import pytest
from ...custom_exceptions import HalotoolsError

from ..nearest_neighbor import nearest_neighbor

__all__ = ['test_nearest_neighbor_func_signature']

def test_nearest_neighbor_func_signature():
    npts = 100
    sample1 = np.random.rand(npts, 3)
    sample2 = np.random.rand(npts, 3)
    r_max = 0.2
    nth_nearest = 1
    nn = nearest_neighbor(sample1, sample2, r_max, 
        nth_nearest=nth_nearest)
    nn2 = nearest_neighbor(sample1, sample2, r_max, 
        nth_nearest=2, period=1.)






