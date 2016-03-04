#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import numpy as np 
from astropy.tests.helper import pytest
from ...custom_exceptions import HalotoolsError

__all__ = ('generate_locus_of_3d_points', )

def generate_locus_of_3d_points(npts, loc=0.1, epsilon=0.001):
	"""
	Function returns a tight locus of points inside a 3d box. 

	Parameters 
	-----------
	npts : int 
		Number of desired points 

	loc : float 
		Midpoint value in all three dimensions 

	epsilon : float 
		Length of the box enclosing the returned locus of points

	Returns 
	---------
	pts : array_like 
		ndarray with shape (npts, 3) of points tightly localized around 
		the point (loc, loc, loc)
	"""
	return np.random.uniform(
		loc - epsilon/2., loc + epsilon/2., npts*3).reshape((npts, 3))


