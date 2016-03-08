#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import numpy as np 
from astropy.tests.helper import pytest
from ...custom_exceptions import HalotoolsError

from ..isolation_criteria import *
from .cf_helpers import generate_locus_of_3d_points

__all__ = ['test_spherical_isolation_criteria1']

sample1 = generate_locus_of_3d_points(100)

def test_spherical_isolation_criteria1():
	sample2 = generate_locus_of_3d_points(100, xc=0.5)
	r_max = 0.3
	iso = spherical_isolation(sample1, sample2, r_max)
	assert np.all(iso == True)

def test_spherical_isolation_criteria2():
	sample2 = generate_locus_of_3d_points(100, xc=0.11)
	r_max = 0.3
	iso = spherical_isolation(sample1, sample2, r_max)
	assert np.all(iso == False)

def test_spherical_isolation_criteria3():
	sample2 = generate_locus_of_3d_points(100, xc=0.95)
	r_max = 0.3
	iso = spherical_isolation(sample1, sample2, r_max, period=1)
	assert np.all(iso == False)
	iso2 = spherical_isolation(sample1, sample2, r_max)
	assert np.all(iso2 == True)

def test_cylindrical_isolation1():
	""" For two tight localizations of points right on top of each other, 
	all points in sample1 should not be isolated. 
	"""
	sample2 = generate_locus_of_3d_points(100)
	pi_max = 0.1
	rp_max = 0.1
	iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max)
	assert np.all(iso == False)

def test_cylindrical_isolation2():
	""" For two tight localizations of distant points, 
	all points in sample1 should be isolated. 
	"""
	sample1 = generate_locus_of_3d_points(3, xc=0.1, yc=0.1, zc=0.1)
	sample2 = generate_locus_of_3d_points(3, xc=0.5, yc=0.5, zc=0.5)
	pi_max = 0.1
	rp_max = 0.1
	spherical_iso = spherical_isolation(sample1, sample2, rp_max)
	assert np.all(spherical_iso == True)
	cylindrical_iso = cylindrical_isolation(sample1, sample2, rp_max, pi_max)
	assert np.all(cylindrical_iso == True)







