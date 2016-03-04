#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import numpy as np 
from astropy.tests.helper import pytest
from ...custom_exceptions import HalotoolsError

from ..pairwise_velocity_stats import *
from .cf_helpers import generate_locus_of_3d_points

__all__ = ('test_mean_radial_velocity_vs_r1', )

# def mean_radial_velocity_vs_r(sample1, velocities1, rbins,
#                               sample2=None, velocities2=None,
#                               period=None, do_auto=True, do_cross=True,
#                               num_threads=1, max_sample_size=int(1e6),
#                               approx_cell1_size = None,
#                               approx_cell2_size = None):

@pytest.mark.slow
def test_mean_radial_velocity_vs_r1():
	npts = 200
	sample1 = np.random.rand(npts, 3)
	velocities1 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
	rbins = np.logspace(-1, 1, 10)
	result = mean_radial_velocity_vs_r(sample1, velocities1, rbins)

@pytest.mark.slow
def test_mean_radial_velocity_vs_r2():
	npts = 200
	sample1 = np.random.rand(npts, 3)
	velocities1 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
	sample2 = np.random.rand(npts, 3)
	velocities2 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))

	rbins = np.logspace(-1, 1, 10)
	result = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
		sample2 = sample2, velocities2 = velocities2)

@pytest.mark.slow
def test_mean_radial_velocity_vs_r_auto_consistency():
	npts = 200
	sample1 = np.random.rand(npts, 3)
	velocities1 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
	sample2 = np.random.rand(npts, 3)
	velocities2 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))

	rbins = np.logspace(-1, 1, 10)
	s1s1a, s1s2a, s2s2a = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
		sample2 = sample2, velocities2 = velocities2)
	s1s1b, s2s2b = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
		sample2 = sample2, velocities2 = velocities2, 
		do_cross = False)

	assert np.all(s1s1a == s1s1b)
	assert np.all(s2s2a == s2s2b)


@pytest.mark.slow
def test_mean_radial_velocity_vs_r_cross_consistency():
	npts = 200
	sample1 = np.random.rand(npts, 3)
	velocities1 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
	sample2 = np.random.rand(npts, 3)
	velocities2 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))

	rbins = np.logspace(-1, 1, 10)
	s1s1a, s1s2a, s2s2a = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
		sample2 = sample2, velocities2 = velocities2)
	s1s2b = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
		sample2 = sample2, velocities2 = velocities2, 
		do_auto = False)

	assert np.all(s1s2a == s1s2b)









