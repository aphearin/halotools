#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import numpy as np 
from astropy.tests.helper import pytest
from ...custom_exceptions import HalotoolsError

from ..pairwise_velocity_stats import *
from .cf_helpers import generate_locus_of_3d_points

__all__ = ('test_mean_radial_velocity_vs_r1', 
	'test_mean_radial_velocity_vs_r2', 'test_mean_radial_velocity_vs_r3', 
	'test_mean_radial_velocity_vs_r_auto_consistency', 
	'test_mean_radial_velocity_vs_r_cross_consistency')

@pytest.mark.slow
def test_mean_radial_velocity_vs_r1():
	np.random.seed(43)

	npts = 200
	sample1 = np.random.rand(npts, 3)
	velocities1 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
	rbins = np.linspace(0, 0.3, 10)
	result = mean_radial_velocity_vs_r(sample1, velocities1, rbins)

@pytest.mark.slow
def test_mean_radial_velocity_vs_r2():
	np.random.seed(43)

	npts = 200
	sample1 = np.random.rand(npts, 3)
	velocities1 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
	rbins = np.linspace(0, 0.3, 10)
	result1 = mean_radial_velocity_vs_r(sample1, velocities1, rbins)
	result2 = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
		approx_cell1_size = [0.2, 0.2, 0.2])
	assert np.allclose(result1, result2, rtol=0.0001)

@pytest.mark.slow
def test_mean_radial_velocity_vs_r3():
	np.random.seed(43)

	npts = 200
	sample1 = np.random.rand(npts, 3)
	velocities1 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
	sample2 = np.random.rand(npts, 3)
	velocities2 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))

	rbins = np.linspace(0, 0.3, 10)
	result = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
		sample2 = sample2, velocities2 = velocities2)

@pytest.mark.slow
def test_mean_radial_velocity_vs_r_auto_consistency():
	np.random.seed(43)

	npts = 200
	sample1 = np.random.rand(npts, 3)
	velocities1 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
	sample2 = np.random.rand(npts, 3)
	velocities2 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))

	rbins = np.linspace(0, 0.3, 10)
	s1s1a, s1s2a, s2s2a = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
		sample2 = sample2, velocities2 = velocities2)
	s1s1b, s2s2b = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
		sample2 = sample2, velocities2 = velocities2, 
		do_cross = False)

	assert np.allclose(s1s1a,s1s1b, rtol=0.001)
	assert np.allclose(s2s2a,s2s2b, rtol=0.001)


@pytest.mark.slow
def test_mean_radial_velocity_vs_r_cross_consistency():
	np.random.seed(43)

	npts = 200
	sample1 = np.random.rand(npts, 3)
	velocities1 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
	sample2 = np.random.rand(npts, 3)
	velocities2 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))

	rbins = np.linspace(0, 0.3, 10)
	s1s1a, s1s2a, s2s2a = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
		sample2 = sample2, velocities2 = velocities2)
	s1s2b = mean_radial_velocity_vs_r(sample1, velocities1, rbins, 
		sample2 = sample2, velocities2 = velocities2, 
		do_auto = False)

	assert np.allclose(s1s2a,s1s2b, rtol=0.001)

@pytest.mark.slow
def test_radial_pvd_vs_r1():
	np.random.seed(43)

	npts = 200
	sample1 = np.random.rand(npts, 3)
	velocities1 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
	rbins = np.linspace(0, 0.3, 10)
	result = radial_pvd_vs_r(sample1, velocities1, rbins)

@pytest.mark.slow
def test_radial_pvd_vs_r2():
	np.random.seed(43)

	npts = 200
	sample1 = np.random.rand(npts, 3)
	velocities1 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
	rbins = np.linspace(0, 0.3, 10)
	result1 = radial_pvd_vs_r(sample1, velocities1, rbins)
	result2 = radial_pvd_vs_r(sample1, velocities1, rbins, 
		approx_cell1_size = [0.2, 0.2, 0.2])
	assert np.allclose(result1, result2, rtol=0.0001)

@pytest.mark.slow
def test_radial_pvd_vs_r3():
	np.random.seed(43)

	npts = 200
	sample1 = np.random.rand(npts, 3)
	velocities1 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
	sample2 = np.random.rand(npts, 3)
	velocities2 = np.random.normal(
		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))

	rbins = np.linspace(0, 0.3, 10)
	result = radial_pvd_vs_r(sample1, velocities1, rbins, 
		sample2 = sample2, velocities2 = velocities2)


# @pytest.mark.slow
# def test_radial_pvd_vs_r_auto_consistency():
# 	np.random.seed(43)

# 	npts = 200
# 	sample1 = np.random.rand(npts, 3)
# 	velocities1 = np.random.normal(
# 		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))
# 	sample2 = np.random.rand(npts, 3)
# 	velocities2 = np.random.normal(
# 		loc = 0, scale = 100, size=npts*3).reshape((npts, 3))

# 	rbins = np.linspace(0, 0.3, 10)
# 	s1s1a, s1s2a, s2s2a = radial_pvd_vs_r(sample1, velocities1, rbins, 
# 		sample2 = sample2, velocities2 = velocities2)
	# s1s1b, s2s2b = radial_pvd_vs_r(sample1, velocities1, rbins, 
	# 	sample2 = sample2, velocities2 = velocities2, 
	# 	do_cross = False)

	# assert np.allclose(s1s1a,s1s1b, rtol=0.001)
	# assert np.allclose(s2s2a,s2s2b, rtol=0.001)


