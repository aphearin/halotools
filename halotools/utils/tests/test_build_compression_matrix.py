#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import numpy as np 
from astropy.tests.helper import pytest

from .. import build_compression_matrix as bcm

from ...custom_exceptions import HalotoolsError

__all__ = ('test_retrieve_sample', 
	'test_nan_array_interpolation1', 'test_nan_array_interpolation2')

def test_retrieve_sample():
	npts = 100
	prop1 = np.linspace(0, 1, npts)
	prop2 = np.linspace(1, 2, npts)
	compression_prop = np.linspace(-100, 100, npts)

	result = bcm.retrieve_sample(prop1, prop2, compression_prop, 0, 0.5)
	assert np.all(result <= 0)
	result = bcm.retrieve_sample(prop1, prop2, compression_prop, 0, 0.5, 1, 1.25)
	assert np.all(result <= 50)

def test_nan_array_interpolation1():
	arr = np.array([0, 1, 2, np.nan, 4])
	result = bcm.nan_array_interpolation(arr, np.arange(5))
	assert np.all(result == np.arange(len(arr)))

def test_nan_array_interpolation2():
	arr = np.array([0, 1, 2, np.nan, 4])
	with pytest.raises(ValueError) as err:
		result = bcm.nan_array_interpolation(arr, np.arange(4))
	substr = "Input ``arr`` and ``abscissa`` must have the same length"
	assert substr in err.value.message

def test_build_compression_matrix_double_prop1():
	npts = 1e4
	prop1 = np.linspace(0, 1, npts)
	prop2 = np.linspace(1, 2, npts)
	compression_prop = np.linspace(-100, 100, npts)

	nbins = 1e2
	prop1_bins = np.linspace(0, 1, nbins)
	prop2_bins = np.linspace(1, 2, nbins)

	compression_matrix = bcm.build_compression_matrix_double_prop(
		prop1, prop2, compression_prop, prop1_bins, prop2_bins, 
		npts_requirement = 99)

	with pytest.raises(ValueError) as err:
		compression_matrix = bcm.build_compression_matrix_double_prop(
			prop1, prop2, compression_prop, prop1_bins, prop2_bins, 
			npts_requirement = 100)
	substr = "entirely composed of NaN"
	assert substr in err.value.message

def test_largest_nontrivial_row_index1():
	m = np.array([
		[1, 1, 1], 
		[2, np.nan, 2], 
		[1, 2, 3], 
		[np.nan, np.nan, np.nan]
		])

	irow = bcm.largest_nontrivial_row_index(m)
	assert irow == 2

def test_largest_nontrivial_row_index2():
	m = np.array([
		[np.nan, np.nan, np.nan], 
		[1, 1, 1], 
		[np.nan, np.nan, np.nan], 
		[np.nan, np.nan, np.nan], 
		[2, np.nan, 2], 
		[1, 2, 3], 
		[np.nan, np.nan, np.nan], 
		[np.nan, np.nan, np.nan]
		])

	irow = bcm.largest_nontrivial_row_index(m)
	assert irow == 5

def test_largest_nontrivial_row_index3():
	m = np.array([
		[np.nan, np.nan, np.nan], 
		[np.nan, np.nan, np.nan], 
		[1, 1, 1], 
		[2, np.nan, 2], 
		[1, 2, 3]
		])

	irow = bcm.largest_nontrivial_row_index(m)
	assert irow == 4

def test_smallest_nontrivial_row_index1():
	m = np.array([
		[1, 1, 1], 
		[2, np.nan, 2], 
		[1, 2, 3], 
		[np.nan, np.nan, np.nan]
		])

	irow = bcm.smallest_nontrivial_row_index(m)
	assert irow == 0

def test_smallest_nontrivial_row_index2():
	m = np.array([
		[np.nan, np.nan, np.nan], 
		[np.nan, np.nan, np.nan], 
		[1, 1, 1], 
		[2, np.nan, 2], 
		[1, 2, 3], 
		[np.nan, np.nan, np.nan], 
		[np.nan, np.nan, np.nan]
		])

	irow = bcm.smallest_nontrivial_row_index(m)
	assert irow == 2

def test_smallest_nontrivial_row_index3():
	m = np.array([
		[np.nan, np.nan, np.nan], 
		[1, 1, 1], 
		[np.nan, np.nan, np.nan], 
		[np.nan, np.nan, np.nan], 
		[2, np.nan, 2], 
		[1, 2, 3], 
		[np.nan, np.nan, np.nan], 
		[np.nan, np.nan, np.nan]
		])

	irow = bcm.smallest_nontrivial_row_index(m)
	assert irow == 1













