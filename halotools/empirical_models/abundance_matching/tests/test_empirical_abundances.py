#!/usr/bin/env python
import numpy as np
from astropy.table import Table
from copy import copy, deepcopy

from astropy.tests.helper import pytest
from unittest import TestCase
import warnings 

from ..empirical_abundance import *

from ....custom_exceptions import HalotoolsError

__all__ = ['TestEmpiricalAbundances']


class TestEmpiricalAbundances(TestCase):

	def setUp(self):
		pass

	def test_empirical_cum_ndensity1(self):
		npts = 100
		Lbox = 250
		x = np.linspace(-1, 1, npts)
		volume = Lbox**3

		cumu_x, x_val = empirical_cum_ndensity(x, volume)
		assert np.all(np.diff(cumu_x) > 0)

	def test_empirical_cum_ndensity2(self):
		npts = 5
		Lbox = 250
		x = np.linspace(-1, 1, npts)
		volume = Lbox**3
		
		cumu_x, x_val = empirical_cum_ndensity(x, volume, 
			nd_increases_with_x = True)
		assert np.all(np.diff(cumu_x) > 0)

		cumu_x2, x_val2 = empirical_cum_ndensity(x, volume, 
			nd_increases_with_x = False)
		assert np.all(cumu_x == cumu_x2)
		assert np.all(x_val == x_val2[::-1])

	def test_empirical_cum_ndensity3(self):
		npts = 5
		Lbox = 250
		x = np.linspace(-1, 1, npts)
		volume = Lbox**3
		
		w = np.zeros(npts) + 0.5
		cumu_x, x_val = empirical_cum_ndensity(x, volume, weights = w)
		cumu_x2, x_val2 = empirical_cum_ndensity(x, volume)
		assert np.all(cumu_x == cumu_x2*w)

	def test_empirical_cum_ndensity4(self):
		npts = 50
		Lbox = 250
		x = np.linspace(-1, 1, npts)
		xbins = np.linspace(-0.99, 0.99, 10)
		volume = Lbox**3
		
		w = np.zeros(npts) + 0.5
		cumu_x, x_val = empirical_cum_ndensity(x, volume)
		cumu_x2, x_val2 = empirical_cum_ndensity(x, volume, xbins=xbins)
		assert len(cumu_x) == len(x)
		assert len(cumu_x2) == len(xbins)

	def tearDown(self):
		pass












