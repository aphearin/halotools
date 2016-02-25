#!/usr/bin/env python
import numpy as np
from astropy.table import Table
from copy import copy, deepcopy

from astropy.tests.helper import pytest
from unittest import TestCase
import warnings 

from ..premade_abundance_functions import Baldry2011
from ..abundance_function import AbundanceFunctionFromTabulated
from ....custom_exceptions import HalotoolsError

__all__ = ['TestAbundanceFunctionFromTabulated']


class TestAbundanceFunctionFromTabulated(TestCase):

	def setUp(self):
		self.baldry11 = Baldry2011()

	def test_consistency_with_tabulated1(self):

		x = np.logspace(9, 11, 100)
		n = self.baldry11.n(x)
		abundance_type = 'cumulative'
		model = AbundanceFunctionFromTabulated(x=x, n=n, 
			abundance_type=abundance_type)

		y = np.logspace(9, 11, 1000)
		assert np.allclose(model.n(y), self.baldry11.n(y), rtol=0.001)
		assert np.allclose(model.dn(y), self.baldry11.dn(y), rtol=0.001)

	def test_consistency_with_tabulated2(self):

		x = np.logspace(9, 11, 100)
		n = self.baldry11.dn(x)
		abundance_type = 'differential'
		model = AbundanceFunctionFromTabulated(x=x, n=n, 
			abundance_type=abundance_type)

		y = np.logspace(9, 10.5, 4)
		assert np.allclose(model.dn(y), self.baldry11.dn(y), rtol=0.001)

	@pytest.mark.xfail
	def test_consistency_with_tabulated3(self):

		x = np.logspace(9, 11, 100)
		n = self.baldry11.dn(x)
		abundance_type = 'differential'
		model = AbundanceFunctionFromTabulated(x=x, n=n, 
			abundance_type=abundance_type)

		y = np.logspace(9, 10.5, 4)
		assert np.allclose(model.n(y), self.baldry11.n(y), rtol=0.001)

	def tearDown(self):
		pass












