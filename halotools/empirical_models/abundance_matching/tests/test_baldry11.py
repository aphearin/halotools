#!/usr/bin/env python
import numpy as np
from astropy.table import Table
from copy import copy, deepcopy

from astropy.tests.helper import pytest
from unittest import TestCase
import warnings 

from ..premade_abundance_functions import Baldry2011

from ....custom_exceptions import HalotoolsError

__all__ = ['TestBaldry2011']


class TestBaldry2011(TestCase):

	def setUp(self):
		self.model = Baldry2011()

	def test_attributes(self):
		assert hasattr(self.model, 'publications')

	def test_deconvolution1(self):
		scatter = 0.2
		x_range = 1e9, 1e11
		x_pad = 9e8, 2e11
		model2 = self.model.compute_deconvolved_galaxy_abundance_function(
			scatter, x_range, x_pad)

		assert model2.n(1e10) != self.model.n(1e10)
		assert model2.dn(1e10) != self.model.dn(1e10)


	def tearDown(self):
		del self.model








