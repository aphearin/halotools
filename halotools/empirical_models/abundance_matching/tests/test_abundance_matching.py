#!/usr/bin/env python
import numpy as np
from astropy.table import Table
from copy import copy, deepcopy

from astropy.tests.helper import pytest
from unittest import TestCase
import warnings 

from ..premade_abundance_functions import Baldry2011, LiWhite2009
from ..abundance_matching import AbundanceMatching

from ....custom_exceptions import HalotoolsError

__all__ = ['TestAbundanceMatching']


class TestAbundanceMatching(TestCase):

	def setUp(self):
		self.baldry11 = Baldry2011()
		self.liwhite09 = LiWhite2009()

	def test_initialize1(self):

		am = AbundanceMatching('stellar_mass', 'halo_mpeak', 
			self.baldry11, np.logspace(9, 11, 100), np.logspace(9, 11, 100), 
			halo_abundance_function=self.liwhite09)

	def test_initialize2(self):
		galprop_name = 'stellar_mass'
		prim_haloprop_key = 'halo_mpeak'
		galaxy_abundance_function = self.baldry11
		galprop_sample_values = np.logspace(9, 11, 100)
		haloprop_sample_values = np.linspace(0, 1, 100)

		mass = np.random.power(2, size=1e4)
		halos = Table({'halo_mpeak': mass})
		am = AbundanceMatching(galprop_name, prim_haloprop_key, 
			galaxy_abundance_function, 
			galprop_sample_values, haloprop_sample_values, 
			complete_subhalo_catalog=halos, 
			Lbox = 250.)

	@pytest.mark.xfail
	def test_mean_relation(self):

		am = AbundanceMatching('stellar_mass', 'halo_mpeak', 
			self.baldry11, np.logspace(9, 11, 100), np.logspace(9, 11, 100), 
			halo_abundance_function=self.liwhite09)

		result = am.mean_stellar_mass(1e10)

	@pytest.mark.xfail
	def test_match(self):

		am = AbundanceMatching('stellar_mass', 'halo_mpeak', 
			self.baldry11, np.logspace(9, 11, 100), np.logspace(9, 11, 100), 
			halo_abundance_function=self.liwhite09)

		x1 = np.logspace(9.5, 10.5, 100)
		x2 = np.logspace(9.5, 10.5, 100)

		am.match(self.liwhite09, self.baldry11, x1, x2)

	def test_match2(self):

		am = AbundanceMatching('stellar_mass', 'halo_mpeak', 
			self.baldry11, np.logspace(9, 11, 100), np.logspace(9, 11, 100), 
			halo_abundance_function=self.liwhite09)

		x1 = np.logspace(9, 11, 100)
		x2 = np.logspace(9.5, 10.5, 100)

		am.match(self.liwhite09, self.baldry11, x1, x2)

	def tearDown(self):
		pass








