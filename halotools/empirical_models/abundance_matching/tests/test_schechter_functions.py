#!/usr/bin/env python
import numpy as np
from astropy.table import Table
from copy import copy, deepcopy

from astropy.tests.helper import pytest
from unittest import TestCase
import warnings 

from ..schechter_functions import *

from ....custom_exceptions import HalotoolsError

__all__ = ['TestSchechterFunctions']


class TestSchechterFunctions(TestCase):

	def setUp(self):
		pass

	def test_schechter(self):
		model = schechter(1e10)
		_ = model(1e10)

	def test_super_schechter(self):
		model = super_schechter(1e10)
		_ = model(1e10)

	def test_log10_schechter(self):
		model = log10_schechter(1e10)
		_ = model(1e10)

	def test_log10_super_schechter(self):
		model = log10_super_schechter(1e10)
		_ = model(1e10)

	def test_mag_schechter(self):
		model = mag_schechter(1e10)
		_ = model(1e10)


	def tearDown(self):
		pass
