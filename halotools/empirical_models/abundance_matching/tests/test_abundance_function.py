#!/usr/bin/env python
import numpy as np
from astropy.table import Table
from copy import copy, deepcopy

from astropy.tests.helper import pytest
from unittest import TestCase
import warnings 

from ....custom_exceptions import HalotoolsError

__all__ = ['TestAbundanceFunction']


class TestAbundanceFunction(TestCase):

	def setUp(self):
		pass

	def tearDown(self):
		pass
