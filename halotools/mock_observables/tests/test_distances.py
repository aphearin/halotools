#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
from copy import deepcopy 

from collections import Counter

import numpy as np 

from astropy.tests.helper import pytest
from astropy.table import Table 

from ..distances import periodic_3d_distance

from ...sim_manager import FakeSim

from ...custom_exceptions import HalotoolsError

__all__ = ['TestCatalogAnalysisHelpers']

def test_distances1():
	x1 = np.random.rand(5)
	y1 = np.random.rand(5)
	z1 = np.random.rand(5)
	x2 = np.random.rand(5)
	y2 = np.random.rand(5)
	z2 = np.random.rand(5)
	Lbox = 1
	d = periodic_3d_distance(x1, y1, z1, x2, y2, z2, Lbox)
