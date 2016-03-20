#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import numpy as np 
from astropy.tests.helper import pytest

from .. import build_compression_matrix as bcm

from ...custom_exceptions import HalotoolsError

__all__ = ('test_retrieve_sample', )

def test_retrieve_sample():
	npts = 100
	prop1 = np.linspace(0, 1, npts)
	prop2 = np.linspace(1, 2, npts)
	compression_prop = np.linspace(-100, 100, npts)

	result = bcm.retrieve_sample(prop1, prop2, compression_prop, 0, 0.5)
	assert np.all(result <= 0)
	result = bcm.retrieve_sample(prop1, prop2, compression_prop, 0, 0.5, 1, 1.25)
	assert np.all(result <= 50)

