#!/usr/bin/env python

import numpy as np 
from astropy.table import Table 
from astropy.io.ascii import read as astropy_ascii_read
from astropy.utils.data import get_pkg_data_filename, get_pkg_data_fileobj
from astropy.tests.helper import remote_data, pytest
from unittest import TestCase
from copy import copy


from ...smhm_models import *
from ... import model_defaults

from ....sim_manager import sim_defaults
from ....custom_exceptions import *
from ....utils import convert_to_ndarray

def test_behroozi10_redshift_safety():
	"""
	"""
	model = Behroozi10SmHm()

	result0 = model.mean_log_halo_mass(11)
	result1 = model.mean_log_halo_mass(11, redshift = 4)
	result2 = model.mean_log_halo_mass(11, redshift = sim_defaults.default_redshift)
	assert result0 == result2
	assert result0 != result1

	result0 = model.mean_stellar_mass(prim_haloprop = 1e12)
	result1 = model.mean_stellar_mass(prim_haloprop = 1e12, redshift = 4)
	result2 = model.mean_stellar_mass(prim_haloprop = 1e12, redshift = sim_defaults.default_redshift)
	assert result0 == result2
	assert result0 != result1

	model = Behroozi10SmHm(redshift = sim_defaults.default_redshift)
	result0 = model.mean_log_halo_mass(11)
	with pytest.raises(HalotoolsError) as exc:
		result1 = model.mean_log_halo_mass(11, redshift = 4)
	result2 = model.mean_log_halo_mass(11, redshift = model.redshift)
	assert result0 == result2

	result0 = model.mean_stellar_mass(prim_haloprop = 1e12)
	with pytest.raises(HalotoolsError) as exc:
		result1 = model.mean_stellar_mass(prim_haloprop = 1e12, redshift = 4)
	result2 = model.mean_stellar_mass(prim_haloprop = 1e12, redshift = model.redshift)
	assert result0 == result2

	# Check also that redshift differences propagate to the the Monte Carlo realizations, 
	### and also that the seeds propagate properly
	model0 = Behroozi10SmHm(redshift = 0)
	model0b = Behroozi10SmHm(redshift = 0)
	model1 = Behroozi10SmHm(redshift = 0)
	model2 = Behroozi10SmHm(redshift = 2)
	mass = np.zeros(1e4) + 1e12
	result0 = model0.mc_stellar_mass(prim_haloprop = mass, seed=43)
	result0b = model0.mc_stellar_mass(prim_haloprop = mass, seed=42)
	result1 = model1.mc_stellar_mass(prim_haloprop = mass, seed=43)
	result2 = model2.mc_stellar_mass(prim_haloprop = mass, seed=43)
	mean0 = convert_to_ndarray(result0.mean())
	mean0b = convert_to_ndarray(result0b.mean())
	mean1 = convert_to_ndarray(result1.mean())
	mean2 = convert_to_ndarray(result2.mean())

	assert np.isclose(mean0, mean1, rtol=0.0001)
	assert not np.isclose(mean0, mean0b, rtol=0.0001)
	assert not np.isclose(mean1, mean2, rtol=0.1)



