""" Module providing unit-testing for the component models in
`halotools.empirical_models.occupation_components.leauthaud11_components` module"
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
from astropy.table import Table
from astropy.tests.helper import pytest

from ..tinker13_components import Tinker13Cens
from ....custom_exceptions import HalotoolsError

__all__ = ('test_quiescent_fraction_z1', )


def test_Tinker13Cens2():
    model = Tinker13Cens()
    t = Table()
    with pytest.raises(HalotoolsError) as err:
        model.mean_occupation(table=t)
    substr = "The ``table`` passed as a keyword argument to the ``mean_occupation`` method"
    assert substr in err.value.args[0]


def test_Tinker13Cens3():
    model = Tinker13Cens()
    t = Table()
    t['halo_mvir'] = np.zeros(5)
    with pytest.raises(HalotoolsError) as err:
        model.mean_occupation(table=t)
    substr = "used for SFR designation"
    assert substr in err.value.args[0]


def test_Tinker13Cens4():
    prim_haloprop = np.ones(5)
    model = Tinker13Cens()
    with pytest.raises(HalotoolsError) as err:
        model.mean_occupation(prim_haloprop=prim_haloprop)
    substr = "must pass both ``prim_haloprop`` and ``sfr_designation``"
    assert substr in err.value.args[0]


def test_Tinker13Cens5():
    prim_haloprop = np.zeros(2) + 1e12
    sfr_designation = np.array(('active', 'quiescent'))
    model = Tinker13Cens()
    result = model.mean_occupation(prim_haloprop=prim_haloprop, sfr_designation=sfr_designation)


def test_Tinker13Cens6():
    prim_haloprop = np.zeros(2) + 1e12
    sfr_designation = 'Cuba Gooding Jr.'
    model = Tinker13Cens()
    with pytest.raises(HalotoolsError) as err:
        __ = model.mean_occupation(prim_haloprop=prim_haloprop, sfr_designation=sfr_designation)
    substr = "The only acceptable values "
    assert substr in err.value.args[0]

    sfr_designation = 'active'
    result = model.mean_occupation(prim_haloprop=prim_haloprop, sfr_designation=sfr_designation)


def test_Tinker13Cens7():
    prim_haloprop = np.zeros(2) + 1e12
    logsm = np.linspace(10, 10.5, 5)
    model = Tinker13Cens()

    result = model.mean_stellar_mass_active(prim_haloprop=prim_haloprop)
    result = model.mean_stellar_mass_quiescent(prim_haloprop=prim_haloprop)
    result = model.mean_log_halo_mass_active(log_stellar_mass=logsm)
    result = model.mean_log_halo_mass_quiescent(log_stellar_mass=logsm)


def test_tinker13_cens_z3():
    model0 = Tinker13Cens(redshift=0)
    model0p5 = Tinker13Cens(redshift=0.5)
    model0p6 = Tinker13Cens(redshift=0.6)
    model1 = Tinker13Cens(redshift=1)

    assert model0.param_dict != model0p5.param_dict
    assert model0p6.param_dict == model0p5.param_dict
    assert model0p6.param_dict != model1.param_dict


def test_quiescent_fraction_z1():
    quiescent_fraction_control_masses = np.logspace(10.8, 14, 5)
    correct_red_fraction = np.array(list((10**(-1.28), 10**(-0.85), 0.54, 0.63, 0.77)))
    model_z1 = Tinker13Cens(redshift=0.35)
    model_red_fraction = model_z1.mean_quiescent_fraction(prim_haloprop=quiescent_fraction_control_masses)
    assert np.allclose(correct_red_fraction, model_red_fraction, rtol=0.01)

