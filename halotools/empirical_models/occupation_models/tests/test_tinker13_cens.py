""" Module providing unit-testing for the component models in
`halotools.empirical_models.occupation_components.leauthaud11_components` module"
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
from astropy.tests.helper import pytest
from astropy.table import Table

from ..tinker13_components import Tinker13Cens

from ....sim_manager.sim_defaults import default_redshift

__all__ = ('test_Tinker13Cens_init1', )


def test_Tinker13Cens_init1():
    model = Tinker13Cens()

    try:
        assert float(model.redshift) == default_redshift
    except AttributeError:
        raise AttributeError("Model must have scalar ``redshift`` attribute")


def test_Tinker13Cens_init2():
    model1 = Tinker13Cens(redshift=0.25)
    model2 = Tinker13Cens(redshift=0.35)
    model3 = Tinker13Cens(redshift=0.5)
    model4 = Tinker13Cens(redshift=0.85)
    model5 = Tinker13Cens(redshift=2)

    assert model1.param_dict == model2.param_dict
    assert model1.param_dict != model3.param_dict
    assert model3.param_dict != model4.param_dict
    assert model4.param_dict == model5.param_dict


def test_Tinker13Cens_init3():
    model = Tinker13Cens(redshift=0.35)
    assert model.param_dict['smhm_m1_active'] == 12.56
    assert model.param_dict['smhm_beta_quiescent'] == 0.32


def test_mean_quiescent_fraction1():
    """ Ensure the quenched fraction evaluated at the control points
    always equals the param_dict values.
    """
    halo_mass = np.logspace(10.8, 14, 5)
    keys = ['quiescent_fraction_ordinates_param'+str(i) for i in range(1, 6)]

    for z in (0.35, 0.5, 1):
        model = Tinker13Cens(redshift=z)
        fq = model.mean_quiescent_fraction(prim_haloprop=halo_mass)
        assert np.all(fq == [model.param_dict[key] for key in keys])
        model.param_dict[keys[0]] *= 1.5
        model.param_dict[keys[1]] /= 1.5
        model.param_dict[keys[2]] /= 1.5
        fq = model.mean_quiescent_fraction(prim_haloprop=halo_mass)
        assert np.all(fq == [model.param_dict[key] for key in keys])


def test_mean_quiescent_fraction2():
    halo_mass = np.logspace(10.8, 14, 5)

    model = Tinker13Cens(redshift=0.35)
    assert model.mean_quiescent_fraction(prim_haloprop=halo_mass[0]) == 10**(-1.28)
    assert model.mean_quiescent_fraction(prim_haloprop=halo_mass[3]) == 0.63


def test_mean_quiescent_fraction3():
    halo_mass = np.logspace(10.8, 14, 5)

    model = Tinker13Cens(redshift=0.35)
    with pytest.raises(KeyError) as err:
        __ = model.mean_quiescent_fraction(x=halo_mass)
    substr = "Must pass either ``prim_haloprop`` or ``table`` keyword argument"
    assert substr in err.value.args[0]


def test_mean_quiescent_fraction4():
    halo_mass = np.logspace(10.8, 14, 5)
    t = Table(dict(x=halo_mass))
    model = Tinker13Cens(redshift=0.35)
    with pytest.raises(KeyError) as err:
        __ = model.mean_quiescent_fraction(table=t)
    substr = "does not have the requested ``{0}`` key".format(model.prim_haloprop_key)
    assert substr in err.value.args[0]

