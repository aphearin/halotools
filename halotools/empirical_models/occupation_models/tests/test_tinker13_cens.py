""" Module providing unit-testing for the component models in
`halotools.empirical_models.occupation_components.leauthaud11_components` module"
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np

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








