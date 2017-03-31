"""
"""
import numpy as np
from astropy.utils.data import get_pkg_data_filename
from astropy.tests.helper import pytest

from ..tinker13_components import Tinker13Cens


__all__ = ('test_centrals_blue_bin1', )


@pytest.mark.xfail
def test_centrals_blue_bin1():
    fname = get_pkg_data_filename('data/test_rb2.HOD_blue_bin1')
    x = np.loadtxt(fname)
    halo_mass = x[:, 0]
    ncen_diff_tinker = x[:, 1]

    low_thresh_straight_up = 10**9
    high_thresh_straight_up = 10**9.5
    h = 0.7
    low_thresh_unity_h = low_thresh_straight_up/h/h
    high_thresh_unity_h = high_thresh_straight_up/h/h

    model_low_thresh = Tinker13Cens(redshift=0.5, threshold=np.log10(low_thresh_unity_h))
    assert model_low_thresh.param_dict['smhm_m0_0_active'] == 10.98
    model_high_thresh = Tinker13Cens(redshift=0.5, threshold=np.log10(high_thresh_unity_h))
    assert model_high_thresh.param_dict['smhm_m0_0_active'] == 10.98

    ncen_low_thresh = model_low_thresh.mean_occupation_active(prim_haloprop=halo_mass)
    ncen_high_thresh = model_high_thresh.mean_occupation_active(prim_haloprop=halo_mass)
    ncen_diff_aph = ncen_low_thresh - ncen_high_thresh

    assert np.allclose(ncen_diff_aph, ncen_diff_tinker, atol=0.01)


def test_shmr_blue():
    fname = get_pkg_data_filename('data/test_rb2.SHMR_blue')
    x = np.loadtxt(fname)
    mask = (x[:, 1] >= 1e11) & (x[:, 1] <= 1e15)
    halo_mass = x[:, 1][mask]
    stellar_mass_tinker = x[:, 0][mask]

    model = Tinker13Cens(redshift=0.5)
    assert model.param_dict['smhm_m0_0_active'] == 10.98

    stellar_mass_aph = model.stellar_mass_active(halo_mass)
    assert np.allclose(stellar_mass_aph, stellar_mass_tinker, rtol=0.2)



