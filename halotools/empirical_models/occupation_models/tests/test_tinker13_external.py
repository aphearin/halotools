"""
"""
import numpy as np
from astropy.utils.data import get_pkg_data_filename

from ..tinker13_components import Tinker13Cens


__all__ = ('test_mean_ncen_blue', )


def test_mean_ncen_blue():
    halo_mass_fname = get_pkg_data_filename('data/tinker13_test_data_mass_abscissa.npy')
    halo_mass = np.load(halo_mass_fname)
    ncen_blue_fname = get_pkg_data_filename('data/tinker13_test_data_mean_ncen_blue_logsm9p5.npy')
    ncen_blue_jt = np.load(ncen_blue_fname)

    logsm_cut_h0p7 = 9.5
    sm_cut_h0p7 = 10**logsm_cut_h0p7
    sm_cut_h1p0 = sm_cut_h0p7*0.7*0.7
    logsm_cut_h1p0 = np.log10(sm_cut_h1p0)

    model = Tinker13Cens(redshift=0.5, threshold=logsm_cut_h1p0)
    ncen_blue_halotools = model.mean_occupation_active(prim_haloprop=halo_mass)

    assert np.allclose(ncen_blue_halotools, ncen_blue_jt, rtol=0.1)


def test_mean_ncen_red():
    halo_mass_fname = get_pkg_data_filename('data/tinker13_test_data_mass_abscissa.npy')
    halo_mass = np.load(halo_mass_fname)
    ncen_red_fname = get_pkg_data_filename('data/tinker13_test_data_mean_ncen_red_logsm9p5.npy')
    ncen_red_jt = np.load(ncen_red_fname)

    logsm_cut_h0p7 = 9.5
    sm_cut_h0p7 = 10**logsm_cut_h0p7
    sm_cut_h1p0 = sm_cut_h0p7*0.7*0.7
    logsm_cut_h1p0 = np.log10(sm_cut_h1p0)

    model = Tinker13Cens(redshift=0.5, threshold=logsm_cut_h1p0)
    ncen_red_halotools = model.mean_occupation_quiescent(prim_haloprop=halo_mass)

    assert np.allclose(ncen_red_halotools, ncen_red_jt, rtol=0.1)


def test_red_fraction():
    halo_mass_fname = get_pkg_data_filename('data/tinker13_test_data_mass_abscissa.npy')
    halo_mass = np.load(halo_mass_fname)
    ncen_red_fname = get_pkg_data_filename('data/tinker13_test_data_mean_ncen_red_logsm9p5.npy')
    ncen_red_jt = np.load(ncen_red_fname)
    ncen_blue_fname = get_pkg_data_filename('data/tinker13_test_data_mean_ncen_blue_logsm9p5.npy')
    ncen_blue_jt = np.load(ncen_blue_fname)

    red_fraction_jt = ncen_red_jt/(ncen_red_jt + ncen_blue_jt)

    logsm_cut_h0p7 = 9.5
    sm_cut_h0p7 = 10**logsm_cut_h0p7
    sm_cut_h1p0 = sm_cut_h0p7*0.7*0.7
    logsm_cut_h1p0 = np.log10(sm_cut_h1p0)

    model = Tinker13Cens(redshift=0.5, threshold=logsm_cut_h1p0)
    red_fraction_halotools = model.mean_quiescent_fraction(prim_haloprop=halo_mass)

    assert np.allclose(red_fraction_halotools, red_fraction_jt, rtol=0.1)

