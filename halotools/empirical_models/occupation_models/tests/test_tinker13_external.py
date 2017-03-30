"""
"""
import numpy as np
from astropy.utils.data import get_pkg_data_filename

__all__ = ('test_blue_bin1', )


def test_blue_bin1():
    fname = get_pkg_data_filename('data/test_rb.HOD_blue_bin1')
    x = np.loadtxt(fname)
    halo_mass = x[:, 0]
    ncen = x[:, 1]
    nsat = x[:, 2]
