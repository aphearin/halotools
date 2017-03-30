"""
"""
import numpy as np
from astropy.utils.data import get_pkg_data_filename

from ..tinker13_components import Tinker13Cens
from ..tinker13_parameter_dictionaries import param_dict_z2

__all__ = ('test_blue_bin1', )


def test_blue_bin1():
    fname = get_pkg_data_filename('data/test_rb.HOD_blue_bin1')
    x = np.loadtxt(fname)
    halo_mass = x[:, 0]
    ncen = x[:, 1]
    nsat = x[:, 2]

    model = Tinker13Cens()
    for key in model.param_dict.keys():
        model.param_dict[key] = param_dict_z2[key]
