"""
"""
import numpy as np

from ..kde_utils import kde_cdf_interpol


def test_edges1():
    data = np.random.uniform(0, 1, 10000)
    cdf = kde_cdf_interpol(data, data, npts_sample=50, npts_interpol=1000)
    assert np.all(cdf > 0)
    assert np.all(cdf < 1)
