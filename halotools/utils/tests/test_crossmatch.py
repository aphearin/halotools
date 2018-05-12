""" Module providing unit-testing of `~halotools.utils.crossmatch` function.
"""
import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from ..crossmatch import crossmatch, hostid_has_matching_host

__all__ = ('test_crossmatch1', )

fixed_seed = 43


@pytest.mark.installation_test
def test_crossmatch1():
    """ x has unique entries. All y values are in x. All x values are in y.
    """
    x = np.array([1, 3, 5])
    y = np.array([5, 1])
    x_idx, y_idx = crossmatch(x, y)

    assert np.all(x[x_idx] == y[y_idx])


def test_crossmatch2():
    """ x has repeated entries. All y values are in x. All x values are in y.
    """
    x = np.array([1, 3, 5, 3, 1, 1, 3, 5])
    y = np.array([5, 1])
    x_idx, y_idx = crossmatch(x, y)

    assert np.all(x[x_idx] == y[y_idx])


def test_crossmatch3():
    """ x has repeated entries. All y values are in x. Some x values are not in y.
    """
    x = np.array([0, 1, 3, 5, 3, -1, 1, 3, 5, -1])
    y = np.array([5, 1])
    x_idx, y_idx = crossmatch(x, y)

    assert np.all(x[x_idx] == y[y_idx])


def test_crossmatch4():
    """ x has repeated entries. Some y values are not in x. Some x values are not in y.
    """
    x = np.array([1, 3, 5, 3, 1, -1, 3, 5, -10, -10])
    y = np.array([5, 1, 100, 20])
    x_idx, y_idx = crossmatch(x, y)

    assert np.all(x[x_idx] == y[y_idx])


def test_crossmatch5():
    """ x has repeated entries. Some y values are not in x. Some x values are not in y.
    """
    xmax = 100
    numx = 10000
    with NumpyRNGContext(fixed_seed):
        x = np.random.randint(0, xmax+1, numx)

    y = np.arange(-xmax, xmax)[::10]
    with NumpyRNGContext(fixed_seed):
        np.random.shuffle(y)

    x_idx, y_idx = crossmatch(x, y)

    assert np.all(x[x_idx] == y[y_idx])


def test_crossmatch6():
    """ x and y have zero overlap.
    """
    x = np.array([-1, -5, -10])
    y = np.array([1, 2, 3, 4])
    x_idx, y_idx = crossmatch(x, y)
    assert len(x_idx) == 0
    assert len(y_idx) == 0
    assert np.all(x[x_idx] == y[y_idx])


def test_error_handling1():
    """ Verify that we raise the proper exception when y has repeated entries.
    """
    x = np.ones(5)
    y = np.ones(5)

    with pytest.raises(ValueError) as err:
        result = crossmatch(x, y)
    substr = "Input array y must be a 1d sequence of unique integers"
    assert substr in err.value.args[0]


def test_error_handling2():
    """ Verify that we raise the proper exception when y has non-integer values.
    """
    x = np.ones(5)
    y = np.arange(0, 5, 0.5)

    with pytest.raises(ValueError) as err:
        result = crossmatch(x, y)
    substr = "Input array y must be a 1d sequence of unique integers"
    assert substr in err.value.args[0]


def test_error_handling3():
    """ Verify that we raise the proper exception when y is multi-dimensional.
    """
    x = np.ones(5)
    y = np.arange(0, 6).reshape(2, 3)

    with pytest.raises(ValueError) as err:
        result = crossmatch(x, y)
    substr = "Input array y must be a 1d sequence of unique integers"
    assert substr in err.value.args[0]


def test_error_handling4():
    """ Verify that we raise the proper exception when x has non-integer values.
    """
    x = np.arange(0, 5, 0.5)
    y = np.arange(0, 6)

    with pytest.raises(ValueError) as err:
        result = crossmatch(x, y)
    substr = "Input array x must be a 1d sequence of integers"
    assert substr in err.value.args[0]


def test_error_handling5():
    """ Verify that we raise the proper exception when x is multi-dimensional.
    """
    x = np.arange(0, 6).reshape(2, 3)
    y = np.arange(0, 6)

    with pytest.raises(ValueError) as err:
        result = crossmatch(x, y)
    substr = "Input array x must be a 1d sequence of integers"
    assert substr in err.value.args[0]


def test_hostid_matching1():
    """
    """
    nsubs, nhosts = int(1e4), 100
    with NumpyRNGContext(fixed_seed):
        subhalo_hostid = np.random.randint(0, nhosts, nsubs)
    host_halo_id = np.arange(nhosts)
    has_match = hostid_has_matching_host(subhalo_hostid, host_halo_id)
    assert np.all(has_match == True)
    assert has_match.shape == (nsubs, )


def test_hostid_matching2():
    """
    """
    subhalo_hostid = [10, 20, -1, 1, -4, 1, 5, -4]
    correct_has_match = [0, 0, 0, 1, 0, 1, 1, 0]

    host_halo_id = np.arange(0, 8)
    with NumpyRNGContext(fixed_seed):
        np.random.shuffle(host_halo_id)

    has_match = hostid_has_matching_host(subhalo_hostid, host_halo_id)
    assert has_match.shape == (len(subhalo_hostid), )
    assert np.all(has_match == correct_has_match)


def test_hostid_matching3():
    """
    """
    satellite_hostid = [1, 3, 0]
    host_halo_id = [0, 1, 2, 3, 4]
    has_match = hostid_has_matching_host(satellite_hostid, host_halo_id)
    correct_has_match = [True, True, True]
    assert np.all(has_match == correct_has_match)


