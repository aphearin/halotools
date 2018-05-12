"""
"""
import numpy as np
from ..matrix_operations_3d import elementwise_norm
from ..satellite_rotations import rotate_satellite_vectors


def test1():
    """ Randomly generate vectors and rotations and ensure
    the rotate_satellite_vectors function preserves norm.
    """
    nsats, nhosts = int(1e4), 100
    satellite_hostid = np.random.randint(0, nhosts, nsats)
    satellite_vectors = np.random.uniform(-1, 1, nsats*3).reshape((nsats, 3))
    satellite_rotation_angles = np.random.uniform(-np.pi, np.pi, nsats)
    host_halo_id = np.arange(nhosts)
    host_halo_axis = np.random.uniform(-1, 1, nhosts*3).reshape((nhosts, 3))
    new_vectors = rotate_satellite_vectors(
            satellite_vectors, satellite_hostid, satellite_rotation_angles,
            host_halo_id, host_halo_axis)
    orig_norms = elementwise_norm(satellite_vectors)
    new_norms = elementwise_norm(new_vectors)
    assert np.allclose(orig_norms, new_norms)


def test2():
    """
    All satellite vectors are normalized and point in x-direction.
    All host vectors are normalized and point in z-direction.
    Rotate the matched satellites by (pi, pi/2, -pi/2) and enforce that we get (-x, y, -y)
    """
    satellite_hostid = [1, 3, 0]
    satellite_vectors = [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
    satellite_rotation_angles = [np.pi, np.pi/2., -np.pi/2.]
    host_halo_id = [0, 1, 2, 3, 4]
    host_halo_axis = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]

    new_vectors = rotate_satellite_vectors(
            satellite_vectors, satellite_hostid, satellite_rotation_angles,
            host_halo_id, host_halo_axis)

    assert new_vectors.shape == (3, 3)
    assert np.allclose(new_vectors[0, :], [-1, 0, 0])
    assert np.allclose(new_vectors[1, :], [0, 1, 0])
    assert np.allclose(new_vectors[2, :], [0, -1, 0])


def test3():
    """
    All satellite vectors are normalized and point in y-direction.
    All host vectors are normalized and point in x-direction.
    Rotate the matched satellites by (pi, pi/2, -pi/2) and enforce that we get (-y, z, -z)
    """
    satellite_hostid = [1, 3, 0]
    satellite_vectors = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
    satellite_rotation_angles = [np.pi, np.pi/2., -np.pi/2.]
    host_halo_id = [0, 1, 2, 3, 4]
    host_halo_axis = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]

    new_vectors = rotate_satellite_vectors(
            satellite_vectors, satellite_hostid, satellite_rotation_angles,
            host_halo_id, host_halo_axis)

    assert new_vectors.shape == (3, 3)
    assert np.allclose(new_vectors[0, :], [0, -1, 0])
    assert np.allclose(new_vectors[1, :], [0, 0, 1])
    assert np.allclose(new_vectors[2, :], [0, 0, -1])


def test4():
    """ Concatenate four explicit cases and use nontrivial sequencing of
    satellite_hostid to ensure all four hard-coded examples are explicitly correct
    """
    satellite_hostid = [1, 3, 0, 2]
    satellite_vectors = [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]]
    satellite_rotation_angles = [np.pi, np.pi/2., -np.pi/2., np.pi/2.]
    host_halo_id = [0, 1, 2, 3, 4]
    host_halo_axis = [[1, 0, 0], [0, 0, 1], [-1, 0, 0], [1, 0, 0], [1, 0, 0]]

    new_vectors = rotate_satellite_vectors(
            satellite_vectors, satellite_hostid, satellite_rotation_angles,
            host_halo_id, host_halo_axis)

    assert new_vectors.shape == (4, 3)
    msg = ("This test4 intends to ensure that the passing test2 and test3\n"
        "have results that propagate through to nontrivial usage of satellite_hostid")
    assert np.allclose(new_vectors[0, :], [-1, 0, 0]), msg
    assert np.allclose(new_vectors[1, :], [0, 0, 1]), msg
    assert np.allclose(new_vectors[2, :], [1, 0, 0]), msg
    assert np.allclose(new_vectors[3, :], [0, 0, -1]), msg


