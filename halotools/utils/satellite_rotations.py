"""
"""
import numpy as np
from .crossmatch import crossmatch, hostid_has_matching_host
from .matrix_operations_3d import rotation_matrices_from_angles, rotate_vector_collection


__all__ = ('rotate_satellite_vectors', )


def rotate_satellite_vectors(satellite_vectors, satellite_hostid, satellite_rotation_angles,
            host_halo_id, host_halo_axis, return_has_match=False):
    """ Rotate an input set of `satellite_vectors` by the input `satellite_rotation_angles`
    about the axis associated with each satellite's host halo.

    Parameters
    ----------
    satellite_vectors : ndarray
        Numpy array of shape (num_sats, 3) storing a 3d vector associated with each satellite.

    satellite_hostid : ndarray
        Numpy integer array of shape (num_sats, ) storing the ID of the associated host halo.
        Entries of `satellite_hostid` without a matching entry in `host_halo_id` will not be rotated

    satellite_rotation_angles : ndarray
        Numpy array of shape (num_sats, ) storing the rotation angles in radians

    host_halo_id : ndarray
        Numpy integer array of shape (num_host_halos, ) storing
        the unique IDs of candidate host halos.

    host_halo_axis : ndarray
        Numpy array of shape (num_host_halos, 3) storing the 3d vector about which
        satellites of that host halo will be rotated.

    return_has_match : bool, optional
        Optionally return a boolean array storing whether the satellite has a matching host.
        Default is False

    Returns
    -------
    rotated_vectors : ndarray
        Numpy array of shape (num_sats, 3) storing the rotated satellite vectors

    has_match : ndarray, optional
        Numpy boolean array of shape (num_sats, ) equals True only for those satellites
        for which there is a matching host_halo_id

    Examples
    --------
    >>> num_sats, num_host_halos = int(1e4), 100
    >>> satellite_hostid = np.random.randint(0, num_host_halos, num_sats)
    >>> satellite_rotation_angles = np.random.uniform(-np.pi/2., np.pi/2., num_sats)
    >>> satellite_vectors = np.random.uniform(-1, 1, num_sats*3).reshape((num_sats, 3))
    >>> host_halo_id = np.arange(num_host_halos)
    >>> host_halo_axis = np.random.uniform(-1, 1, num_host_halos*3).reshape((num_host_halos, 3))
    >>> rotated_vectors = rotate_satellite_vectors(satellite_vectors, satellite_hostid, satellite_rotation_angles, host_halo_id, host_halo_axis)
    """
    satellite_hostid = np.atleast_1d(satellite_hostid)
    satellite_rotation_angles = np.atleast_1d(satellite_rotation_angles)
    satellite_vectors = np.atleast_2d(satellite_vectors)
    host_halo_id = np.atleast_1d(host_halo_id)
    host_halo_axis = np.atleast_2d(host_halo_axis)

    has_match = hostid_has_matching_host(satellite_hostid, host_halo_id)
    satellite_rotation_angles[~has_match] = 0.

    idxA, idxB = crossmatch(satellite_hostid, host_halo_id)
    matched_host_halo_axes = host_halo_axis[idxB]
    matched_satellite_vectors = satellite_vectors[idxA]
    matched_rotation_angles = satellite_rotation_angles[idxA]

    rotation_matrices = rotation_matrices_from_angles(matched_rotation_angles, matched_host_halo_axes)
    result = rotate_vector_collection(rotation_matrices, matched_satellite_vectors)
    new_vectors = np.zeros_like(result)
    new_vectors[idxA] = result

    if return_has_match:
        return new_vectors, has_match
    else:
        return new_vectors
