"""
"""
import numpy as np
cimport cython

from ....utils import unsorting_indices

__all__ = ('cython_bin_free_cam_kernel', )


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef int _bisect_left_kernel(double[:] arr, double value):
    """ Return the index where to insert ``value`` in list ``arr`` of length ``n``,
    assuming ``arr`` is sorted.

    This function is equivalent to the bisect_left function implemented in the
    python standard libary bisect.
    """
    cdef int n = arr.shape[0]
    cdef int ifirst_subarr = 0
    cdef int ilast_subarr = n
    cdef int imid_subarr

    while ilast_subarr-ifirst_subarr >= 2:
        imid_subarr = (ifirst_subarr + ilast_subarr)/2
        if value > arr[imid_subarr]:
            ifirst_subarr = imid_subarr
        else:
            ilast_subarr = imid_subarr
    if value > arr[ifirst_subarr]:
        return ilast_subarr
    else:
        return ifirst_subarr


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef void _insert_first_pop_last_kernel(int* arr, int value_in1, int n):
    """ Insert the element ``value_in1`` into the input array and pop out the last element
    """
    cdef int i
    for i in range(n-2, -1, -1):
        arr[i+1] = arr[i]
    arr[0] = value_in1


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef int _correspondence_indices_shift(int idx_in1, int idx_out1, int idx):
    """ Update the correspondence indices array
    """
    cdef int shift = 0
    if idx_in1 < idx_out1:
        if idx_in1 <= idx < idx_out1:
            shift = 1
    elif idx_in1 > idx_out1:
        if idx_out1 < idx <= idx_in1:
            shift = -1
    return shift


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef void _insert_pop_kernel(double* arr, int idx_in1, int idx_out1, double value_in1):
    """ Pop out the value stored in index ``idx_out1`` of array ``arr``,
    and insert ``value_in1`` at index ``idx_in1`` of the final array.
    """
    cdef int i

    if idx_in1 <= idx_out1:
        for i in range(idx_out1-1, idx_in1-1, -1):
            arr[i+1] = arr[i]
    else:
        for i in range(idx_out1, idx_in1):
            arr[i] = arr[i+1]
    arr[idx_in1] = value_in1


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def cython_bin_free_cam_kernel(double[:] y1, double[:] y2, int[:] i2_match, int nwin,
            int add_subgrid_noise=0):
    """
    """
    cdef int nhalfwin = int(nwin/2)
    cdef int npts1 = y1.shape[0]
    cdef int npts2 = y2.shape[0]

    cdef int iy1, i, j, idx, idx2, iy2_match
    cdef int idx_in1, idx_out1, idx_in2, idx_out2
    cdef double value_in1, value_out1, value_in2, value_out2

    cdef double[:] y1_new = np.zeros(npts1, dtype='f8') - 1
    cdef int rank1, rank2

    #  Set up window arrays for y1
    cdf_values1 = np.copy(y1[:nwin])
    # print("initial cdf_values1 = {0}".format(np.array(cdf_values1)))
    idx_sorted_cdf_values1 = np.argsort(cdf_values1)
    cdef double[:] sorted_cdf_values1 = np.ascontiguousarray(
        cdf_values1[idx_sorted_cdf_values1], dtype='f8')
    cdef int[:] correspondence_indx1 = np.ascontiguousarray(
        unsorting_indices(idx_sorted_cdf_values1)[::-1], dtype='i4')

    #  Set up window arrays for y2
    cdef int iy2 = i2_match[nhalfwin]
    # print("initial iy2 = {0}".format(np.array(iy2)))

    cdef int init_iy2_low = iy2 - nhalfwin
    cdef int init_iy2_high = iy2+nhalfwin+1

    if init_iy2_low < 0:
        init_iy2_low = 0
        init_iy2_high = init_iy2_low + nwin
    elif init_iy2_high > npts2 - nhalfwin:
        init_iy2_high = npts2 - nhalfwin
        init_iy2_low = init_iy2_high - nwin

    msg = ("Bookkeeping error internal to cython_bin_free_cam_kernel\n"
        "init_iy2_low = {0}, init_iy2_high = {1}, nwin = {2}")
    assert init_iy2_high - init_iy2_low == nwin, msg.format(
            init_iy2_low, init_iy2_high, nwin)

    # print("initial low = {0}".format(init_iy2_low))
    # print("initial high = {0}".format(init_iy2_high))

    cdf_values2 = np.copy(y2[init_iy2_low:init_iy2_high])
    # print("initial cdf_values2 = {0}".format(np.array(cdf_values2)))

    idx_sorted_cdf_values2 = np.argsort(cdf_values2)
    # print("initial idx_sorted_cdf_values2 = {0}".format(
    #     np.array(idx_sorted_cdf_values2)))

    cdef double[:] sorted_cdf_values2 = np.ascontiguousarray(
        cdf_values2[idx_sorted_cdf_values2], dtype='f8')
    cdef int[:] correspondence_indx2 = np.ascontiguousarray(
        unsorting_indices(idx_sorted_cdf_values2)[::-1], dtype='i4')

    # print("initial sorted_cdf_values2 = {0}".format(np.array(sorted_cdf_values2)))

    for iy1 in range(nhalfwin, npts1-nhalfwin):
        print("\niy1 = {0}".format(iy1))

        rank1 = correspondence_indx1[nhalfwin]
        print("rank1 = {0}".format(rank1))
        iy2_match = i2_match[iy1]
        print("iy2_match = {0}".format(iy2_match))

        while iy2 < iy2_match:
            print("Traversing while loop for iy2 = {0}".format(iy2))
            value_in2 = y2[iy2 + nhalfwin + 1]
            idx_out2 = correspondence_indx2[nwin-1]
            value_out2 = sorted_cdf_values2[idx_out2]

            idx_in2 = _bisect_left_kernel(sorted_cdf_values2, value_in2)
            if value_in2 > value_out2:
                idx_in2 -= 1

            _insert_first_pop_last_kernel(&correspondence_indx2[0], idx_in2, nwin)
            for j in range(1, nwin):
                idx2 = correspondence_indx2[j]
                correspondence_indx2[j] += _correspondence_indices_shift(
                    idx_in2, idx_out2, idx2)

            _insert_pop_kernel(&sorted_cdf_values2[0], idx_in2, idx_out2, value_in2)

            iy2 += 1


        print("sorted_cdf_values2 = {0}".format(np.array(sorted_cdf_values2)))
        if add_subgrid_noise == 0:
            y1_new[iy1] = sorted_cdf_values2[rank1]
            print("y1_new[iy1] = {0}".format(y1_new[iy1]))
        else:
            raise NotImplementedError

        #  Move on to the next value in y1
        value_in1 = y1[iy1 + nhalfwin + 1]

        idx_out1 = correspondence_indx1[nwin-1]
        value_out1 = sorted_cdf_values1[idx_out1]

        idx_in1 = _bisect_left_kernel(sorted_cdf_values1, value_in1)
        if value_in1 > value_out1:
            idx_in1 -= 1

        _insert_first_pop_last_kernel(&correspondence_indx1[0], idx_in1, nwin)
        for i in range(1, nwin):
            idx = correspondence_indx1[i]
            correspondence_indx1[i] += _correspondence_indices_shift(
                idx_in1, idx_out1, idx)

        _insert_pop_kernel(&sorted_cdf_values1[0], idx_in1, idx_out1, value_in1)

    return y1_new
