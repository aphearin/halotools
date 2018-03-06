"""
"""
import numpy as np


def pure_python_bin_free_rank_matching(x_sample1, ranks_sample1, x_sample2, y_sample2, nwin):
    """
    Examples
    --------
    >>> n1, n2, nwin = 1001, 501, 51
    >>> x_sample1 = np.linspace(0, 1, n1)
    >>> ranks_sample1 = np.random.randint(0, nwin, n1)
    >>> x_sample2 = np.linspace(0, 1, n2)
    >>> ranks_sample2 = np.random.randint(0, nwin, n2)
    >>> y_sample2 = np.linspace(0, 1, n2)
    >>> result = pure_python_bin_free_rank_matching(x_sample1, ranks_sample1, x_sample2, y_sample2, nwin)
    """
    result = np.zeros_like(x_sample1)
    n1 = len(x_sample1)
    n2 = len(x_sample2)
    for i in range(n1):
        idx2 = np.searchsorted(x_sample2, x_sample1[i])
        low = max(0, idx2-nwin/2)
        high = min(n2, idx2+nwin/2)
        window = y_sample2[low:high]
        idx_sorted_window = np.argsort(window)
        result[i] = window[idx_sorted_window[ranks_sample1[i]]]
    return result
