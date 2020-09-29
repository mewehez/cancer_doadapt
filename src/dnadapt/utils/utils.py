import os
import numpy as np


def folder_if_not_exist(path):
    """create folders in path if not exist"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def pcts_to_sizes(size: int, pcts: np.ndarray) -> np.ndarray:
    """
    Splits a size into corresponding percentages.
    :param size: The total size N.
    :param pcts: The list of percentages [p1, p2, p3, ...]
    :return: N * [p1, p2, p3, ...].
    """
    sizes = (size * np.array(pcts)).astype(int)
    amin = np.argmin(sizes)
    sizes[amin] += size - sizes.sum()  # make sure total is size
    return sizes
