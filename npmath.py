#!/usr/bin/env python
# coding=utf-8

# Functions commonly used but somehow still missing in standard Numpy
# packages.


import numpy as np
from numpy.linalg import norm


def normalize(vec, axis=None):
    """
    return a normalized vector/matrix
    axis=None : normalize entire vector/matrix
    axis=0    : normalize by column
    axis=1    : normalize by row
    """
    if axis is None:
        return vec/norm(vec)
    else:
        return np.divide(vec,
                         np.tile(norm(vec, axis=axis),
                                 vec.shape[axis],
                                ).reshape(vec.shape,
                                          order='F' if axis==1 else 'C',
                                         )
                        )

