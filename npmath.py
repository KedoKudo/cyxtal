#!/usr/bin/env python
# coding=utf-8

# Functions commonly used but somehow still missing in standard Numpy
# packages. 


import numpy as np


def normalize(vec, axis=None):
    """
    return a normalized vector/matrix 
    axis=None : normalize entire vector/matrix 
    axis=0    : normalize by column
    axis=1    : normalize by row
    """
    if axis is None:
        return vec/np.linalg.norm(vec)
    else:
        return np.divide(vec, 
                         np.tile(np.linalg.norm(vec, axis=axis), 
                                 vec.shape[axis],
                                ).reshape(vec.shape, 
                                          order='F' if axis==1 else 'C',
                                         )
                        )


def norm(vec, axis=None):
    """
    return the norm of a vector/matrix using numpy.linalg.norm 
    axis=None : norm of entire vector/matrix 
    axis=0    : norm of each column
    axis=1    : norm of each row
    """
    return np.linalg.norm(vec, axis=axis)
