import cython
import math, random, os
import numpy as np
cimport numpy as np


######################
# MODULE DECLARATION #
######################
cpdef symmetry(lattice)

cdef class Quaternion:

    cdef public double w,x,y,z

    @staticmethod
    cdef eulers2Quaternion(double[:] e)

    @staticmethod
    cdef rodrigues2Quaternion(double[:] r)

    @staticmethod
    cdef oMatrix2Quaternion(double[:,:] m)

