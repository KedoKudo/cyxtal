import cython
import math, random, os
import numpy as np
cimport numpy as np


#############################
# SETUP FOR TYPE DEFINITION #
#############################
np.import_array()

cdef extern from "numpy/npy_math.h":
    double NPY_INFINITY

# Determine the right dtype for arrays of indices at compile time.
IF UNAME_MACHINE[-2:] == '64':
    ctypedef np.int64_t INTP_t
ELSE:
    ctypedef np.int32_t INTP_t

ctypedef np.float64_t DTYPE_t


cdef inline DTYPE_t DTYPE_sgn(DTYPE_t a): return 1.0 if a >= 0.0 else -1.0
cdef inline DTYPE_t DTYPE_abs(DTYPE_t a): return a if a>=0 else -a


######################
# MODULE DECLARATION #
######################
cpdef symmetry(str lattice)


cdef class Quaternion:
    cdef public double w,x,y,z


cdef class Rodrigues:
    cdef public np.ndarray v
    cdef        Quaternion __q


cdef class Eulers:
    cdef public double     phi1, PHI, phi2
    cdef        Quaternion __q


cdef class OrientationMatrix:
    cdef public np.ndarray g
    cdef        Quaternion   __q