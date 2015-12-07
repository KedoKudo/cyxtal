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


######################
# MODULE DECLARATION #
######################
cpdef symmetry(lattice)

cdef class Quaternion:
    cdef public double w,x,y,z

