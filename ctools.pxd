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
    intp = np.int64
ELSE:
    ctypedef np.int32_t INTP_t
    intp = np.int32

ctypedef np.float64_t DTYPE_t
DTYPE = np.float64


cdef inline DTYPE_t DTYPE_max(DTYPE_t a, DTYPE_t b): return a if a >= b else b
cdef inline DTYPE_t DTYPE_min(DTYPE_t a, DTYPE_t b): return a if a <= b else b
cdef inline DTYPE_t DTYPE_abs(DTYPE_t a): return a if a>=0 else -a
cdef inline INTP_t  INTP_abs(INTP_t a): return a if a>=0 else -a


cpdef tuple kmeans(DTYPE_t[:,:] data,
                   INTP_t k,
                   INTP_t max_iter=*,
                   DTYPE_t threshold=*)