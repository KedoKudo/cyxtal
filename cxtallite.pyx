#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-
# filename: corientation.pyx

"""
   ________  ___  ___________    __
  / ____/\ \/ / |/ /_  __/   |  / /
 / /      \  /|   / / / / /| | / /
/ /___    / //   | / / / ___ |/ /___
\____/   /_//_/|_|/_/ /_/  |_/_____/

Copyright (c) 2015, C. Zhang.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1) Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2) Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

DESCRIPTION
-----------
    symmetry: function
        provide symmetry operators in quaternion for different crystal structure
    Quaternion: extension class
        quaternion representation of 3D orientation
    Eulers: extension class
        Euler angle representation of 3D orientation
    OrientationMatrix: extension class
        Matrix representation of 3D orientation
    Rodrigues: extension class
        Rodrigue vector representation of 3D orientation
    Xtallite: extension class
        physically equivalent to material point
    Aggregate: extension class
        physically equivalent to grain
"""

import  cython
import  math, random, os
import  numpy as np
cimport numpy as np
from    libc.math cimport sin, cos, sqrt


#############################
# SETUP FOR TYPE DEFINITION #
#############################
np.import_array()

cdef extern from "numpy/npy_math.h":
    double NPY_INFINITY

# Determine the right dtype for arrays of indices at compile time.
IF UNAME_MACHINE[-2:] == '64':
    intp = np.int64
ELSE:
    intp = np.int32

DTYPE = np.float64


cpdef symmetry(lattice):
    """
    DESCRIPTION
    -----------
    sym_operator = symmetry(lattice)
        Return the symmetry operators in list of quaternion for given
        lattice structure.
    PARAMETERS
    ----------
    lattice: string ['cubic', 'hexagonal', 'tetragonal', 'orthorhombic']
        Lattice structure for target crystal system
    RETURNS
    -------
    sym_operator: ndarray
        The symmetry operators used to calculate equivalent crystal
        orientation.
    """
    cdef double tmp
    cdef list   lattice_hcp   = ['hexagonal', 'hex', 'hcp']
    cdef list   lattice_cubic = ['bcc', 'fcc', 'cubic']

    lattice = str(lattice.lower())
    if lattice in lattice_cubic:
        tmp = sqrt(2)
        symQuats = [
                    [ 1.0,     0.0,     0.0,     0.0     ],
                    [ 0.0,     1.0,     0.0,     0.0     ],
                    [ 0.0,     0.0,     1.0,     0.0     ],
                    [ 0.0,     0.0,     0.0,     1.0     ],
                    [ 0.0,     0.0,     0.5*tmp, 0.5*tmp ],
                    [ 0.0,     0.0,     0.5*tmp,-0.5*tmp ],
                    [ 0.0,     0.5*tmp, 0.0,     0.5*tmp ],
                    [ 0.0,     0.5*tmp, 0.0,    -0.5*tmp ],
                    [ 0.0,     0.5*tmp,-0.5*tmp, 0.0     ],
                    [ 0.0,    -0.5*tmp,-0.5*tmp, 0.0     ],
                    [ 0.5,     0.5,     0.5,     0.5     ],
                    [-0.5,     0.5,     0.5,     0.5     ],
                    [-0.5,     0.5,     0.5,    -0.5     ],
                    [-0.5,     0.5,    -0.5,     0.5     ],
                    [-0.5,    -0.5,     0.5,     0.5     ],
                    [-0.5,    -0.5,     0.5,    -0.5     ],
                    [-0.5,    -0.5,    -0.5,     0.5     ],
                    [-0.5,     0.5,    -0.5,    -0.5     ],
                    [-0.5*tmp, 0.0,     0.0,     0.5*tmp ],
                    [ 0.5*tmp, 0.0,     0.0,     0.5*tmp ],
                    [-0.5*tmp, 0.0,     0.5*tmp, 0.0     ],
                    [-0.5*tmp, 0.0,    -0.5*tmp, 0.0     ],
                    [-0.5*tmp, 0.5*tmp, 0.0,     0.0     ],
                    [-0.5*tmp,-0.5*tmp, 0.0,     0.0     ],
                   ]
    elif lattice in lattice_hcp:
        tmp = sqrt(3)
        symQuats =  [
                     [ 1.0,      0.0,     0.0,      0.0     ],
                     [-0.5*tmp,  0.0,     0.0,     -0.5     ],
                     [ 0.5,      0.0,     0.0,      0.5*tmp ],
                     [ 0.0,      0.0,     0.0,      1.0     ],
                     [-0.5,      0.0,     0.0,      0.5*tmp ],
                     [-0.5*tmp,  0.0,     0.0,      0.5     ],
                     [ 0.0,      1.0,     0.0,      0.0     ],
                     [ 0.0,     -0.5*tmp, 0.5,      0.0     ],
                     [ 0.0,      0.5,    -0.5*tmp,  0.0     ],
                     [ 0.0,      0.0,     1.0,      0.0     ],
                     [ 0.0,     -0.5,    -0.5*tmp,  0.0     ],
                     [ 0.0,      0.5*tmp, 0.5,      0.0     ],
                    ]
    elif lattice == 'tetragonal':
        tmp = sqrt(2)
        symQuats =  [
                     [ 1.0,     0.0,     0.0,     0.0     ],
                     [ 0.0,     1.0,     0.0,     0.0     ],
                     [ 0.0,     0.0,     1.0,     0.0     ],
                     [ 0.0,     0.0,     0.0,     1.0     ],
                     [ 0.0,     0.5*tmp, 0.5*tmp, 0.0     ],
                     [ 0.0,    -0.5*tmp, 0.5*tmp, 0.0     ],
                     [ 0.5*tmp, 0.0,     0.0,     0.5*tmp ],
                     [-0.5*tmp, 0.0,     0.0,     0.5*tmp ],
                    ]
    elif lattice == 'triclinic':
        symQuats =  [
                     [ 1.0,0.0,0.0,0.0 ],
                    ]
    else:
        raise ValueError("Unknown lattice structure: {}".format(lattice))

    return np.array(symQuats)


cdef class Quaternion:
    """
    DESCRIPTION
    -----------
    Quaternion(np.array([w,x,y,z]))
        Quaternion is a set of numerics that extends from complex number,
        where a imaginary space (x,y,z) is constructed to facilitate a close
        set.
        Particularly, the unitary quaternions correspond to the rotation
        operation in 3D space, which is why many computer graphics used it
        to perform fast rotation calculations.
    PARAMETERS
    ----------
    q: DTYPE[:]
        Simple vector with length 4
    METHODS
    -------
    """

    def __init__(self, DTYPE_t[:] q):
        cdef DTYPE_t sgn = DTYPE_sgn(q[0])

        self.w = q[0] * sgn
        self.x = q[1] * sgn
        self.y = q[2] * sgn
        self.z = q[3] * sgn

    def __add__(self, Quaternion other):
        cdef np.ndarray newQ = np.zeros(4, dtype=DTYPE)

        newQ[0] = self.w + other.w
        newQ[1] = self.x + other.x
        newQ[2] = self.y + other.y
        newQ[3] = self.z + other.z
        return Quaternion(newQ)

    def __iadd__(self, Quaternion other):
        self.w = self.w + other.w
        self.x = self.x + other.x
        self.y = self.y + other.y
        self.z = self.z + other.z
        return self

    def __sub__(self, Quaternion other):
        cdef np.ndarray newQ = np.zeros(4, dtype=DTYPE)

        newQ[0] = self.w - other.w
        newQ[1] = self.x - other.x
        newQ[2] = self.y - other.y
        newQ[3] = self.z - other.z
        return Quaternion(newQ)

    def __isub__(self, Quaternion other):
        self.w = self.w - other.w
        self.x = self.x - other.x
        self.y = self.y - other.y
        self.z = self.z - other.z
        return self

    def __mul__(self, Quaternion other):
        cdef np.ndarray newQ = np.zeros(4, dtype=DTYPE)
        cdef DTYPE_t Aw = self.w
        cdef DTYPE_t Ax = self.x
        cdef DTYPE_t Ay = self.y
        cdef DTYPE_t Az = self.z
        cdef DTYPE_t Bw = other.w
        cdef DTYPE_t Bx = other.x
        cdef DTYPE_t By = other.y
        cdef DTYPE_t Bz = other.z

        newQ[0] = - Ax * Bx - Ay * By - Az * Bz + Aw * Bw
        newQ[1] = + Ax * Bw + Ay * Bz - Az * By + Aw * Bx
        newQ[2] = - Ax * Bz + Ay * Bw + Az * Bx + Aw * By
        newQ[3] = + Ax * By - Ay * Bx + Az * Bw + Aw * Bz
        return Quaternion(newQ)

    def __div__(self, Quaternion other):
        cdef np.ndarray newQ = np.zeros(4, dtype=DTYPE)
        pass

    def __str__(self):
        tmp = "({}, <{},{},{}>)".format(self.w, self.x, self.y, self.z)
        return tmp

    def __repr__(self):
        tmp = "Quaternion(real={:.4f}, imag=<{:.4f},{:.4f},{:.4f}>".format(self.w,
                                                                           self.x,
                                                                           self.y,
                                                                           self.z)
        return tmp

    def __abs__(self):
        cdef double tmp

        tmp = self.w*self.w + \
              self.x*self.x + \
              self.y*self.y + \
              self.z*self.z
        tmp = sqrt(tmp)
        return tmp

    def __len__(self):
        return 4

    def __neg__(self):
        self.w = -self.w
        self.x = -self.x
        self.y = -self.y
        self.z = -self.z
        return self

    def unitary(self):
        cdef double length = abs(self)
        cdef np.ndarray newQ = np.zeros(4, dtype=DTYPE)

        newQ[0] = self.w/length
        newQ[1] = self.x/length
        newQ[2] = self.y/length
        newQ[3] = self.z/length
        return Quaternion(newQ)

    def tolist(self):
        return [self.w, self.x, self.y, self.z]

    def tondarray(self):
        return np.array(self.tolist())

    def rotate(self, DTYPE_t[:] pt):
        pass


cdef class Eulers:
    """
    DESCRIPTION
    -----------
    Euler angle representation of orientation.
    Calculation is carries out by converting to quaternions.

    PARAMETERS
    ----------
    phi1: double
        first of Euler angle
    PHI:  double
        second of Euler angle
    phi2: double
        third of Euler angle
    METHODS
    -------
    """

    def __init__(self, phi1, PHI, phi2):
        self.phi1 = phi1
        self.PHI  = PHI
        self.phi2 = phi2

cdef class OrientationMatrix:
    """
    Matrix representation of orientation, this is defined as
    the transpose of the rotation matrix.

    PARAMETERS
    ----------
    METHODS
    """

    def __init__(self, g):
        self.g = g.copy()


cdef class Xtallite:
    """Aggregate class to represent real crystallite in material.

    NOTE
    ----
    PARAMETERS
    ----------
        eulers  : Euler angles in degrees
        lattice : lattice structure, e.g. "hexagonal"
    ATTRIBUTES
    ----------
    METHODS
    -------
    """
    cdef public Quaternion   orientation
    cdef public double[:,:]  op_sym

    def __init__(self, eulers, lattice):
        self.orientation = Quaternion.eulers2Quaternion(eulers)
        self.op_sym = symmetry(lattice)
