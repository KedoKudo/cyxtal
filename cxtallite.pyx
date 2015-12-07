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
    Quaternion representation of orientation
    All methods and naming conventions based off
    http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions

    PARAMETERS
    ----------
    q: double[:]
        Simple vector with length 4
    ATTRIBUTES
    ----------
    METHODS
    -------
    """

    def __init__(self, double[:] q):
        if q[0] < 0:
            self.w = -q[0]
            self.x = -q[1]
            self.y = -q[2]
            self.z = -q[3]
        else:
            self.w = q[0]
            self.x = q[1]
            self.y = q[2]
            self.z = q[3]


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
