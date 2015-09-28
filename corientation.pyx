#!/usr/bin/env python
# encoding: utf-8
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
"""

import cython
import math, random, os
import numpy as np
cimport numpy as np


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
    sym_operator: list of Quaternion
        The symmetry operators used to calculate equivalent crystal
        orientation.
    """
    cdef double tmp

    lattice = lattice.lower()
    if lattice == 'cubic':
        tmp = math.sqrt(2)
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
    elif lattice == 'hexagonal':
        tmp = math.sqrt(3)
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
        tmp = math.sqrt(2)
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
    else:
        symQuats =  [
                     [ 1.0,0.0,0.0,0.0 ],
                    ]
    return symQuats


cdef class Quaternion:
    """
    Quaternion representation of orientation
    All methods and naming conventions based off
    http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions
    """

    def __init__(self, double[:] q):
        """
        DESCRIPTION
        -----------
        new_q = Quaternion(v)
            Initialize Quaternion instance with vector.
        PARAMETERS
        ----------
        q: double[:]
            Simple vector with length 4
        """
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

    def __copy__(self):
        cdef double[4] q = [self.w,self.x,self.y,self.z]
        return Quaternion(q)

    @staticmethod
    cdef eulers2Quaternion(double[:] e):
        """
        DESCRIPTION
        -----------
            Q = eulers2Quaternion(eulerAgnles)
            convert euler_angle to quaternion representation
        PARAMETERS
        ----------
        e: double[3]
            Euler angles in degrees
        RETURNS
        -------
        q: Quaternion
            Return a quaternion instance of given orientation
        NOTE
        ----
        Always assume using Bunge convention (z-x-z)
        """
        cdef double     c1,s1,c2,s2,c3,s3
        cdef double[4]  q
        cdef double[3]  halfEulers
        cdef int        i

        for i in range(3):
            halfEulers[i] = e[i] * 0.5

        c1 = math.cos(halfEulers[0])
        s1 = math.sin(halfEulers[0])
        c2 = math.cos(halfEulers[1])
        s2 = math.sin(halfEulers[1])
        c3 = math.cos(halfEulers[2])
        s3 = math.sin(halfEulers[2])

        q[0] = c1 * c2 * c3 - s1 * s2 * s3
        q[1] = s1 * s2 * c3 + c1 * c2 * s3
        q[2] = s1 * c2 * c3 + c1 * s2 * s3
        q[3] = c1 * s2 * c3 - s1 * c2 * s3

        return Quaternion(q)


    @staticmethod
    cdef rodrigues2Quaternion(double[:] r):
        """
        DESCRIPTION
        -----------
            Q = rodrigues2Quaternion(r)
            Convert a Rodrigues vector (length 3) to a Quaternion
        PARAMETERS
        ----------
        r: double[:]
            Memoryview of a length 3 vector
        RETURNS
        Q: Quaternion()
            Quaternion instance of given orientation
        -------

        """
        pass

    @staticmethod
    cdef oMatrix2Quaternion(double[:,:] m):
        """
        """
        pass



































