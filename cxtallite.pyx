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

import  cython
import  math, random, os
import  numpy as np
cimport numpy as np
from    libc.math cimport sin, cos, sqrt


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

    lattice = lattice.lower()
    if lattice == 'cubic':
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
    elif lattice == 'hexagonal':
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
    else:
        symQuats =  [
                     [ 1.0,0.0,0.0,0.0 ],
                    ]

    return np.array(symQuats)


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

    def asEulers(self):
        """
        DESCRIPTION
        -----------
            eulers = Q.asEulers()
            Return orientation in Euler angles
        RETURNS
        -------
        eulers : list [phi1, PHI, phi2]
            Euler angles in degrees
        NOTE
        ----
        CONVERSION TAKEN FROM:
        Melcher, A.; Unser, A.; Reichhardt, M.; Nestler, B.; Pötschke, M.; Selzer, M.
        Conversion of EBSD data by a quaternion based algorithm to be used for grain
        structure simulations Technische Mechanik 30 (2010) pp 401--413
        """
        cdef double     x, y, chi
        cdef np.ndarray eulers = np.zeros(3, type=float)

        if abs(self.x) < 1e-4 and abs(self.y) < 1e-4:
            x = self.w**2 - self.z**2
            y = 2.*self.w*self.z
            eulers[0] = math.atan2(y,x)
        elif abs(self.w) < 1e-4 and abs(self.z) < 1e-4:
            x = self.x**2 - self.y**2
            y = 2.*self.x*self.y
            eulers[0] = math.atan2(y,x)
            eulers[1] = math.pi
        else:
            chi = math.sqrt((self.w**2 + self.z**2)*(self.x**2 + self.y**2))

            x = (self.w * self.x - self.y * self.z)/2./chi
            y = (self.w * self.y + self.x * self.z)/2./chi
            eulers[0] = math.atan2(y,x)

            x = self.w**2 + self.z**2 - (self.x**2 + self.y**2)
            y = 2.*chi
            eulers[1] = math.atan2(y,x)

            x = (self.w * self.x + self.y * self.z)/2./chi
            y = (self.z * self.x - self.y * self.w)/2./chi
            eulers[2] = math.atan2(y,x)

        # convert to standard range
        eulers[0] %= 2*math.pi
        if eulers[1] < 0.0:
            eulers[1] += math.pi
            eulers[2] *= -1.0
        eulers[2] %= 2*math.pi

        return np.degrees(eulers)

    def asOrientationMatrix(self):
        """
        """
        pass

    def asRodrigues(self):
        """
        """
        pass

    def asAngleAxis(self):
        """
        """
        pass

    @staticmethod
    def eulers2Quaternion(double[:] e):
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

        c1 = cos(halfEulers[0])
        s1 = sin(halfEulers[0])
        c2 = cos(halfEulers[1])
        s2 = sin(halfEulers[1])
        c3 = cos(halfEulers[2])
        s3 = sin(halfEulers[2])

        q[0] = c1 * c2 * c3 - s1 * s2 * s3
        q[1] = s1 * s2 * c3 + c1 * c2 * s3
        q[2] = s1 * c2 * c3 + c1 * s2 * s3
        q[3] = c1 * s2 * c3 - s1 * c2 * s3

        return Quaternion(q)

    @staticmethod
    def rodrigues2Quaternion(double[:] r):
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
        -------
        Q: Quaternion()
            Quaternion instance of given orientation
        """
        cdef double        norm, halfAngle
        cdef double[4]     q
        cdef int           i

        norm      = np.linalg.norm(r)
        halfAngle = np.arctan(norm)
        q[0]      = cos(halfAngle)

        for i in range(3):
            q[i+1] = sin(halfAngle) * r[i] / norm

        return Quaternion(q)

    @staticmethod
    def oMatrix2Quaternion(double[:,:] m):
        """
        DESCRIPTION
        -----------
            Q = oMatrix2Quaternion(m)
            convert orientation matrix to Quaternion representation
            ref: http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        PARAMETERS
        ----------
        m : np.ndarray
            3x3 orientation matrix
        RETURNS
        -------
        Q : Quaternion
            Quaternion instance of given orientation
        """
        cdef double    trace, s, t
        cdef double[4] q
        cdef int       i

        trace = np.trace(m)

        if trace > 1e-8:
            s    = sqrt(trace + 1.0) * 2.0

            q[0] = s * 0.25
            q[1] = (m[2,1] - m[1,2])/s
            q[2] = (m[0,2] - m[2,0])/s
            q[3] = (m[1,0] - m[0,1])/s

        elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
            t    = m[0,0] - m[1,1] - m[2,2] + 1.0
            s    = 2.0*sqrt(t)

            q[0] = (m[2,1] - m[1,2])/s
            q[1] = s*0.25
            q[2] = (m[0,1] + m[1,0])/s
            q[3] = (m[2,0] + m[0,2])/s

        elif m[1,1] > m[2,2]:
            t    = -m[0,0] + m[1,1] - m[2,2] + 1.0
            s    = 2.0 * sqrt(t)

            q[0] = (m[0,2] - m[2,0])/s
            q[1] = (m[0,1] + m[1,0])/s
            q[2] = s*0.25
            q[3] = (m[1,2] + m[2,1])/s

        else:
            t    = -m[0,0] - m[1,1] + m[2,2] + 1.0
            s    = 2.0 * sqrt(t)

            q[0] = (m[1,0] - m[0,1])/s
            q[1] = (m[2,0] + m[0,2])/s
            q[2] = (m[1,2] + m[2,1])/s
            q[3] = s*0.25

        return Quaternion(q)


cdef class crystallite:
    cdef public Quaternion   orientation
    cdef public double[:,:]  op_sym

    def __init__(eulers, symmetry):
        pass






























