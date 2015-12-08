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
from    libc.math cimport sin, cos, sqrt, atan2, M_PI


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
    unitary(self)
        Return a unitary quaternion, useful for using quaternion to represent
        rotation/orientation.
    conj(self)
        Return the conjugate of the quaternion
    tolist(self)
        Return the quaternion as a simple python list
    tondarray(self)
        Return the quaternion as a numpy array (preferred)
    toEulers(self)
        Convert a unitary quaternion into Euler Angles (np.ndarray)
    toRodrigues(self)
        Convert a unitary quaternion into Rodrigue vector (np.ndarray)
    toOrientationMatrix(self)
        Convert a unitary quaternion into Orientation Matrix (np.ndarray)
    -------
    """

    def __init__(self, DTYPE_t[:] q):
        cdef DTYPE_t sgn = DTYPE_sgn(q[0])

        self.w = q[0] * sgn
        self.x = q[1] * sgn
        self.y = q[2] * sgn
        self.z = q[3] * sgn

    def __copy__(self):
        return Quaternion([self.w,self.x,self.y,self.z])

    def __richcmp__(self, Quaternion other, int op):
        cdef bint flag

        flag = ( abs( self.w-other.w) < 1e-8 and \
                 abs( self.x-other.x) < 1e-8 and \
                 abs( self.y-other.y) < 1e-8 and \
                 abs( self.z-other.z) < 1e-8)    \
                or                               \
               ( abs(-self.w-other.w) < 1e-8 and \
                 abs(-self.x-other.x) < 1e-8 and \
                 abs(-self.y-other.y) < 1e-8 and \
                 abs(-self.z-other.z) < 1e-8)
        if op == 2:  #__eq__
            return flag
        elif op == 3:
            return not flag
        else:
            return NotImplemented

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

    def __imul__(self, Quaternion other):
        cdef DTYPE_t Aw = self.w
        cdef DTYPE_t Ax = self.x
        cdef DTYPE_t Ay = self.y
        cdef DTYPE_t Az = self.z
        cdef DTYPE_t Bw = other.w
        cdef DTYPE_t Bx = other.x
        cdef DTYPE_t By = other.y
        cdef DTYPE_t Bz = other.z

        self.w = - Ax * Bx - Ay * By - Az * Bz + Aw * Bw
        self.x = + Ax * Bw + Ay * Bz - Az * By + Aw * Bx
        self.y = - Ax * Bz + Ay * Bw + Az * Bx + Aw * By
        self.z = + Ax * By - Ay * Bx + Az * Bw + Aw * Bz
        return self

    def __div__(self, Quaternion other):
        return self * other.conj()

    def __idiv__(self, Quaternion other):
        cdef Quaternion tmp = self * other.conj()

        self.w = tmp.w
        self.x = tmp.x
        self.y = tmp.y
        self.z = tmp.z
        return self

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

    def conj(self):
        """
        DESCRIPTION
        -----------
        q.conj()
            Representing the inverse rotation of q, provided
        q is a unitary quaternion.
        """
        cdef np.ndarray newQ = np.zeros(4, dtype=DTYPE)

        newQ[0] =  self.w
        newQ[1] = -self.x
        newQ[2] = -self.y
        newQ[3] = -self.z
        return Quaternion(newQ)

    def tolist(self):
        return [self.w, self.x, self.y, self.z]

    def tondarray(self):
        return np.array(self.tolist())

    def toEulers(self, inDegrees=True, inStandardRange=True):
        """
        Conversion of ACTIVE rotation to Euler angles taken from:
        Melcher, A.; Unser, A.; Reichhardt, M.; Nestler, B.; Pötschke, M.; Selzer, M.
        Conversion of EBSD data by a quaternion based algorithm to be used for grain structure simulations
        Technische Mechanik 30 (2010) pp 401--413
        """
        cdef np.ndarray angles = np.zeros(3, dtype=DTYPE)
        cdef DTYPE_t    x      = 0.0
        cdef DTYPE_t    y      = 0.0
        cdef DTYPE_t    chi    = 0.0
        cdef Quaternion q = self.unitary()

        if DTYPE_abs(q.x)<1e-4 and DTYPE_abs(q.y)<1e-4:
            x = q.w**2.0 - q.z**2.0
            y = 2.0 * q.w * q.z
            angles[0] = atan2(y,x)
        elif DTYPE_abs(q.w) < 1e-4 and DTYPE_abs(q.z)<1e-4:
            x = q.x**2.0 - q.y**2.0
            y = 2.0*q.x*q.y
            angles[0] = atan2(y,x)
            angles[1] = M_PI
        else:
            chi = sqrt((q.w**2 + q.z**2)*(q.x**2 + q.y**2))

            x   = (q.w * q.x - q.y * q.z)/2./chi
            y   = (q.w * q.y + q.x * q.z)/2./chi
            angles[0] = atan2(y,x)

            x   = q.w**2 + q.z**2 - (q.x**2 + q.y**2)
            y   = 2.*chi
            angles[1] = atan2(y,x)

            x = (q.w * q.x + q.y * q.z)/2./chi
            y = (q.z * q.x - q.y * q.w)/2./chi
            angles[2] = atan2(y,x)

        if inStandardRange:
            angles[0] %= 2*M_PI
            if angles[1] < 0.0:
                angles[1] =  angles[1] + M_PI
                angles[2] = -angles[2]
            angles[2] %= 2*M_PI

        if inDegrees:
            angles = np.degrees(angles)

        return angles

    def toRodrigues(self):
        cdef np.ndarray r = np.ones(3, dtype=DTYPE)

        if DTYPE_abs(self.w)<1e-6:
            r = np.inf * r
        else:
            r[0] = self.x/self.w
            r[1] = self.y/self.w
            r[2] = self.z/self.w

        return r

    def toOrientationMatrix(self):
        cdef Quaternion q = self.unitary()
        cdef np.ndarray m = np.empty((3,3), dtype=DTYPE)

        m = np.array([[1.0-2.0*(q.y*q.y+q.z*q.z),     2.0*(q.x*q.y-q.z*q.w),     2.0*(q.x*q.z+q.y*q.w)],
                      [    2.0*(q.x*q.y+q.z*q.w), 1.0-2.0*(q.x*q.x+q.z*q.z),     2.0*(q.y*q.z-q.x*q.w)],
                      [    2.0*(q.x*q.z-q.y*q.w),     2.0*(q.x*q.w+q.y*q.z), 1.0-2.0*(q.x*q.x+q.y*q.y)]])

        return m

    def toAngleAxis(self):
        """
        DESCRIPTION
        -----------
        angle, rotation_axis = q.toAngleAxis()
            Return the angle-axis pair that equivalent to the rotation
        represented by q.unitary().
        RETURNS
        -------
        (angle, v) : tuple
        """
        cdef np.ndarray v = np.zeros(3, dtype=DTYPE)
        cdef Quaternion q = self.unitary()
        cdef DTYPE_t    s,x,y,angle

        s = sqrt(1.0 - q.w**2)
        x = 2.0*q.w**2 - 1.0
        y = 2.0*q.w*s

        angle = atan2(y,x)
        if angle < 0.0:
            angle = -angle
            s     = -s

        if DTYPE_abs(angle) < 1e-4:
            v[0] = 1.0
        else:
            v[0] = q.x/s
            v[1] = q.y/s
            v[2] = q.z/s

        return (angle, v)

    @classmethod
    def scale(cls, Quaternion q, double s):
        """
        DESCRIPTION
        -----------
        newQ = Quaternion.scale(q, s)
            Scale a quaternion with given scalar
        PARAMETERS
        ----------
        q: Quaternion
            Quaternion to scale
        s: double
            Scaling amount
        """
        cdef np.ndarray newQ = np.zeros(4, dtype=DTYPE)

        newQ[0] = q.w * s
        newQ[1] = q.x * s
        newQ[2] = q.y * s
        newQ[3] = q.z * s

        return Quaternion(newQ)

    @classmethod
    def rotate(cls, Quaternion q, DTYPE_t[:] pt):
        """
        DESCRIPTION
        -----------
        newPt = Quaternion.rotate(q, pt)
            active rotation of pt using quaternion q, namely
            'newPt = q * pt * q.conj()'
        PARAMETERS
        ----------
        q: Quaternion
            quaternion defining rotation
        pt: np.ndarray
            Point vector
        RETURNS
        -------
        newPt: np.ndarray
            new point vector
        """
        cdef np.ndarray newPt = np.zeros(3, dtype=DTYPE)
        cdef Quaternion rotQ  = q.unitary()
        cdef DTYPE_t w  = rotQ.w
        cdef DTYPE_t x  = rotQ.x
        cdef DTYPE_t y  = rotQ.y
        cdef DTYPE_t z  = rotQ.z
        cdef DTYPE_t Vx = pt[0]
        cdef DTYPE_t Vy = pt[1]
        cdef DTYPE_t Vz = pt[2]

        newPt[0] = w * w * Vx + 2 * y * w * Vz - 2 * z * w * Vy + \
                   x * x * Vx + 2 * y * x * Vy + 2 * z * x * Vz - \
                   z * z * Vx - y * y * Vx
        newPt[1] = 2 * x * y * Vx + y * y * Vy + 2 * z * y * Vz + \
                   2 * w * z * Vx - z * z * Vy + w * w * Vy - \
                   2 * x * w * Vz - x * x * Vy
        newPt[2] = 2 * x * z * Vx + 2 * y * z * Vy + \
                   z * z * Vz - 2 * w * y * Vx - y * y * Vz + \
                   2 * w * x * Vy - x * x * Vz + w * w * Vz
        return newPt

    @classmethod
    def average(cls, list qs):
        """
        DESCRIPTION
        -----------
        Q_avg = Quaternion.average(listOfQuaternions)
            Return the average quaternion based on algorithm published in
        F. Landis Markley, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman.
        Averaging Quaternions,
        Journal of Guidance, Control, and Dynamics,
        Vol. 30, No. 4 (2007), pp. 1193-1197.
        doi: 10.2514/1.28949
        NOTE
        ----
            No crystal symmetry considered at this level, just plain averaging
        list of unitary quaternions.
        """
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

    def __init__(self):
        pass
