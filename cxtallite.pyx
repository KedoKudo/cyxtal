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
from    libc.math cimport sin, cos, sqrt, atan2, M_PI, atan


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

# CONSTANT #
cdef list       lattice_hcp   = ['hexagonal', 'hex', 'hcp']
cdef list       lattice_cubic = ['bcc', 'fcc', 'cubic']
cdef list       lattice_tet   = ['tetragonal']
cdef list       lattice_orth  = ['orthorhombic']
cdef list       lattice_tric  = ['triclinic']
cdef DTYPE_t    d2r           = M_PI/180.0
cdef DTYPE_t    sqrt_2        = sqrt(2.0)
cdef DTYPE_t    sqrt_3        = sqrt(3.0)


#----------------------#
# MODULE LEVEL CLASSES #
#----------------------#
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
    CLASSMETHOD
    -----------
    scale(Quaternion q, DTYPE_t scalar)
        Scale a quaternion vector with given scalar.
    rotate(Quaternion q, DTYPE_t[:] pt)
        Rotate pt around origin by q.
    average(list qs)
        Return an approximation of the average quaternion (forced to unitary)
        for qs (list of quaternions).
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
        list of unitary quaternions. Also the results coming out of this method
        is not accurate, e.g. Euler angles [10,0,0], [30,0,0], [90,0,0]
            theoretical results: [43.33, 0.0, 0.0]
            numerical results:   [42.12, 0.0, 0.0]
        """
        cdef int        N   = len(qs)
        cdef np.ndarray M   = np.zeros((4,4), dtype=DTYPE)
        cdef np.ndarray eig = np.empty( 4,    dtype=DTYPE)
        cdef np.ndarray vec = np.empty((4,4), dtype=DTYPE)

        cdef Quaternion q
        cdef int        i


        for i in range(N):
            q = qs[i].unitary()
            M += np.outer(q.tondarray(), q.tondarray())

        eig, vec = np.linalg.eig(M/N)

        return Quaternion(np.real(vec.T[eig.argmax()]))


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

    def __init__(self, double phi1, double PHI, double phi2):
        self.phi1 = phi1
        self.PHI  = PHI
        self.phi2 = phi2
        self.__q  = Quaternion(self.__getq())

    def __getq(self):
        """ Return a quaternion based on given Euler Angles """
        cdef np.ndarray  qv = np.zeros(4, dtype=DTYPE)
        cdef DTYPE_t    c1,s1,c2,s2,c3,s3

        c1 = cos(self.phi1 * d2r / 2.0)
        s1 = sin(self.phi1 * d2r / 2.0)
        c2 = cos(self.PHI  * d2r / 2.0)
        s2 = sin(self.PHI  * d2r / 2.0)
        c3 = cos(self.phi2 * d2r / 2.0)
        s3 = sin(self.phi2 * d2r / 2.0)

        qv[0] =   c1 * c2 * c3 - s1 * c2 * s3
        qv[1] =   c1 * s2 * c3 + s1 * s2 * s3
        qv[2] = - c1 * s2 * s3 + s1 * s2 * c3
        qv[3] =   c1 * c2 * s3 + s1 * c2 * c3

        return qv

    def toQuaternion(self):
        return self.__q

    def tolist(self):
        return [self.phi1, self.PHI, self.phi2]

    def tondarray(self):
        return np.array(self.tolist())

    def toRodrigues(self):
        return self.__q.toRodrigues()

    def toOrientationMatrix(self):
        return self.__q.toOrientationMatrix()

    def toAngleAxis(self):
        return self.__q.toAngleAxis()


cdef class OrientationMatrix:
    """
    Matrix representation of orientation, this is defined as
    the transpose of the rotation matrix.

    PARAMETERS
    ----------
    METHODS
    """

    def __init__(self, DTYPE_t[:,:] g):
        cdef int i, j

        self.g = np.zeros((3,3), dtype=DTYPE)
        for i in range(3):
            for j in range(3):
                self.g[i,j]   = g[i,j]

        self.__q = self.__getq()

    cdef Quaternion __getq(self):
        cdef DTYPE_t trace, s, t
        cdef np.ndarray qv = np.zeros(4, dtype=DTYPE)
        cdef DTYPE_t w, x, y, z

        trace = np.trace(self.g)
        if trace > 1e-8:
            s = sqrt(trace + 1.0)*2.0

            qv[0] = s*0.25
            qv[1] = (self.g[2,1] - self.g[1,2])/s
            qv[2] = (self.g[0,2] - self.g[2,0])/s
            qv[3] = (self.g[1,0] - self.g[0,1])/s
        elif (self.g[0,0] > self.g[1,1]) and (self.g[0,0] > self.g[2,2]):
            t = self.g[0,0] - self.g[1,1] - self.g[2,2] + 1.0
            s = 2.0*sqrt(t)

            qv[0] = (self.g[2,1] - self.g[1,2])/s
            qv[1] = s*0.25
            qv[2] = (self.g[0,1] + self.g[1,0])/s
            qv[3] = (self.g[2,0] + self.g[0,2])/s
        elif self.g[1,1] > self.g[2,2]:
            t = -self.g[0,0] + self.g[1,1] - self.g[2,2] + 1.0
            s = 2.0*sqrt(t)

            qv[0] = (self.g[0,2] - self.g[2,0])/s
            qv[1] = (self.g[0,1] + self.g[1,0])/s
            qv[2] = s*0.25
            qv[3] = (self.g[1,2] + self.g[2,1])/s
        else:
            t = -self.g[0,0] - self.g[1,1] + self.g[2,2] + 1.0
            s = 2.0*sqrt(t)

            qv[0] = (self.g[1,0] - self.g[0,1])/s
            qv[1] = (self.g[2,0] + self.g[0,2])/s
            qv[2] = (self.g[1,2] + self.g[2,1])/s
            qv[3] = s*0.25

        return Quaternion(qv)

    def tondarray(self):
        return self.g

    def toEulers(self):
        return self.__q.toEulers()

    def toRodrigues(self):
        return self.__q.toRodrigues()

    def toQuaternion(self):
        return self.__q


cdef class Rodrigues:
    """
    DESCRIPTION
    -----------
    Rodrigues representation of orientation, a wrapper class that use
    Quaternion class as engine.
    """

    def __init__(self, DTYPE_t[:] v):
        cdef int i

        self.v = np.zeros(3, dtype=DTYPE)
        for i in range(3):
            self.v[i] = v[i]

        self.__q = self.__getq()

    def __getq(self):
        cdef np.ndarray qv = np.zeros(4, dtype=DTYPE)
        cdef DTYPE_t    halfangle, c, s

        halfangle = atan(np.linalg.norm(self.v))
        c         = cos(halfangle)
        s         = sin(halfangle)

        qv[0] = s
        qv[1] = c * self.v[0]
        qv[2] = c * self.v[1]
        qv[3] = c * self.v[2]

        return Quaternion(qv)

    def tolist(self):
        return list(self.v)

    def tondarray(self):
        return np.array(self.tolist())

    def toQuaternion(self):
        return self.__q

    def toAngleAxis(self):
        return self.__q.toAngleAxis()

    def toEulers(self):
        return self.__q.toEulers()

    def toOrientationMatrix(self):
        return self.__q.toOrientationMatrix()


cdef class Xtallite:
    """
    DESCRIPTION
    -----------
    Composite class to represent material point in general crystal plasticity
    simulation.

    PARAMETERS
    ----------

    METHODS
    -------
    """

    def __init__(self,
                 eulers=np.zeros(3, dtype=DTYPE),
                 pt=np.zeros(3, dtype=DTYPE),
                 lattice='bcc',
                 dv=np.zeros(3, dtype=DTYPE),
                 stress=np.zeros((3,3), dtype=DTYPE),
                 strain=np.zeros((3,3), dtype=DTYPE)
                 ):
        cdef double phi1,PHI,phi2

        self.eulers = np.copy(eulers)
        phi1        = self.eulers[0]
        PHI         = self.eulers[1]
        phi2        = self.eulers[2]
        self.__q    = Eulers(phi1, PHI, phi2).toQuaternion()

        self.pt      = np.copy(pt)
        self.dv      = np.copy(dv)
        self.stress  = np.copy(stress)
        self.strain  = np.copy(strain)

        self.lattice = lattice

    def setEulers(self, DTYPE_t phi1, DTYPE_t PHI, DTYPE_t phi2):
        self.eulers = np.array([phi1, PHI, phi2])
        self.__q    = Eulers(phi1, PHI, phi2).toQuaternion()

    def setLattice(self, str newLattice):
        self.lattice = newLattice

    def getOrientation(self, str mode='eulers'):
        """
        DESCRIPTION
        -----------
        """
        mode = mode.lower()

        if mode == 'eulers':
            return self.__q.toEulers()
        elif mode == 'quaternion':
            return self.__q
        elif mode == 'rodrigue':
            return self.__q.toRodrigues()
        else:
            raise ValueError("Unknown mode: {}".format(mode))

    def toFundamentalZone(self):
        cdef Quaternion q
        cdef np.ndarray symop
        cdef int        i

        symops = symmetry(self.lattice, mode='quaternion')
        for i in range(len(symops)):
            q = self.__q * symops[i]
            if Xtallite.inFundamentalZone(q.toRodrigues(), self.lattice):
                self.__q = q
                self.eulers = q.toEulers()

        return self.eulers

    def disorientation(self, Xtallite other, str mode='angle'):
        """
        DESCRIPTION
        -----------
        """
        cdef Quaternion q0 = self.__q
        cdef Quaternion q1 = other.getOrientation(mode='quaternion')
        cdef Quaternion dq
        cdef str        lattice

        if self.lattice.lower() != other.lattice.lower():
            raise ValueError("ERROR: {}!={}".format(self.lattice, other.lattice))
        else:
            lattice = self.lattice.lower()

        dq = self.getDq(q0, q1, lattice)

        mode = mode.lower()
        if mode == 'angle':
            return np.degrees(dq.toAngleAxis()[0])
        elif mode == 'quaternion':
            return dq
        elif mode == 'angleaxis':
            return dq.toAngleAxis()
        elif mode == 'eulers':
            return dq.toEulers()
        elif mode == 'axis':
            return dq.toAngleAxis()[1]
        else:
            raise ValueError("Unknown mode: {}".format(mode))

    cdef Quaternion getDq(self, Quaternion q0, Quaternion q1, str lattice):
        """
        DESCRIPTION
        -----------
        dQ = getDq(q0, q1, lattice)
        """
        cdef Quaternion deltaQ, tmp0, tmp1, tmpQ
        cdef np.ndarray symop
        cdef int        i, j
        cdef DTYPE_t    angle = 360

        symops = symmetry(lattice, mode='quaternion')

        # use brutal force to get the smallest rotation angle
        for i in range(len(symops)):
            for j in range(len(symops)):
                tmp0   = q0*symops[i]
                tmp1   = q1*symops[j]
                tmpQ   = tmp0.conj() * tmp1
                if angle > tmpQ.toAngleAxis()[0]:
                    angle  = tmpQ.toAngleAxis()[0]
                    deltaQ = tmpQ

        return deltaQ

    def disorientations(self, list others):
        """
        DESCRIPTION
        -----------
        dQs = disorientations(list ListOfXtallite)
            Provide batch processing capability of disorientation
        calculation. Assuming the crystallites in the list has the
        same lattice structure as self.
        """
        cdef Quaternion q0  = self.__q
        cdef Quaternion q1, dq
        cdef np.ndarray dqs = np.empty(len(others), dtype=DTYPE)
        cdef int        N   = len(others)
        cdef int        i

        for i in range(N):
            q1     = others[i].getOrientation(mode='quaternion')
            dq     = self.getDq(q0,q1,self.lattice)
            dqs[i] = np.degrees(dq.toAngleAxis()[0])

        return dqs

    @classmethod
    def random(cls):
        eulers  = np.degrees(np.random.random(3))
        return Xtallite(eulers=eulers)

    @classmethod
    def inDisorientationStandardZone(cls,
                                     Quaternion deltaQ,
                                     str        lattice):
        '''
        DESCRIPTION
        -----------
        flag = Xtallite.inDisorientationStandardZone(q, lattice)
        Check whether given Rodrigues vector (of misorientation) falls into standard stereographic triangle of own symmetry.
        Determination of disorientations follow the work of A. Heinz and P. Neumann:
        Representation of Orientation and Disorientation Data for Cubic, Hexagonal, Tetragonal and Orthorhombic Crystals
        Acta Cryst. (1991). A47, 780-789

        PARAMETERS
        ----------
        '''
        cdef DTYPE_t    epsilon = 0.0
        cdef DTYPE_t[:] R       = deltaQ.toRodrigues()

        if lattice in lattice_cubic:
            return R[0] >= R[1]+epsilon                and R[1] >= R[2]+epsilon    and R[2] >= epsilon

        elif lattice in lattice_hcp:
            return R[0] >= math.sqrt(3)*(R[1]-epsilon) and R[1] >= epsilon         and R[2] >= epsilon

        elif lattice in lattice_tet:
            return R[0] >= R[1]-epsilon                and R[1] >= epsilon         and R[2] >= epsilon

        elif lattice in lattice_orth:
            return R[0] >= epsilon                     and R[1] >= epsilon         and R[2] >= epsilon

        else:
            return True

    @classmethod
    def inFundamentalZone(cls,
                          DTYPE_t[:] r,
                          str lattice):
        """
        DESCRIPTION
        -----------
        flag = Xtallite.inFundamentalZone(r, lattice)
            Check whether given Rodrigues vector is in the fundamental zone
        for lattice structure
        PARAMETERS
        ----------
        r: DTYPE_t[:]
            Rodrigues vector in numpy array
        lattice: str
            lattice structure
        """
        cdef np.ndarray R     = np.absolute(r)

        if lattice in lattice_cubic:
            return     (sqrt_2 - 1.0 >= R[0]) \
                   and (sqrt_2 - 1.0 >= R[1]) \
                   and (sqrt_2 - 1.0 >= R[2]) \
                   and 1.0 >= R[0]+R[1]+R[2]
        elif lattice in lattice_hcp:
            return     1.0 >= R[0] \
                   and 1.0 >= R[1] \
                   and 1.0 >= R[2] \
                   and 2.0 >= sqrt_3*R[0] + R[1] \
                   and 2.0 >= sqrt_3*R[1] + R[0] \
                   and 2.0 >= sqrt_3 + R[2]
        elif lattice in lattice_tet:
            return     1.0 >= R[0] \
                   and 1.0 >= R[1] \
                   and sqrt_2 >= R[0] + R[1] \
                   and sqrt_2 >= R[2] + 1.0
        elif lattice in lattice_orth:
            return     1.0 >= R[0] \
                   and 1.0 >= R[1] \
                   and 1.0 >= R[2]
        else:
            raise ValueError("Unknown lattice structure: {}".format(lattice))


cdef class Aggregate:
    """
    DESCRIPTION
    -----------
    grainX = Aggregate(ListOfXtallites)
        A container class that holds several
    """

    def __init__(self, list xtals,
                 INTP_t texture=0,
                 INTP_t gid=0):
        self.xtals   = xtals
        self.texture = texture
        self.gid     = gid

    def getOrientation(self, mode='eulers'):
        """
        DESCRIPTION
        -----------
        Orientaiton_grain = grain.getOrientation()
            Return the grain average orientation by averaging all the
        orientations within this aggregate.
        """
        cdef Quaternion Qavg

        Qavg = self.__findAverageQ()

        mode = mode.lower()
        if mode == 'eulers':
            return Qavg.toEulers()
        elif mode == 'quaternion':
            return Qavg
        elif mode == 'rodrigue':
            return Qavg.toRodrigues()
        else:
            raise ValueError("Unknown mode: {}".format(mode))

    cdef Quaternion __findAverageQ(self):
        """
        DESCRIPTION
        -----------
        """
        cdef INTP_t     N       = len(self.xtals)
        cdef list       qs      = [None] * N
        cdef str        lattice = self.xtals[0].lattice
        cdef list       symops  = symmetry(lattice, mode='quaternion')
        cdef INTP_t     Nsyms   = len(symops)

        cdef int        i
        cdef double     angle, dA
        cdef Quaternion Qavg, refQ, Qtmp, dQ, close2refQ
        cdef Xtallite   refXtal = self.xtals[0]

        # use first xtallite as orientation reference
        refXtal.toFundamentalZone()
        refQ  = refXtal.getOrientation(mode='quaternion')
        qs[0] = refQ

        for i in range(1, N):
            angle = 360.0
            Qtmp  = self.xtals[i].getOrientation(mode='quaternion')
            qs[i] = Qtmp
            # find q equivalent q close to reference
            for j in range(Nsyms):
                dQ = refXtal.getDq(refQ, Qtmp*symops[j], lattice)
                dA = np.degrees(dQ.toAngleAxis()[0])
                if angle > dA:
                    angle = dA
                    qs[i] = Qtmp*symops[j]

        Qavg = Quaternion.average(qs)
        return Qavg

#-------------------------#
# MODULE LEVEL SINGLETONS #
#-------------------------#
def symmetry(str lattice,
             str mode='numpy'):
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
    lattice = lattice.lower()

    if lattice in lattice_cubic:
        symQuats = [
                    [ 1.0,       0.0,       0.0,       0.0       ],
                    [ 0.0,       1.0,       0.0,       0.0       ],
                    [ 0.0,       0.0,       1.0,       0.0       ],
                    [ 0.0,       0.0,       0.0,       1.0       ],
                    [ 0.0,       0.0,       0.5*sqrt_2, 0.5*sqrt_2 ],
                    [ 0.0,       0.0,       0.5*sqrt_2,-0.5*sqrt_2 ],
                    [ 0.0,       0.5*sqrt_2, 0.0,       0.5*sqrt_2 ],
                    [ 0.0,       0.5*sqrt_2, 0.0,      -0.5*sqrt_2 ],
                    [ 0.0,       0.5*sqrt_2,-0.5*sqrt_2, 0.0       ],
                    [ 0.0,      -0.5*sqrt_2,-0.5*sqrt_2, 0.0       ],
                    [ 0.5,       0.5,       0.5,       0.5       ],
                    [-0.5,       0.5,       0.5,       0.5       ],
                    [-0.5,       0.5,       0.5,      -0.5       ],
                    [-0.5,       0.5,      -0.5,       0.5       ],
                    [-0.5,      -0.5,       0.5,       0.5       ],
                    [-0.5,      -0.5,       0.5,      -0.5       ],
                    [-0.5,      -0.5,      -0.5,       0.5       ],
                    [-0.5,       0.5,      -0.5,      -0.5       ],
                    [-0.5*sqrt_2, 0.0,       0.0,       0.5*sqrt_2 ],
                    [ 0.5*sqrt_2, 0.0,       0.0,       0.5*sqrt_2 ],
                    [-0.5*sqrt_2, 0.0,       0.5*sqrt_2, 0.0       ],
                    [-0.5*sqrt_2, 0.0,      -0.5*sqrt_2, 0.0       ],
                    [-0.5*sqrt_2, 0.5*sqrt_2, 0.0,       0.0       ],
                    [-0.5*sqrt_2,-0.5*sqrt_2, 0.0,       0.0       ],
                   ]
    elif lattice in lattice_hcp:
        symQuats =  [
                     [ 1.0,        0.0,       0.0,        0.0       ],
                     [-0.5*sqrt_3,  0.0,       0.0,       -0.5       ],
                     [ 0.5,        0.0,       0.0,        0.5*sqrt_3 ],
                     [ 0.0,        0.0,       0.0,        1.0       ],
                     [-0.5,        0.0,       0.0,        0.5*sqrt_3 ],
                     [-0.5*sqrt_3,  0.0,       0.0,        0.5       ],
                     [ 0.0,        1.0,       0.0,        0.0       ],
                     [ 0.0,       -0.5*sqrt_3, 0.5,        0.0       ],
                     [ 0.0,        0.5,      -0.5*sqrt_3,  0.0       ],
                     [ 0.0,        0.0,       1.0,        0.0       ],
                     [ 0.0,       -0.5,      -0.5*sqrt_3,  0.0       ],
                     [ 0.0,        0.5*sqrt_3, 0.5,        0.0       ],
                    ]
    elif lattice in lattice_tet:
        symQuats =  [
                     [ 1.0,       0.0,       0.0,       0.0       ],
                     [ 0.0,       1.0,       0.0,       0.0       ],
                     [ 0.0,       0.0,       1.0,       0.0       ],
                     [ 0.0,       0.0,       0.0,       1.0       ],
                     [ 0.0,       0.5*sqrt_2, 0.5*sqrt_2, 0.0       ],
                     [ 0.0,      -0.5*sqrt_2, 0.5*sqrt_2, 0.0       ],
                     [ 0.5*sqrt_2, 0.0,       0.0,       0.5*sqrt_2 ],
                     [-0.5*sqrt_2, 0.0,       0.0,       0.5*sqrt_2 ],
                    ]
    elif lattice in lattice_orth:
        symQuats =  [
                     [ 1.0,0.0,0.0,0.0 ],
                     [ 0.0,1.0,0.0,0.0 ],
                     [ 0.0,0.0,1.0,0.0 ],
                     [ 0.0,0.0,0.0,1.0 ],
                    ]
    elif lattice in lattice_tric:
        symQuats =  [
                     [ 1.0,0.0,0.0,0.0 ],
                    ]
    else:
        raise ValueError("Unknown lattice structure: {}".format(lattice))

    mode = mode.lower()
    if mode == 'numpy':
        symQuats = np.array(symQuats)
    elif mode == 'quaternion':
        symQuats = [Quaternion(np.array(item)) for item in symQuats]
    else:
        raise ValueError("Unknown mode: {}".format(mode))

    return symQuats


def slip_systems(str lattice):
    """
    DESCRIPTION
    -----------
    ss = slipSystems('latticeStructure')
        Return slip systems for given lattice structure
    PARAMETERS
    ----------
    lattice: str
        Name of the lattice structure ['hexagonal', 'hcp', 'bcc', 'fcc',
                                       'hexagonal', 'hcp+' ]
    RETURNS
    -------
    ss: np.ndarray
        Multi-dimensional arrays
            [[slipDirection1, slipPlane1],
              slipDirection2, slipPlane2],
              ...]
    """
    cdef np.ndarray ss = np.zeros((50,2,4), dtype=DTYPE)
    cdef int slicer

    lattice = lattice.lower()

    if lattice in ['hcp+', 'hexagonal+']:
        # Basal slip <1120>{0001}
        ss[ 0, :, :] = np.array([[ 2,-1,-1, 0], [ 0, 0, 0, 1]])
        ss[ 1, :, :] = np.array([[-1, 2,-1, 0], [ 0, 0, 0, 1]])
        ss[ 2, :, :] = np.array([[-1,-1, 2, 0], [ 0, 0, 0, 1]])
        # Prism Slip <1120>{1010}
        ss[ 3, :, :] = np.array([[ 2,-1,-1, 0], [ 0, 1,-1, 0]])
        ss[ 4, :, :] = np.array([[-1, 2,-1, 0], [-1, 0, 1, 0]])
        ss[ 5, :, :] = np.array([[-1,-1, 2, 0], [ 1,-1, 0, 0]])
        # 2nd Prism Slip <1010>{1120}
        ss[ 6, :, :] = np.array([[ 0, 1,-1, 0], [ 2,-1,-1, 0]])
        ss[ 7, :, :] = np.array([[-1, 0, 1, 0], [-1, 2,-1, 0]])
        ss[ 8, :, :] = np.array([[ 1,-1, 0, 0], [-1,-1, 2, 0]])
        # Pyramidal a Slip <1120>{1011}
        ss[ 9, :, :] = np.array([[ 2,-1,-1, 0], [ 0, 1,-1, 1]])
        ss[10, :, :] = np.array([[-1, 2,-1, 0], [-1, 0, 1, 1]])
        ss[11, :, :] = np.array([[-1,-1, 2, 0], [ 1,-1, 0, 1]])
        ss[12, :, :] = np.array([[ 1, 1,-2, 0], [-1, 1, 0, 1]])
        ss[13, :, :] = np.array([[-2, 1, 1, 0], [ 0,-1, 1, 1]])
        ss[14, :, :] = np.array([[ 1,-2, 1, 0], [ 1, 0,-1, 1]])
        # Pyramidal c+a Slip <2113>{1011}
        ss[15, :, :] = np.array([[ 2,-1,-1, 3], [-1, 1, 0, 1]])
        ss[16, :, :] = np.array([[ 1,-2, 1, 3], [-1, 1, 0, 1]])
        ss[17, :, :] = np.array([[-1,-1, 2, 3], [ 1, 0,-1, 1]])
        ss[18, :, :] = np.array([[-2, 1, 1, 3], [ 1, 0,-1, 1]])
        ss[19, :, :] = np.array([[-1, 2,-1, 3], [ 0,-1, 1, 1]])
        ss[20, :, :] = np.array([[ 1, 1,-2, 3], [ 0,-1, 1, 1]])
        ss[21, :, :] = np.array([[-2, 1, 1, 3], [ 1,-1, 0, 1]])
        ss[22, :, :] = np.array([[-1, 2,-1, 3], [ 1,-1, 0, 1]])
        ss[23, :, :] = np.array([[ 1, 1,-2, 3], [-1, 0, 1, 1]])
        ss[24, :, :] = np.array([[ 2,-1,-1, 3], [-1, 0, 1, 1]])
        ss[25, :, :] = np.array([[ 1,-2, 1, 3], [ 0, 1,-1, 1]])
        ss[26, :, :] = np.array([[-1,-1, 2, 3], [ 0, 1,-1, 1]])
        # Set slicer
        slicer = 27
    elif lattice in ['hcp', 'hexagonal']:
        # Basal slip <1120>{0001}
        ss[ 0, :, :] = np.array([[ 2,-1,-1, 0], [ 0, 0, 0, 1]])
        ss[ 1, :, :] = np.array([[-1, 2,-1, 0], [ 0, 0, 0, 1]])
        ss[ 2, :, :] = np.array([[-1,-1, 2, 0], [ 0, 0, 0, 1]])
        # Prism Slip <1120>{1010}
        ss[ 3, :, :] = np.array([[ 2,-1,-1, 0], [ 0, 1,-1, 0]])
        ss[ 4, :, :] = np.array([[-1, 2,-1, 0], [-1, 0, 1, 0]])
        ss[ 5, :, :] = np.array([[-1,-1, 2, 0], [ 1,-1, 0, 0]])
        # Pyramidal a Slip <1120>{1011}
        ss[ 6, :, :] = np.array([[ 2,-1,-1, 0], [ 0, 1,-1, 1]])
        ss[ 7, :, :] = np.array([[-1, 2,-1, 0], [-1, 0, 1, 1]])
        ss[ 8, :, :] = np.array([[-1,-1, 2, 0], [ 1,-1, 0, 1]])
        ss[ 9, :, :] = np.array([[ 1, 1,-2, 0], [-1, 1, 0, 1]])
        ss[10, :, :] = np.array([[-2, 1, 1, 0], [ 0,-1, 1, 1]])
        ss[11, :, :] = np.array([[ 1,-2, 1, 0], [ 1, 0,-1, 1]])
        # Pyramidal c+a Slip <2113>{1011}
        ss[12, :, :] = np.array([[ 2,-1,-1, 3], [-1, 1, 0, 1]])
        ss[13, :, :] = np.array([[ 1,-2, 1, 3], [-1, 1, 0, 1]])
        ss[14, :, :] = np.array([[-1,-1, 2, 3], [ 1, 0,-1, 1]])
        ss[15, :, :] = np.array([[-2, 1, 1, 3], [ 1, 0,-1, 1]])
        ss[16, :, :] = np.array([[-1, 2,-1, 3], [ 0,-1, 1, 1]])
        ss[17, :, :] = np.array([[ 1, 1,-2, 3], [ 0,-1, 1, 1]])
        ss[18, :, :] = np.array([[-2, 1, 1, 3], [ 1,-1, 0, 1]])
        ss[19, :, :] = np.array([[-1, 2,-1, 3], [ 1,-1, 0, 1]])
        ss[20, :, :] = np.array([[ 1, 1,-2, 3], [-1, 0, 1, 1]])
        ss[21, :, :] = np.array([[ 2,-1,-1, 3], [-1, 0, 1, 1]])
        ss[22, :, :] = np.array([[ 1,-2, 1, 3], [ 0, 1,-1, 1]])
        ss[23, :, :] = np.array([[-1,-1, 2, 3], [ 0, 1,-1, 1]])
        # Set slicer
        slicer = 24
    elif lattice in ['bcc']:
        # Slip system <111>{110}
        ss[ 0, :, :] = np.array([[ 1,-1, 1], [ 0, 1, 1]])
        ss[ 1, :, :] = np.array([[-1,-1, 1], [ 0, 1, 1]])
        ss[ 2, :, :] = np.array([[ 1, 1, 1], [ 0,-1, 1]])
        ss[ 3, :, :] = np.array([[-1, 1, 1], [ 0,-1, 1]])
        ss[ 4, :, :] = np.array([[-1, 1, 1], [ 1, 0, 1]])
        ss[ 5, :, :] = np.array([[-1,-1, 1], [ 1, 0, 1]])
        ss[ 6, :, :] = np.array([[ 1, 1, 1], [-1, 0, 1]])
        ss[ 7, :, :] = np.array([[ 1,-1, 1], [-1, 0, 1]])
        ss[ 8, :, :] = np.array([[-1, 1, 1], [ 1, 1, 0]])
        ss[ 9, :, :] = np.array([[-1, 1,-1], [ 1, 1, 0]])
        ss[10, :, :] = np.array([[ 1, 1, 1], [-1, 1, 0]])
        ss[11, :, :] = np.array([[ 1, 1,-1], [-1, 1, 0]])
        # Slip system <111>{112}
        ss[12, :, :] = np.array([[-1, 1, 1], [ 2, 1, 1]])
        ss[13, :, :] = np.array([[ 1, 1, 1], [-2, 1, 1]])
        ss[14, :, :] = np.array([[ 1, 1,-1], [ 2,-1, 1]])
        ss[15, :, :] = np.array([[ 1,-1, 1], [ 2, 1,-1]])
        ss[16, :, :] = np.array([[ 1,-1, 1], [ 1, 2, 1]])
        ss[17, :, :] = np.array([[ 1, 1,-1], [-1, 2, 1]])
        ss[18, :, :] = np.array([[ 1, 1, 1], [ 1,-2, 1]])
        ss[19, :, :] = np.array([[-1, 1, 1], [ 1, 2,-1]])
        ss[20, :, :] = np.array([[ 1, 1,-1], [ 1, 1, 2]])
        ss[21, :, :] = np.array([[ 1,-1, 1], [-1, 1, 2]])
        ss[22, :, :] = np.array([[-1, 1, 1], [ 1,-1, 2]])
        ss[23, :, :] = np.array([[ 1, 1, 1], [ 1, 1,-2]])
        # Set slicer
        slicer = 24
    elif lattice in ['fcc']:
        ss[ 0, :, :] = np.array([[ 0, 1,-1], [ 1, 1, 1]])
        ss[ 1, :, :] = np.array([[-1, 0, 1], [ 1, 1, 1]])
        ss[ 2, :, :] = np.array([[ 1,-1, 0], [ 1, 1, 1]])
        ss[ 3, :, :] = np.array([[ 0,-1,-1], [-1,-1, 1]])
        ss[ 4, :, :] = np.array([[ 1, 0, 1], [-1,-1, 1]])
        ss[ 5, :, :] = np.array([[-1, 1, 0], [-1,-1, 1]])
        ss[ 6, :, :] = np.array([[ 0,-1, 1], [ 1,-1,-1]])
        ss[ 7, :, :] = np.array([[-1, 0,-1], [ 1,-1,-1]])
        ss[ 8, :, :] = np.array([[ 1, 1, 0], [ 1,-1,-1]])
        ss[ 9, :, :] = np.array([[ 0, 1, 1], [-1, 1,-1]])
        ss[10, :, :] = np.array([[ 1, 0,-1], [-1, 1,-1]])
        ss[11, :, :] = np.array([[-1,-1, 0], [-1, 1,-1]])
    else:
        raise ValueError("Unknown lattice structure: {}".format(lattice))

    return np.copy(ss[0:slicer,:,:])


def bravais2cartesian(m_bravais, n_bravais, covera):
    """
    DESCRIPTION
    -----------
    m_carteisna, n_cartesian = bravais2cartesian(m_bravais, n_bravais)
        Using a 3x4 matrix that can convert both slip direction and plane normal
    from Bravis-Miller to standard Miller indices.
    PARAMETERS
    ----------
    m_bravais: numpy.ndarray, 1x4
        Slip direction
    n_bravais: numpy.ndarray, 1x4
        Plane normal
    covera: float
        c/a ratio, for Ti it is 1.58.
    RETURNS
    -------
    m_carteisna: numpy.ndarray, 1x3
        Slip direction in Cartesian coordinate system.
    n_cartesian: numpy.ndarray, 1x3
        Plane normal in Cartesian coordinate system.
    NOTE
    ----
    1. The returned slip direction and plane normal are normalized
    """
    cdef converter = np.array([[1, 0, 0, 0],
                               [1/np.sqrt(3), 2/np.sqrt(3), 0, 0],
                               [0, 0, 0, 1/covera]])
    cdef m_carteisna = np.empty(3, dtype=DTYPE)
    cdef n_cartesian = np.empty(3, dtype=DTYPE)

    # converting to Cartesian
    m_carteisna = np.dot(converter, m_bravais)
    n_cartesian = np.dot(converter, n_bravais)
    # normalization
    m_carteisna = m_carteisna / np.linalg.norm(m_carteisna)
    n_cartesian = n_cartesian / np.linalg.norm(n_cartesian)
    return m_carteisna, n_cartesian


#-------------#
# END OF FILE #
#-------------#