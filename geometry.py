#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-

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
Standard geometry class for 2D/3D geometry calculation.
"""

import numpy as np


class Point(object):
    """
    DESCRIPTION
    -----------
    Point(x,y,z)
        Point in 3D space, base class provide bare bone abstraction
        for point related calculation.
    PARAMETERS
    ----------
    x,y,z: float
        Standard Cartesian coordinates for location description.
    METHODS
    -------
    dist2point(Point other)
        Return the distance to another instance of Point.
    dist2line(Line other)
        Return the distance to given instance of Line.
    on_line(Line other)
        Whether the current instance lies on a given instance of Line.
    in_plane(Plane other)
        Whether the current point lies in a given instance of Plane.
    CLASSMETHOD
    -----------
    """
    def __init__(self, x, y, z):
        self._coord = [x, y, z]

    @property
    def x(self):
        return self._coord[0]

    @x.setter
    def x(self, val):
        self._coord[0] = val

    @property
    def y(self):
        return self._coord[1]

    @y.setter
    def y(self, val):
        self._coord[1] = val

    @property
    def z(self):
        return self._coord[2]

    @z.setter
    def z(self, val):
        self._coord[2] = val

    @property
    def coord(self):
        return self._coord

    @coord.setter
    def coord(self, val):
        if len(val) != 3:
            raise ValueError("Need 3 coordinates")
        self._coord = val

    def __str__(self):
        """String representation of Point"""
        return "(" + str(self._coord)[1: -1] + ")"

    def __eq__(self, other):
        if np.absolute(self.x - other.x) < 1e-6:
            if np.absolute(self.y - other.y) < 1e-6:
                if np.absolute(self.z - other.z) < 1e-6:
                    return True
        return False

    def __ne__(self, other):
        return not self == other

    def __len__(self):
        return 3

    def dist2point(self, other):
        """Return the distance to another point"""
        distance = (self.x - other.x)**2 + \
                   (self.y - other.y)**2 + \
                   (self.z - other.z)**2
        return np.sqrt(distance)

    def dist2line(self, line):
        """Return the distance to another line"""
        return line.dist2point(self)

    def on_line(self, line):
        """Quick test is the point is on the given line"""
        return line.contain_point(self)

    def in_plane(self, plane):
        """Quick test if a point is in a given plane"""
        return plane.contain_point(self)


