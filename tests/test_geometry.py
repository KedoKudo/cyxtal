#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-

"""
   ________  ___  ___________    __
  / ____/\ \/ / |/ /_  __/   |  / /
 / /      \  /|   / / / / /| | / /
/ /___    / //   | / / / ___ |/ /___
\____/   /_//_/|_|/_/ /_/  |_/_____/

Copyright (c) 2016, C. Zhang.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification are permitted provided that the following conditions are met:

1) Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2) Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

DESCRIPTION
-----------
Unit testing script for cyxtal.geometry.
"""

import unittest
import numpy as np

from cyxtal.geometry import Point
from cyxtal.geometry import Line
from cyxtal.geometry import Plane
from cyxtal.geometry import Point2D
from cyxtal.geometry import Line2D
from cyxtal.geometry import Polygon2D
from cyxtal.geometry import convex_hull

# void random seed for testing
np.random.seed(1960)


class test3DGeo(unittest.TestCase):

    def setUp(self):
        self.pt1 = Point(0, 0, 0)
        self.pt2 = Point(2, 0, 0)
        self.pt3 = Point(0, 3, 0)
        self.pt4 = Point(0, 0, 4)
        self.pt5 = Point(2, 3, 4)

        self.line12 = Line(self.pt1, self.pt2)
        self.line13 = Line(self.pt1, self.pt3)
        self.line14 = Line(self.pt1, self.pt4)
        self.line25 = Line(self.pt2, self.pt5)

        self.plane123 = Plane(self.pt1, self.pt2, self.pt3)
        self.plane134 = Plane(self.pt1, self.pt3, self.pt4)
        self.plane234 = Plane(self.pt2, self.pt3, self.pt4)

    def test_dist_point2point(self):
        dist12 = self.pt1.dist2point(self.pt2)
        np.testing.assert_almost_equal(dist12, 2.0)
        dist25 = self.pt2.dist2point(self.pt5)
        np.testing.assert_almost_equal(dist25, 5.0)

    def test_dist_point2line(self):
        dist_pt2_line12 = self.pt2.dist2line(self.line12)
        np.testing.assert_almost_equal(dist_pt2_line12, 0.0)
        dist_pt2_line13 = self.pt2.dist2line(self.line13)
        np.testing.assert_almost_equal(dist_pt2_line13, 2.0)

    def test_point_on_line(self):
        np.testing.assert_almost_equal(self.pt1.on_line(self.line12),
                                       True)
        np.testing.assert_almost_equal(self.pt2.on_line(self.line13),
                                       False)

    def test_point_in_plane(self):
        np.testing.assert_almost_equal(Point(1, 2, 0).in_plane(self.plane123),
                                       True)
        np.testing.assert_almost_equal(Point(0, 5, 1).in_plane(self.plane123),
                                       False)

    def test_line_parallel(self):
        np.testing.assert_almost_equal(self.line13.parallel_to(self.line25),
                                       False)
        tmpLine = Line(self.pt1, Point(0, 3, 4))
        np.testing.assert_almost_equal(tmpLine.parallel_to(self.line25),
                                       True)

    def test_line_skewed(self):
        np.testing.assert_almost_equal(self.line13.skewed_from(self.line25),
                                       True)
        tmpLine = Line(self.pt1, Point(0, 3, 4))
        np.testing.assert_almost_equal(tmpLine.skewed_from(self.line25),
                                       False)


class test2DGeo(unittest.TestCase):

    def setUp(self):
        self.pt1 = Point2D(0, 0)

if __name__ == '__main__':
    unittest.main()
