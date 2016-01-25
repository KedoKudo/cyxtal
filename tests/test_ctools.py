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
    TestKeamns:
        testing module of kmeans in cyxtal.ctools
"""

import unittest
import numpy as np
from cyxtal import kmeans
from cyxtal import get_vonMisesStress
from cyxtal import get_vonMisesStrain

# void random seed for testing
np.random.seed(1960)

class TestKmeans(unittest.TestCase):

    def setUp(self):
        # generate 1D points and reshape to 2D
        pt1 = np.random.normal(1, 0.2, (100,2))
        pt2 = np.random.normal(2, 0.5, (300,2))
        pt3 = np.random.normal(3, 0.3, (100,2))
        pt4 = np.random.normal(5, 0.5, (100,2))
        # slightly move sets 2 and 3 (for a prettier output)
        pt2[:,0] += 1
        pt3[:,0] -= 20
        pt4[:,0] += 10
        # for testing
        self.data      = np.concatenate((pt1, pt2, pt3, pt4))
        self.k         = 4
        self.max_iter  = 1e4
        self.threshold = 1e-2
        self.max_outer = 2**8

    def test_clustering(self):
        # get results and unpack it
        rst = kmeans(self.data,
                     self.k,
                     max_iter=self.max_iter,
                     threshold=self.threshold)
        # try multiple times to get better results
        for i in range(self.max_outer):
            tmp = kmeans(self.data,
                         self.k,
                         max_iter=self.max_iter,
                         threshold=self.threshold)
            if tmp[-1] < rst[-1]:
                rst = tmp

        # get final result
        assign = rst[0]
        centroid = rst[1]
        iteration = rst[2]
        converged = rst[3]
        cnt = rst[4]
        inertia = rst[5]

        # assert find right number of clusters
        self.assertEqual(len(cnt), self.k)


class TestGeneral(unittest.TestCase):

    def setUp(self):
        self.stress = np.array([[ 1, 2, 3],
                                [ 2, 4, 5],
                                [ 3, 5,-5]], dtype=np.float64)
        self.strain = np.array([[ 1, 2,-1],
                                [ 2, 0, 1],
                                [-1, 1,-1]], dtype=np.float64)

    def test_vonMisesStress(self):
        target = 13.3041347
        np.testing.assert_almost_equal(target,
                                       get_vonMisesStress(self.stress))

    def test_vonMisesStrain(self):
        target = 3.055050463
        np.testing.assert_almost_equal(target,
                                       get_vonMisesStrain(self.strain))

if __name__ == '__main__':
    unittest.main()