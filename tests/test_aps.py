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

"""

import unittest
import numpy as np
from cyxtal import get_base

# void random seed for testing
np.random.seed(1960)

class TestBase(unittest.TestCase):

    def setUp(self):
        # generate several different set of lattice constants
        self.lc1 = np.array([0.29508, 0.29508, 0.46855, 90.0, 90.0, 120.0])
        self.lc2 = np.array([1.0, 1.0, 1.0, 90.0, 90.0, 90.0])
        self.lc3 = np.array([0.2965, 0.2965, 0.4747, 90, 90, 120])

    def test_bcc(self):
        lvs = get_base(self.lc2)
        target = np.eye(3)*2*np.pi
        np.testing.assert_almost_equal(lvs, target)
        rlvs = get_base(self.lc2, reciprocal=True)
        np.testing.assert_almost_equal(rlvs, target)

    def test_hcp(self):
        rlvs = get_base(self.lc1)
        target = np.array([[21.29,   0.0,   0.0],
                           [12.29, 24.58,   0.0],
                           [  0.0,   0.0, 13.41]])
        np.testing.assert_almost_equal(rlvs, target, decimal=2)