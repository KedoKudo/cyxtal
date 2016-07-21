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
from cyxtal import get_reciprocal_base
# from cyxtal import VoxelStep
# from cyxtal import get_vonMisesStrain
from cyxtal import parse_xml

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
        target = np.eye(3)
        np.testing.assert_almost_equal(lvs, target)
        target = np.eye(3)*2*np.pi
        rlvs = get_reciprocal_base(self.lc2)
        np.testing.assert_almost_equal(rlvs, target)

    def test_hcp(self):
        lvs = get_base(self.lc1)
        rlvs = get_reciprocal_base(self.lc1)
        target_v = np.array([
                             [+0.2950800,  -0.1475400,  +0.0000000],
                             [+0.0000000,  +0.2555468,  +0.0000000],
                             [+0.0000000,  +0.0000000,  +0.4685500]])
        target_rv = np.array([
                              [+21.293159,    +0.000000,   +0.000000],
                              [+12.293611,   +24.587222,   -0.000000],
                              [-0.000000,    -0.000000,  +13.409850]])
        # test real lattice
        np.testing.assert_almost_equal(lvs, target_v)
        # test reciprocal lattice
        np.testing.assert_almost_equal(rlvs, target_rv, decimal=6)


class TestStrainRefine(unittest.TestCase):
    """
    Using test.xml file as testing data.
    """

    def setUp(self):
        apsnspace = "http://sector34.xray.aps.anl.gov/34ide:indexResult"
        xmlFile = 'tests/test.xml'
        tmp = parse_xml(xmlFile,
                        namespace={'step': apsnspace})
        self.data = tmp[0]
        # e is extracted from Igor
        # --> use test.pxp to see the full process
        e = np.array([-0.001113,  0.000542, 0.000571,
                     0.001978, -0.001261, 0.000394])
        self.strain_igorAPS = np.array([e[0], e[5], e[4],
                                        e[5], e[1], e[3],
                                        e[4], e[3], e[2]])

    def test_strainRefine(self):
        # epsilon = U - I
        # epsilon_D = epsilon - 1./3*tr(epsilon)*I
        strain_tishler = self.data.get_strain(ref='APS',
                                              disp=True,
                                              deviatoric='tishler',
                                              xtor=1e-6,
                                              maxiter=1e6)
        np.testing.assert_almost_equal(self.strain_igorAPS,
                                       strain_tishler)
        # epsilon_D = U - J^(1./3)*I
        strain_m1 = self.data.get_strain(ref='APS',
                                         disp=True,
                                         deviatoric='m1',
                                         xtor=1e-6,
                                         maxiter=1e6)
        np.testing.assert_almost_equal(self.strain_igorAPS,
                                       strain_m1)
        # epsilon_D = 0.5*(U^2 - J^(2/3)I)
        strain_m2 = self.data.get_strain(ref='APS',
                                         disp=True,
                                         deviatoric='m2',
                                         xtor=1e-6,
                                         maxiter=1e6)
        np.testing.assert_almost_equal(self.strain_igorAPS,
                                       strain_m2)

if __name__ == '__main__':
    unittest.main()
