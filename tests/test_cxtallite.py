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
from cyxtal.cxtallite import Quaternion
from cyxtal.cxtallite import symmetry
from cyxtal.cxtallite import Xtallite

# void random seed for testing
np.random.seed(1960)


class testSymmetry(unittest.TestCase):

    def test_hcp(self):
        sym1 = symmetry('hexagonal')
        sym2 = symmetry('hcp')
        sym3 = symmetry('hex')
        self.assertListEqual(sym1.tolist(), sym2.tolist())
        self.assertListEqual(sym2.tolist(), sym3.tolist())

    def test_cubic(self):
        sym1 = symmetry('bcc')
        sym2 = symmetry('fcc')
        sym3 = symmetry('cubic')
        self.assertListEqual(sym1.tolist(), sym2.tolist())
        self.assertListEqual(sym2.tolist(), sym3.tolist())

    def test_unknownStructure(self):
        self.assertRaises(ValueError,
                          symmetry, 'Peter Parker')


class testQuaternion(unittest.TestCase):

    def setUp(self):

        self.q1 = Quaternion(np.array([-1.0, 0.0, 0.0, 0.0]))
        self.q2 = Quaternion(np.array([ 1.0, 2.0, 3.0, 4.0]))
        self.q3 = Quaternion(np.array([ 3.0, 7.0, 9.0, 2.0]))
        self.q4 = Quaternion(np.array([ 1.0, 2.0,-5.0, 1.0]))
        self.q5 = Quaternion(np.array([ 0.7071, 0, 0, 0.7071]))

        self.u1 = self.q1.unitary()
        self.u2 = self.q2.unitary()
        self.u3 = self.q3.unitary()
        self.u4 = self.q4.unitary()
        self.u5 = self.q5.unitary()

        self.p1 = np.array([1.0,0.0,0.0])

    def test_add(self):
        calc1   = self.q1 + self.q2
        target1 = [2.0, 2.0, 3.0, 4.0]
        np.testing.assert_almost_equal(calc1.tolist(), target1)

    def test_sub(self):
        calc1   = self.q1 - self.q2
        target1 = [0.0, -2.0, -3.0, -4.0]
        np.testing.assert_almost_equal(calc1.tolist(), target1)

    def test_mul(self):
        calc1   = self.u1 * self.u2
        np.testing.assert_almost_equal(calc1.tolist(), self.u2.tolist())

        calc2   = self.u2 * self.u4
        target2 = [0.2623303, 0.8853649, 0.1311652, -0.3607042]
        np.testing.assert_almost_equal(calc2.tolist(), target2)

    def test_rotate(self):
        """
        Rotate [1,0,0] around z-axis by 90 degree
        """
        calc1   = Quaternion.rotate(self.u5, self.p1)
        target  = [0.0, 1.0, 0.0]
        np.testing.assert_almost_equal(calc1, target)

    def test_div(self):
        calc1   = self.q2 / self.q3
        target1 = [52, 29, -24, 13]
        np.testing.assert_almost_equal(calc1.tolist(), target1)

        calc2   = self.u2 / self.u3
        target2 = [0.7939163, 0.4427610, -0.3664228, 0.1984791]
        np.testing.assert_almost_equal(calc2.tolist(), target2)

    def test_scale(self):
        calc1  = Quaternion.scale(self.q3, 5)
        target = np.array([3.0, 7.0, 9.0, 2.0]) * 5
        np.testing.assert_almost_equal(calc1.tondarray(), target)

    def test_2Eulers(self):
        e1 = np.array([  -0.,            0.,            0.        ])
        e2 = np.array([ 132.27368901,   82.33774434,   19.65382406])
        e3 = np.array([  85.81508387,  144.90319877,  341.56505118])
        np.testing.assert_almost_equal(self.q1.toEulers(), e1)
        np.testing.assert_almost_equal(self.q2.toEulers(), e2)
        np.testing.assert_almost_equal(self.q3.toEulers(), e3)

    def test_2Rodrigues(self):
        r1 = np.array([-0., -0., -0.])
        r2 = np.array([ 2.,  3.,  4.])
        r3 = np.array([ 2.33333333,  3.        ,  0.66666667])
        np.testing.assert_almost_equal(self.q1.toRodrigues(), r1)
        np.testing.assert_almost_equal(self.q2.toRodrigues(), r2)
        np.testing.assert_almost_equal(self.q3.toRodrigues(), r3)

    def test_2OrientationMatrix(self):
        pass

    def test_2AngleAxis(self):
        a1, v1 = (0.0,                np.array([ 1.,  0.,  0.]))
        a2, v2 = (2.774384633031956,  np.array([ 0.37139068,  0.55708601,  0.74278135]))
        a3, v3 = (2.6344294922932665, np.array([ 0.6047079 ,  0.77748158,  0.17277369]))
        mya1, myv1 = self.q1.toAngleAxis()
        mya2, myv2 = self.q2.toAngleAxis()
        mya3, myv3 = self.q3.toAngleAxis()

        np.testing.assert_almost_equal(a1, mya1)
        np.testing.assert_almost_equal(v1, myv1)
        np.testing.assert_almost_equal(a2, mya2)
        np.testing.assert_almost_equal(v2, myv2)
        np.testing.assert_almost_equal(a3, mya3)
        np.testing.assert_almost_equal(v3, myv3)


class testEulers(unittest.TestCase):
    pass


class testOrienationMatrix(unittest.TestCase):
    pass


class testRodrigues(unittest.TestCase):
    pass


class testOrientations(unittest.TestCase):

    def setUp(self):
        pass

    def test_fromQuaternions(self):
        pass

    def test_fromEulers(self):
        pass

    def test_fromRodrigues(self):
        pass

    def test_fromMatrices(self):
        pass


class testAverageOrientations(unittest.TestCase):

    def setUp(self):
        pass

    def test_avgQuaternions(self):
        pass

    def test_avgEulers(self):
        pass

    def test_avgRogrigues(self):
        pass

    def test_avgOrientationMatrix(self):
        pass


if __name__ == '__main__':
    unittest.main()