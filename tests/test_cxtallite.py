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
    pass

#     def setUp(self):
#         q1 = Quaternion()
#         self.data = []

#     def test_add(self):
#         pass

#     def test_rotation(self):
#         pass

#     def test_convert2Eulers(self):
#         pass

#     def test_convert2gMatrix(self):
#         pass

#     def test_avergateOrientation(self):
#         pass



if __name__ == '__main__':
    unittest.main()