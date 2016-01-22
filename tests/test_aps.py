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
from cyxtal import VoxelStep

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
        lvs = get_base(self.lc1, reciprocal=False)
        rlvs = get_base(self.lc1)
        target_v = np.array([
                             [+0.2950800,  -0.1475400,  +0.0000000],
                             [+0.0000000,  +0.2555468,  +0.0000000],
                             [+0.0000000,  +0.0000000,  +0.4685500] ])
        target_rv = np.array([
                              [+21.293159,    +0.000000,   +0.000000],
                              [+12.293611,   +24.587222,   -0.000000],
                              [ -0.000000,    -0.000000,  +13.409850] ])
        # test real lattice
        np.testing.assert_almost_equal(lvs, target_v)
        # test reciprocal lattice
        np.testing.assert_almost_equal(rlvs, target_rv, decimal=6)

class TestStrainRefine(unittest.TestCase):
    """
    Testing xml step
<step xmlns="http://sector34.xor.aps.anl.gov/34ide:indexResult">
    <title>Ti</title>
    <sampleName>Victoria</sampleName>
    <userName>Bieler</userName>
    <beamline />
    <scanNum>12318</scanNum>
    <date>2012-08-23T07:52:57-06:00</date>
    <beamBad>0</beamBad>
    <CCDshutter>out</CCDshutter>
    <lightOn>0</lightOn>
    <monoMode>white slitted</monoMode>
    <Xsample>616.0</Xsample>
    <Ysample>-3257.7</Ysample>
    <Zsample>-408.0</Zsample>
    <depth>-29.0</depth>
    <energy unit="keV">14.9995</energy>
    <hutchTemperature>22.8056</hutchTemperature>
    <sampleDistance>0.0</sampleDistance>
    <detector>
        <inputImage>/data34a/BielerAug2012/Victoria/Victoria_H3-recon/Victoria_H_37171_1.h5</inputImage>
        <detectorID>PE1621 723-3335</detectorID>
        <exposure unit="sec">0.5</exposure>
        <Nx>2048</Nx>
        <Ny>2048</Ny>
        <totalSum>39184200.0</totalSum>
        <sumAboveThreshold>29647.5</sumAboveThreshold>
        <numAboveThreshold>465.0</numAboveThreshold>
        <geoFile>/data34a/BielerAug2012/geoN_2012-07-11_15-12-01.txt</geoFile>
        <ROI endx="2047" endy="2047" groupx="1" groupy="1" startx="0" starty="0" />
        <peaksXY Npeaks="20" boxsize="10" executionTime="2.45" maxCentToFit="10.0" maxRfactor="0.5" max_number="50" maxwidth="15.0" min_separation="20" minwidth="0.25" peakProgram="peaksearch" peakShape="Gaussian" smooth="True" threshold="50.0" thresholdRatio="-1">
            <Xpixel>835.057 1704.854 1564.614 1228.783 1321.293 654.085 543.274 1292.355 1669.4 1593.863 1367.624 1637.085 1696.329 1609.955 1522.339 764.536 1653.076 1515.627 1109.609 1097.977</Xpixel>
            <Ypixel>1088.925 461.925 1025.009 805.767 1082.892 1664.212 1538.215 1951.519 1188.092 200.817 1457.228 2008.896 776.231 1044.247 1035.816 627.941 838.989 1687.486 769.95 1289.832</Ypixel>
            <Intens>281.7841 204.7827 96.25 74.3251 69.8411 69.1358 66.628 65.4453 64.7486 62.9118 62.5197 62.3724 61.2166 60.3871 59.9176 55.8342 55.7814 54.9429 54.5022 52.0358</Intens>
            <Integral>2.77586 1.6806 6.37156 2.80552 1.68294 0.94962 0.98782 2.57575 1.44614 2.57356 1.68811 2.56151 3.2453 2.64896 2.11547 0.54717 2.54636 1.94845 1.23716 1.50356</Integral>
            <hwhmX unit="pixel">1.182 1.339 6.567 3.385 1.332 1.465 1.032 2.054 1.437 8.528 2.949 8.177 4.967 11.098 4.017 1.161 2.723 5.842 1.707 3.496</hwhmX>
            <hwhmY unit="pixel">1.287 1.224 3.699 2.034 1.972 1.081 1.205 2.555 2.754 4.214 1.919 5.313 4.428 10.635 7.808 1.877 1.853 3.624 2.031 1.667</hwhmY>
            <tilt unit="degree">9.3354 61.2896 165.376 155.7191 89.7256 134.6433 176.0934 109.6243 28.3401 178.2944 171.3658 164.8854 41.5873 172.7367 60.0173 23.425 129.8213 149.6835 173.1839 161.4556</tilt>
            <chisq>0.072897 0.12429 0.066823 0.089246 0.061012 0.094156 0.087014 0.059364 0.096183 0.045748 0.068661 0.057868 0.050314 0.064958 0.060817 0.11368 0.06579 0.057707 0.044263 0.089673</chisq>
            <Qx>0.0140348 -0.1796392 -0.0060059 -0.069112 0.012206 0.1614276 0.1275091 0.2547204 0.0449557 -0.2484809 0.1234327 0.2815161 -0.084668 -0.0001569 -0.0025131 -0.1090872 -0.0646075 0.1921886 -0.0775609 0.0716437</Qx>
            <Qy>0.7004301 0.7853091 0.7931083 0.7494709 0.764578 0.6571017 0.648113 0.7150724 0.8033227 0.7515843 0.7600083 0.7453083 0.8022366 0.7981589 0.7883831 0.6816235 0.7997547 0.7622336 0.7331666 0.7328844</Qy>
            <Qz>-0.713583 -0.5924689 -0.609051 -0.65842 -0.6444157 -0.7363141 -0.7507936 -0.6509908 -0.5938447 -0.6110468 -0.6380844 -0.6043709 -0.5909719 -0.6024469 -0.6151795 -0.7235257 -0.5968403 -0.6181128 -0.6756116 -0.6765705</Qz>
        </peaksXY>
    </detector>
    <indexing Nindexed="5" Npatterns="1" Npeaks="20" angleTolerance="0.1" cone="91.0" executionTime="0.46" hklPrefer="1 0 0" indexProgram="euler" keVmaxCalc="18.0" keVmaxTest="30.0">

        <pattern Nindexed="5" goodness="56.7207" num="0" rms_error="0.044">
            <recip_lattice unit="1/nm">
                <astar>-18.1537190 15.2611391 6.0245343</astar>
                <bstar>-6.0619130 18.3373843 -15.0249274</bstar>
                <cstar>-8.6729759 -7.8946099 -6.1359146</cstar>
            </recip_lattice>
            <hkl_s>
                <h>-2 -2 -2 -1 -1</h>
                <k>5 6 7 6 5</k>
                <l>-3 -3 -2 -1 0</l>
            </hkl_s>
        </pattern>
        <xtl>
            <structureDesc>Ti crystal (Bieler 525, hexagnol axis)</structureDesc>
            <SpaceGroup>194</SpaceGroup>
            <latticeParameters unit="nm">0.2965 0.2965 0.4747 90 90 120</latticeParameters>
           <atom Z="29" label="Cu001" n="1" occ="1" symbol="Cu">0.33333 0.66667 0.75</atom>
           <atom Z="29" label="Cu001" n="2" occ="1" symbol="Cu">0.66667 0.33333 0.25</atom>
        </xtl>
    </indexing>
</step>
    """

    def setUp(self):
        tmp = VoxelStep()
        # some dummy position value
        # (not important for strain refinement)
        tmp.Xsample = 616.0
        tmp.Ysample = -3257.7
        tmp.Zsample = -408.0
        tmp.depth   = 29.0
        # set q vectors
        qx = [0.0140348, -0.1796392, -0.0060059, -0.069112,
              0.012206,   0.1614276,  0.1275091,  0.2547204,
              0.0449557, -0.2484809,  0.1234327,  0.2815161,
             -0.084668,  -0.0001569, -0.0025131, -0.1090872,
             -0.0646075,  0.1921886, -0.0775609,  0.0716437]
        qy = [0.7004301,  0.7853091,  0.7931083,  0.7494709,
              0.764578,   0.6571017,  0.648113,   0.7150724,
              0.8033227,  0.7515843,  0.7600083,  0.7453083,
              0.8022366,  0.7981589,  0.7883831,  0.6816235,
              0.7997547,  0.7622336,  0.7331666,  0.7328844]
        qz = [-0.713583, -0.5924689, -0.609051,  -0.65842,
             -0.6444157, -0.7363141, -0.7507936, -0.6509908,
             -0.5938447, -0.6110468, -0.6380844, -0.6043709,
             -0.5909719, -0.6024469, -0.6151795, -0.7235257,
             -0.5968403, -0.6181128, -0.6756116, -0.6765705]
        tmp.qs = np.column_stack((qx,qy,qz))
        # set hkl
        h = [-2, -2, -2, -1, -1]
        k = [ 5,  6,  7,  6,  5]
        l = [-3, -3, -2, -1,  0]
        tmp.hkls = np.column_stack((h,k,l))
        # set reciprocal lattice
        tmp.astar = [-18.1537190, 15.2611391,   6.0245343]
        tmp.bstar = [-6.0619130,  18.3373843, -15.0249274]
        tmp.cstar = [-8.6729759,  -7.8946099,  -6.1359146]
        # set lattice constant
        # (use HCP as example)
        tmp.lc = [0.29508, 0.29508, 0.46855, 90, 90, 120]
        # validate this voxel
        tmp.validate()
        self.data = tmp

    def test_strainRefine(self):
        # set target values
        epsilonAPS100111 = np.array([
               [ 0.00120322, -0.00592183, -0.0101301 ],
               [-0.00592183,  0.00843847,  0.00431117],
               [-0.0101301 ,  0.00431117, -0.00964169]])
        epsilonAPS110111 = np.array([
               [ 0.0013487 , -0.0130564 ,  -0.00731337],
               [-0.0130564 ,  0.0360299 ,   0.00756606],
               [-0.00731337,  0.00756606,  -0.0373786]])
        # perform strain refine
        epsilon_aps = self.data.get_strain(ref='APS')
        np.testing.assert_almost_equal(epsilonAPS100111, epsilon_aps)
        np.testing.assert_almost_equal(epsilonAPS110111, epsilon_aps)

if __name__ == '__main__':
    unittest.main()