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

import h5py  as h5
import numpy as np
import xml.etree.cElementTree as ET
from scipy.optimize import minimize

# module level constant
theta     = -0.75*np.pi
R_APS2TSL = np.array([[1.0,            0.0,            0.0],
                      [0.0,  np.cos(theta), -np.sin(theta)],
                      [0.0,  np.sin(theta),  np.cos(theta)]])


class VoxelStep(object):
    """
    DESCRIPTION
    -----------
    Container class for parsing through data, all the data is stored
    as it is in the xml file. Additional methods are provided for
    various other purposes.
    PARAMETERS
    ----------
    X|Y|Zsample:    sample motor position during scan (X|Y|Z)
    depth:          wire position
    qs:             identified diffraction vectors
    hkls:           hkl indices identified
    a|b|cstar:      strain free reciprocal lattice identified
    lc:             lattice constants used in indexation
    lattice:        lattice structure
    """

    def __init__(self):
        # coordinates
        self.Xsample = np.nan
        self.Ysample = np.nan
        self.Zsample = np.nan
        self.depth   = np.nan
        # indexing (shape unknown)
        self.qs      = np.nan
        self.hkls    = np.nan
        # strain free reciprocal lattice
        self.astar   = np.nan
        self.bstar   = np.nan
        self.cstar   = np.nan
        # lattice constant
        self.lc      = np.nan
        self.lattice = None
        # validation
        self._valid  = False

    def validate(self):
        """
        DESCRIPTION
        -----------
        self.validate()
            Validate all parameters are parsed;
            Prune q vectors, ensure correct mapping between
            self.hkls and self.qs;
            Instance of VoxelStep can only be used when validated.
        """
        pass

    def get_coord(self, ref='TSL'):
        """
        DESCRIPTION
        -----------
        coord = self.get_coord(ref='TSL')
            Return the coordinates of the voxel in given reference
            system
        PARAMETERS
        ----------
        ref: string(case insensitive)
            Name for reference configuration ['TSL'|'APS'|'XHF']
        """
        pass

    def get_eulers(self, ref='TSL'):
        """
        """
        pass

    def get_strain(self):
        """
        return strain tensor inferred with current data
        """
        strain = np.empty((3,3))
        # check if voxel data is valid
        if not(self._valid):
            print "Corrupted voxel found!"
            return strain.fill(np.nan)
        # continue on the strain refinement

    def strain_refine(self, new_lc):
        """
        DESCRIPTION
        -----------
        err = self.strain_refine(new_lc)
            Return the errors between calculated qs and measurements
        PARAMETERS
        ----------
        new_lc: lattice constant (perturbed)
        RETURNS
        -------
        err:    angular difference between calculated qs using new_lc
                and measurements (self.qs).
        """
        pass


def parser_xml(intput,
               output_mode='txt',
               ref_configuration='TSL',
               strain_refine=True):
    """
    DESCRIPTION
    -----------
    parser_xml(DAXM_DATA.xml,
               output='output.txt',
               ref_configuration='aps',
               strain_refine=True)
        Parse the DAXM data from Beamline 34-I-DE to simple ASCII table
    """
    pass