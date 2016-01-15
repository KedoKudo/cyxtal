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

    __slots__ = ['_Xsample', '_Ysample', '_Zsample', '_depth',
                 '_qs', '_hkls',
                 '_astar', '_bstar', '_castar',
                 '_lc', '_lattice',
                 '_valid']

    def __init__(self):
        # coordinates
        self._Xsample = None
        self._Ysample = None
        self._Zsample = None
        self._depth   = None
        # indexing (shape unknown)
        self._qs      = None
        self._hkls    = None
        # strain free reciprocal lattice
        self._astar   = None
        self._bstar   = None
        self._cstar   = None
        # lattice constant
        self._lc      = None
        self._lattice = None
        # validation
        self._valid  = False

    # Not the best way to handle data access constrain, but
    # we have to settle with this for now. No specific doc
    # is necessary for this section.
    @property
    def Xsample(self):
        return self._Xsample

    @Xsample.setter
    def Xsample(self, data):
        self._Xsample = data

    @property
    def Ysample(self):
        return self._Ysample

    @Ysample.setter
    def Ysample(self, data):
        self._Ysample = data

    @property
    def Zsample(self):
        return self._Zsample

    @Zsample.setter
    def Zsample(self, data):
        self._Zsample = data

    # Validate data to make sure we got all the fields
    # we need from the DAXM data file. Sometime the results file
    # can be corrupted such that only part of the data is available.
    # In this case, we have no choice but to mark the affect voxel as
    # corrupted and discard it from the calculation.
    def validate(self, skip=False):
        """
        DESCRIPTION
        -----------
        self.validate()
            Validate all parameters are parsed;
            Prune q vectors, ensure correct mapping between
            self.hkls and self.qs;
            Instance of VoxelStep can only be used when validated.
        """
        # allow bypass the security check
        if skip:
            self._valid = True
            return self._valid
        # type assert for all data


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
        if not(self._valid):
            msg = "Self validation failed, try self.validate()?."
            raise ValueError(msg)
        pass

    def get_eulers(self, ref='TSL'):
        """
        """
        if not(self._valid):
            raise ValueError("Invalid data, abort.")
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

    @static_method
    def get_bases(lc, lattice_structure='hcp'):
        """
        DESCRIPTION
        -----------
        PARAMETERS
        ----------
        RETURNS
        -------
        """
        lattice_structure = lattice_structure.lower()
        if lattice_structure == 'hcp':
            pass
        else:
            return NotImplemented


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