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
VoxelStep: class
    Container class to store voxel information and perform strain refinement
parser_xml: function
    Parsing xml output from APS and store them in different format
    (with/without strain refinement)
NOTE
----
More information regarding the coordinate transformation can be found at:
    http://www.aps.anl.gov/Sectors/33_34/microdiff/Instrument/coordinates-PE-system.pdf
"""

import h5py  as h5
import numpy as np
import xml.etree.cElementTree as ET
from scipy.optimize import minimize
from cyxtal.cxtallite import OrientationMatrix

##
# MODULE LEVEL CONSTANTS RELATING TO COORDINATE TRANSFORMATION
# <NOTE>
#   These are defined in terms of rotation matrices since it is
#   more intuitive to see how each system is connected through
#   simple rotation around x-axis (see cyxtal/documentation)
#
#** XHF <-> TSL
theta_1   = -np.pi
R_XHF2TSL = np.array([[1.0,              0.0,              0.0],
                      [0.0,  np.cos(theta_1), -np.sin(theta_1)],
                      [0.0,  np.sin(theta_1),  np.cos(theta_1)]])
R_TSL2XHF = R_XHF2TSL.T
#** XHF <-> APS
theta_2   = -0.25*np.pi
R_XHF2APS = np.array([[1.0,              0.0,              0.0],
                      [0.0,  np.cos(theta_2), -np.sin(theta_2)],
                      [0.0,  np.sin(theta_2),  np.cos(theta_2)]])
R_APS2XHF = R_XHF2APS.T
#** APS <-> TSL
theta_3   = -0.75*np.pi
R_APS2TSL = np.array([[1.0,              0.0,              0.0],
                      [0.0,  np.cos(theta_1), -np.sin(theta_1)],
                      [0.0,  np.sin(theta_1),  np.cos(theta_1)]])
R_TSL2APS = R_APS2TSL.T
#** self <-> self
R_TSL2TSL = R_APS2APS = R_XHF2XHF = np.eye(3)


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
        self._Xsample = float(data)

    @property
    def Ysample(self):
        return self._Ysample

    @Ysample.setter
    def Ysample(self, data):
        self._Ysample = float(data)

    @property
    def Zsample(self):
        return self._Zsample

    @Zsample.setter
    def Zsample(self, data):
        self._Zsample = float(data)

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, data):
        self._depth = float(data)

    @property
    def qs(self):
        return self._qs

    @qs.setter
    def qs(self, data):
        """
        DESCRIPTION
        -----------
        Q vectors much be stack in rows, the xml file from aps
        are storing Q vectors by column.
        """
        self._qs = np.array(data)

    @property
    def hkls(self):
        return self._hkls

    @hkls.setter
    def hkls(self, data):
        self._hkls = np.array(data)

    @property
    def astar(self):
        return self._astar

    @astar.setter
    def astar(self, data):
        self._astar = np.array(data)

    @property
    def bstar(self):
        return self._bstar

    @bstar.setter
    def bstar(self, data):
        self._bstar = np.array(data)

    @property
    def cstar(self):
        return self._cstar

    @cstar.setter
    def cstar(self, data):
        self._cstar = np.array(data)

    @property
    def lc(self):
        return self._lc

    @lc.setter
    def lc(self, data):
        self._lc = np.array(data)

    @property
    def lattice(self):
        return self._lattice

    @lattice.setter
    def lattice(self, data):
        self._lattice = np.array(data)

    @property
    def reciprocal_basis(self):
        # base vectors (matrix) of reciprocal lattice
        return np.column_stack((self.astar,self.bstar,self.cstar))

    # Validate data to make sure we got all the fields
    # we need from the DAXM data file. Sometime the results file
    # can be corrupted such that only part of the data is available.
    # In this case, we have no choice but to mark the affect voxel as
    # corrupted and discard it from the calculation.
    def validate(self,
                 skip=False,
                 tor=1e-2):
        """
        DESCRIPTION
        -----------
        self.validate()
            Validate all parameters are parsed;
            Prune q vectors, ensure correct mapping between
            self.hkls and self.qs;
            Instance of VoxelStep can only be used when validated.
            If strain refinement is not required, set skip=True for
            quick data process.
        PARAMETERS
        ----------
        skip: boolean
            This flag allow a simple bypass of the type check that
            ensures all attributes are properly assigned.
        tor: float
            Tolerance for q vectors pruning.
        RETURNS
        -------
        self._valid: boolean
            Return the state of the voxel (valid/invalid)
        """
        # allow bypass security if necessary
        if not skip:
            # assert that all attributes are assigned
            assert self.Xsample is not None
            assert self.Ysample is not None
            assert self.Zsample is not None
            assert self.depth   is not None
            # prune through qs to remove non-indexed peaks
            rl = self.reciprocal_basis
            n_peaks = self.hkls.shape[0]
            new_qs = np.empty(self.hkls.shape)
            #**STAET:PRUNING_Q
            # brutal search to locate the q vectors associated
            # with hkl indexation
            # NOTE:
            #   The searching here should be optimized at some point
            for i in range(n_peaks):
                threshold = tor
                hkl = self.hkls[i,:]
                q_calc = np.dot(rl, hkl)
                q_calc = q_calc/np.linalg.norm(q_calc)
                for j in range(self.qs.shape[0]):
                    tmp = 1.0 - abs(np.dot(q_calc, self.qs[j,:]))
                    if tmp < threshold:
                        # try to located the closet match
                        threshold = tmp
                        new_qs[i,:] = self.qs[j,:]
            #**END:PRUNING_Q
            # save pruning results
            self.qs = new_qs
        # update flag to unlock access to this voxel
        self._valid = True
        return self._valid

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
        NOTE
        ----
        The rotation matrix and orientation matrix are a very confusing
        couple, especially when it comes to crystallography. This is
        most due to the fact both the crystal and the reference are constantly
        transform during crystallography calculation. The general rule of thumb
        in determine which matrix should be used should be as follows:
            if crystal.rotated is True & reference.rotated is False:
                use Rotation_Matrix
            elif reference.rotated is True & crystal.rotated if False:
                use Orientation_Matrix
            else:
                call divide_and_couqure()
            endif
        """
        if not(self._valid):
            msg = "Self validation failed, try self.validate()?."
            raise ValueError(msg)
        # calculate voxel position based on motor position in XHF
        # // This is the quickest way to get the coordinates
        coord = [-self.Xsample, -self.Ysample, -self.Zsample+self.depth]
        # depends on the desired configuration, change the rotation
        # matrix accordingly
        ref = ref.upper()
        if ref ==  'TSL':
            r = R_XHF2TSL
        elif ref == 'APS':
            r = R_XHF2APS
        elif ref == 'XHF':
            r = R_XHF2XHF
        else:
            raise ValueError("unknown configuration")
        # since we are changing coordinate system, use orientation matrix
        # or rotation_matrix.transposed
        rst = np.dot(r.T, coord)
        return rst

    def get_eulers(self, ref='TSL'):
        """
        DESCRIPTION
        -----------
        phi1, PhH, phi2 = self.get_eulers(ref='TSL')
        PARAMETERS
        ----------
        ref: string
            The configuration in which the Euler Angles is computed.
            The default output (a*,b*,c*) in the xml file is in the
            APS coordinate system according to
            http://www.aps.anl.gov/Sectors/33_34/microdiff/Instrument/coordinates-PE-system.pdf
        RETURNS
        -------
        phi1,PHI,phi2: tuple
            Computed Euler angles in degrees
        NOTE
        ----
        The change of reference configuration will affect the output
        of the Euler angle calculation, as a result, it is necessary
        define what configuration/reference the calculation is in and
        make sure all calculation is done under the same reference
        configuration.
        """
        if not(self._valid):
            raise ValueError("Invalid data, ABORT!")
        # get the rotation matrix first
        ref = ref.upper()
        if ref ==  'TSL':
            r = R_APS2TSL
        elif ref == 'APS':
            r = R_APS2APS
        elif ref == 'XHF':
            r = R_APS2XHF
        else:
            raise ValueError("unknown configuration")
        # calculate the real lattice first
        c = np.cross(self.astar, self.bstar)
        c = c/np.linalg.norm(c)
        a = np.cross(self.bstar, self.cstar)
        a = a/np.linalg.norm(a)
        b = np.cross(c, a)
        b = b/np.linalg.norm(b)
        # rotate real lattice into given configuration
        a = np.dot(r.T, a)
        b = np.dot(r.T, b)
        c = np.dot(r.T, c)
        # use the three basis vectors to form a base,
        # its column stack will be the orientation matrix, which
        # is used to perform reference transformation
        g = np.column_stack((a, b, c))
        # use cyxtal package to calculate the Euler angles
        # equivalent method using damask (for reference):
        # eulers = Orientation(matrix=g, symmetry='hexagonal').reduced()
        # eulers = np.degrees(eulers.asEulers())
        return OrientationMatrix(g).toEulers()

    def get_strain(self,
                   ref='TSL',
                   method='nelder-mead',
                   mask=(0,0,0,0,0,0),
                   xtor=1e-8,
                   disp=True,
                   deviatoric=True,
                   maxiter=1e6,
                   approximate=False):
        """
        DESCRIPTION
        -----------
        epsilon = self.get_strain(ref='TSL')
            Return strain tensor extracted/inferred through strain
            refinement process for current voxel. The returned strain
            tensor is transformed into designated coordinate system.
        PARAMETERS
        ----------
        RETURNS
        -------
        NOTE
        ----
            Since the strain is approximated using the (a*,b*,c*), which are
            in the APS coordinate system.
        """
        # check if voxel data is valid
        if not(self._valid):
            print "Corrupted voxel found!"
            return strain.fill(np.nan)
        # some preparation before hard computing
        ref = ref.upper()
        if ref == "TSL":
            r = R_APS2TSL
        elif ref == "APS":
            r = R_APS2APS
        elif ref == "XHF":
            r = R_APS2XHF
        else:
            raise ValueError("Unknown reference configuration")
        g = r.T  # orientation matrix is used to reference transformation
        ##
        # step 1: extract rotation (transformation).
        lc_std = self.lc
        # Bstar_0: strain free, rotation free reciprocal basis
        # Bstar_1: stretched, rotation free reciprocal basis,
        #          Bstar_1 = u_lattice * Bstar_0
        # Bstar_2: stretch and rotated basis,
        #          Bstar_2 = r_lattice * Bstar_1
        # Bstar_3: strain free, rotated reciprocal lattice
        #          forced u_lattice = I
        Bstar_0 = get_base(lc_std)
        Bstar_3 = self.reciprocal_basis
        # find the rotation matrix that converts a standard reciprocal basis
        # to the APS configuration
        r_lattice = np.dot(Bstar_3, np.linalg.inv(Bstar_0))
        ##
        # step 2: call scipy.optmize.minimize on the objective function
        #         self.get_qmismatch to find the ideal set of lattice
        #         constants that provide best match to measured Q vectors.
        lc_ini  = lc_std
        refine  = minimize(self.strain_refine,
                           lc_ini,
                           args=tuple([r_lattice, mask]),
                           method=method,
                           options={'xtol': xtor,
                                    'disp': disp,
                                    'maxiter': int(maxiter),
                                    'maxfev' : int(maxiter)})
        print "ideal: ", self.lc
        print refine
        lc_fin = refine.x
        ##
        # step 3: calculate the stretch tensor using the deformation gradient
        #         Dr. Tischler is doing all the calculation in the reciprocal
        #         space, however the deformation gradient is in real space.
        #         Based on the derivation in the reference, the
        # ref: cyxtal/documentation
        Bstar_1 = get_base(lc_fin)
        # Bstar_2 = np.dot(r_lattice, Bstar_1)
        u_fin = np.dot(np.linalg.inv(Bstar_1.T), Bstar_0.T)
        epsilon = 0.5*(np.dot(u_fin.T, u_fin) - np.eye(3))
        if approximate:
            epsilon = u_fin - np.eye(3)  # approximation
        # if no white beam energy provided, remove the hydrostatic component
        # as it has no physical meaning
        if deviatoric:
            epsilon = epsilon - np.eye(3)*np.trace(epsilon)/3.0
        ##
        # step 4: transform strain tensor to requested configuration
        return np.dot(g, np.dot(epsilon, g.T))

    def strain_refine(self, lc, r, msk):
        """
        DESCRIPTION
        -----------
        rst = self.strain_refine(new_lc)
            Return the errors between calculated qs and measurements
        PARAMETERS
        ----------
        new_lc: lattice constant (perturbed)
        RETURNS
        -------
        rst:  float
            angular difference between calculated qs using new_lc and
            measurements (self.qs). A penalty term (delta_V) is added
            to ensure no large strain happens to the unit cell.
        NOTE
        ----

        """
        # only perturb the lattice parameter indicated by the mask
        # 1 means keep ideal, 0 means perturb
        for i in range(6):
            if msk[i] == 1:
                lc[i] = self.lc[i]
        Bstar_1 = get_base(lc)
        Bstar_2 = np.dot(r, Bstar_1)
        # Penalty: delta_Vcell
        #   first calculate the changes in the unit cell volume
        #   use 20% of the change in volume as the penalty term
        wgt = 0.008
        Vcell0 = np.linalg.det(self.reciprocal_basis)
        Vcell2 = np.linalg.det(Bstar_2)
        dVcell = abs(Vcell2 - Vcell0)
        rst = dVcell*wgt
        # now add angular differences into the control
        hkls = self.hkls
        qs = self.qs
        for i in range(qs.shape[0]):
            # calculate new Q vector based on perturbed unit cell
            q_tmp = np.dot(Bstar_2, hkls[i])
            q_tmp = q_tmp/np.linalg.norm(q_tmp)
            rst += abs(np.dot(q_tmp, qs[i]))
        return rst

##################################
# MODULE LEVEL FUNCTIONS/METHODS #
##################################
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


def get_base(lc,
             reciprocal=True,
             degrees=True):
    """
    DESCRIPTION
    -----------
    basis = get_base(lc)
        return the basis constructed based given lattice constant.
    PARAMETERS
    ----------
    lc: numpy.array/list/tuple [a,b,c,alpha,beta,gamma]
        Should contain necessary lattice constants that defines
        crystal structure.
    reciprocal: boolean
        Whether the returned basis vectors in real reciprocal space
        or real space.
    va: basis vector construct convention ['x'/'y']
        The a vector should either lies along x-axis or y-axis. Default
        set to x-axis as this is what has been used in APS@ANL.
    degree: boolean
        The angular lattice parameter are in degrees or radians.
    RETURNS
    -------
    rst: numpy.array
        A 3x3 numpy array formed by the base vectors of given
        lattice constant. The base vectors are stack by column.
    """
    # unpack lattice constant
    a,b,c,alpha,beta,gamma = lc
    if degrees:
        alpha, beta, gamma = np.radians([alpha, beta, gamma])
    # just trying to make syntax cleaner
    s_a = np.sin(alpha)
    c_a = np.cos(alpha)
    s_b = np.sin(beta)
    c_b = np.cos(beta)
    s_g = np.sin(gamma)
    c_g = np.cos(gamma)
    # calculating base vectors using lattice constants
    # ref: cyxtal/documentation
    factor = 1 + 2*c_a*c_b*c_g - c_a**2 - c_b**2 - c_g**2
    vol_cell = a*b*c*np.sqrt(factor)
    v1 = [a, 0.0, 0.0]
    v2 = [b*c_g, b*s_g, 0.0]
    v3 = [c*c_b, c*(c_a-c_b*c_g)/(s_g), vol_cell/(a*b*s_g)]
    # form the base
    rst = np.column_stack((v1,v2,v3))
    # calculating reciprocal lattice based on real lattice
    # ref: https://en.wikipedia.org/wiki/Reciprocal_lattice
    if reciprocal:
        rst = 2*np.pi*np.linalg.inv(rst)
        rst = rst.T
    return rst