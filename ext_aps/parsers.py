"""
   ________  ___  ___________    __
  / ____/\ \/ / |/ /_  __/   |  / /
 / /      \  /|   / / / / /| | / /
/ /___    / //   | / / / ___ |/ /___
\____/   /_//_/|_|/_/ /_/  |_/_____/

Copyright (c) 2017, C. Zhang.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

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
VoxelStep: class
  Container class to store voxel information and perform strain refinement.
parser_xml: function
  Parsing xml output from APS (with/without strain refinement).
get_reciprocal_base:
  Return reciprocal basis according to given lattice constants.
get_base: function
  Return lattice basis according to given lattice constants.
NOTE
----
More information regarding the coordinate transformation can be found at:
  http://www.aps.anl.gov/Sectors/33_34/microdiff/Instrument/coordinates-PE-system.pdf
"""

# import h5py as h5
import numpy as np
import xml.etree.cElementTree as ET
from scipy.optimize import minimize
from scipy.linalg import sqrtm
from cyxtal.cxtallite import OrientationMatrix
# from cyxtal import get_vonMisesStrain

##
# MODULE LEVEL CONSTANTS RELATING TO COORDINATE TRANSFORMATION
# <NOTE>
#   These are defined in terms of rotation matrices since it is
#   more intuitive to see how each system is connected through
#   simple rotation around x-axis (see cyxtal/documentation)
#
# ** XHF <-> TSL
theta_1 = -np.pi
R_XHF2TSL = np.array([[1.0,              0.0,              0.0],
                      [0.0,  np.cos(theta_1), -np.sin(theta_1)],
                      [0.0,  np.sin(theta_1),  np.cos(theta_1)]])
R_TSL2XHF = R_XHF2TSL.T
# ** XHF <-> APS
theta_2 = -0.25*np.pi
R_XHF2APS = np.array([[1.0,              0.0,              0.0],
                      [0.0,  np.cos(theta_2), -np.sin(theta_2)],
                      [0.0,  np.sin(theta_2),  np.cos(theta_2)]])
R_APS2XHF = R_XHF2APS.T
# ** APS <-> TSL
theta_3 = -0.75*np.pi
R_APS2TSL = np.array([[1.0,              0.0,              0.0],
                      [0.0,  np.cos(theta_3), -np.sin(theta_3)],
                      [0.0,  np.sin(theta_3),  np.cos(theta_3)]])
R_TSL2APS = R_APS2TSL.T
# ** self <-> self
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
  X|Y|Zsample: sample motor position during scan (X|Y|Z)
  depth:       wire position
  qs:          identified diffraction vectors
  hkls:        hkl indices identified
  a|b|cstar:   strain free reciprocal lattice identified
  lc:          lattice constants used in indexation
  lattice:     lattice structure
  goodness:    the indexation goodness of first pattern (highest confidence)
  _validonly:  validation state of the voxel
  """

  def __init__(self):
    # coordinates
    self._Xsample = None
    self._Ysample = None
    self._Zsample = None
    self._depth = None
    # indexing (shape unknown)
    self._qs = None
    self._hkls = None
    self._peaks = None
    # diffraction image name
    self._h5Img = None
    # strain free reciprocal lattice
    self._astar = None
    self._bstar = None
    self._cstar = None
    # lattice constant
    self._lc = None
    self._lattice = None
    # pattern goodness
    self._goodness = None
    # validation
    self._validonly = False

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
  def goodness(self):
    return self._goodness

  @goodness.setter
  def goodness(self, data):
    self._goodness = float(data)

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
  def peaks(self):
    return self._peaks

  @peaks.setter
  def peaks(self, data):
    self._peaks = np.array(data)

  @property
  def h5Img(self):
    return self._h5Img

  @h5Img.setter
  def h5Img(self, img):
    self._h5Img = str(img)

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
    return np.column_stack((self.astar, self.bstar, self.cstar))

  # Validate data to make sure we got all the fields
  # we need from the DAXM data file. Sometime the results file
  # can be corrupted such that only part of the data is available.
  # In this case, we have no choice but to mark the affect voxel as
  # corrupted and discard it from the calculation.
  def validate(self, skip=False, tor=1e-4):
    """
    DESCRIPTION
    -----------
    self.validate()
      Validate all parameters are parsed;
      Prune q vectors, ensure correct mapping between
      self.hkls and self.qs, and self.peaks;
      // the peak position on the detector can be important for
      // streak analysis
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
    self._validonly: boolean
      Return the state of the voxel (valid/invalid)
    """
    # allow bypass security if necessary
    if not skip:
      # assert that all attributes are assigned
      assert self.Xsample is not None
      assert self.Ysample is not None
      assert self.Zsample is not None
      assert self.depth is not None
      # prune through qs to remove non-indexed peaks
      rl = self.reciprocal_basis
      new_qs = np.empty(self.hkls.shape)
      new_peaks = np.empty((self.hkls.shape[0], 2))
      # ----- START:PRUNING_Q ----- #
      # The way DAXM indexation works is like this. The peak fining
      # program will find as many peaks as it can, and the indexation
      # will only index a subset of the peaks found. Thus the number of
      # peaks and qs should always greater than the number of indexations
      # found (hkls). So the pruning essentially is to correctly pairing
      # HKL with q vectors and peaks position.
      # NOTE:
      #   The searching here should be optimized at some point
      for i, hkl in enumerate(self.hkls):
        q_calc = np.dot(rl, hkl)
        q_calc = q_calc/np.linalg.norm(q_calc)
        # use brutal force to test the difference from all qs
        diff = [1.0-abs(np.dot(q_mea, q_calc)) for q_mea in self.qs]
        if min(diff) > tor:
          msg = "threshold too tight! "
          msg += "Cannot find q vectors to match indexation"
          raise ValueError(msg)
        else:
          new_qs[i, :] = self.qs[np.argmin(diff), :]
          new_peaks[i, :] = self.peaks[np.argmin(diff), :]
      # ----- END:PRUNING_Q ----- #
      # save pruning results
      self.qs = new_qs
      self.peaks = new_peaks
    # update flag to unlock access to this voxel
    self._validonly = True
    return self._validonly

  def __str__(self):
    if not(self._validonly):
      return "Not validated"
    msg = 'DAXM voxel:\n'
    msg += ' Motor/wire position:\n'
    msg += '  Xsample: {}\n'.format(self.Xsample)
    msg += '  Ysample: {}\n'.format(self.Ysample)
    msg += '  Zsample: {}\n'.format(self.Zsample)
    msg += '    depth: {}\n'.format(self.depth)
    msg += '  H5 iamge: {}\n'.format(self.h5Img)
    msg += ' Q vectors(qx,qy,qz):\n'
    msg += str(self.qs) + '\n'
    msg += ' peaks on detector(pixelx, pixely):\n'
    msg += str(self.peaks) + '\n'
    msg += ' HKLs (h,k,l):\n'
    msg += str(self.hkls) + '\n'
    msg += ' Reciprocal lattice vectors:\n'
    msg += '  a*:' + str(self.astar) + '\n'
    msg += '  b*:' + str(self.bstar) + '\n'
    msg += '  c*:' + str(self.cstar) + '\n'
    msg += ' Pattern Goodness:\n'
    msg += '  ' + str(self.goodness) + '\n'
    msg += ' Lattice Constants for Indexation:\n'
    msg += '  ' + str(self.lc) + '\n'
    return msg

  def get_coord(self, ref='TSL', translate=(0, 0, 0)):
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
    translate: array
      Translate voxel with given translation vector after
      rotating to the desired reference system.
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
    if not(self._validonly):
      msg = "Self validation failed, try self.validate()?."
      raise ValueError(msg)
    # calculate voxel position based on motor position in XHF
    # // This is the quickest way to get the coordinates
    coord = [-self.Xsample, -self.Ysample, -self.Zsample+self.depth]
    coord = np.array(coord)
    # depends on the desired configuration, change the rotation
    # matrix accordingly
    ref = ref.upper()
    if ref == 'TSL':
      r = R_APS2TSL
    elif ref == 'APS':
      r = R_APS2APS
    elif ref == 'XHF':
      r = R_APS2XHF
    else:
      raise ValueError("unknown configuration")
    # since we are changing coordinate system, use orientation matrix
    # or rotation_matrix.transposed
    rst = np.dot(r.T, coord) + np.array(translate)
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
    if not(self._validonly):
      raise ValueError("Invalid data, ABORT!")
    # get the rotation matrix first
    ref = ref.upper()
    if ref == 'TSL':
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
    # equivalent to b*,
    # b*/b go through atoms
    # a go through faces
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

    return OrientationMatrix(g).toEulers()

  def get_strain(self,
                 ref='TSL', xtor=1e-10, verbose=False, deviatoric='m2',
                 maxiter=1e10, opt_method='nelder-mead', min_qv=4):
    """
    DESCRIPTION
    -----------
    epsilon = self.get_strain(ref='TSL')
      Return strain tensor extracted/inferred through strain
      refinement process for current voxel. The returned strain
      tensor is transformed into designated coordinate system.
    PARAMETERS
    ----------
    ref:  str ['APS', 'TSL', XHF]
      The coordinate system in which the refined strain tensor
      will be returned.
    xtor: float
      Tolerance used in the optimization of finding strained unit
      cell
    verbose: boolean
      Toggle the reporting of optimization process results
    deviatoric: str ['tishler', 'm1', 'm2']
      Specify which method should be used for the calculation of
      deviatoric strain.
    maxiter: float
      Maximum iterations/calls allowed during the optimization
    min_qv: int
      Minimum amount of q vectors to start strain refinement. Igor
      default setting is 4. @Dr.Tischler
    RETURNS
    -------
    epsilon: np.array (3,3)
      Green--Lagrange strain tensor in given reference configuration
    NOTE
    ----
      The strain is approximated using the (a*,b*,c*), which are
      in the APS coordinate system.
    """
    # check if voxel data is valid
    if not(self._validonly):
      print "Corrupted voxel found!"
      raise ValueError('Validate data first before strain refinement!')
    if self.hkls.shape[0] < min_qv:
      print "insufficient diffractions spots"
      epsilon = np.empty((3, 3))
      return epsilon.fill(np.nan)
    # feature vectors:
    #   [a*_1, a*_2, a*_3, b*_1, b*_2, b*_3, c*_1, c*_2, c*_3]
    v_features = np.reshape(self.reciprocal_basis, 9, order='F')
    # use scipy minimization module for optimization
    refine = minimize(self.strain_refine,
                      v_features,
                      method=opt_method,
                      options={'xtol': xtor,
                               'disp': verbose,
                               'maxiter': int(maxiter),
                               'maxfev': int(maxiter)})
    # verbose reporting
    if verbose:
      Bstar = refine.x.reshape((3, 3), order='F')  # final B*
      B = 2*np.pi*np.linalg.inv(Bstar).T  # final B from B*
      print "reciprocal:[a*;b*;c*]\n", Bstar
      print "real:[a;b;c]\n", B
    # extract refined reciprocal basis
    B_fin = np.reshape(refine.x, (3, 3), order='F')
    B_org = self.reciprocal_basis
    # force rescaling
    # B_fin = B_fin / np.linalg.det(B_fin) * np.linalg.det(B_org)
    # calc deformation gradient
    F_fin = np.dot(B_org, np.linalg.inv(B_fin)).T
    # *** switching to new deviatoric strain calculation
    epsilon = F2DeviatoricStrain(F_fin, method=deviatoric, debug=verbose)

    # transform strain tensor to requested configuration
    ref = ref.upper()
    if ref == "TSL":
      r = R_APS2TSL
    elif ref == "APS":
      r = R_APS2APS
    elif ref == "XHF":
      r = R_APS2XHF
    else:
      raise ValueError("Unknown reference configuration")
    # orientation matrix is used to reference transformation
    g = r.T
    epsilon = np.dot(g, np.dot(epsilon, g.T))
    return epsilon

  def strain_refine(self, v_features):
    """
    DESCRIPTION
    -----------
    rst = self.strain_refine(v_features)
      This is the objective function for the strain refinement.
    PARAMETERS
    ----------
    v_features: np.array
      feature vectors
      (a*_1, a*_2, a*_3, b*_1, b*_2, b*_3, c*_1, c*_2, c*_3)
    RETURNS
    -------
    rst: float
      1-cos(q_calc, q_meas).
    NOTE
    ----
      This approach is still under construction. Further change of
      the objective function is possible
    """
    # convert to reciprocal basis
    B_new = np.reshape(v_features, (3, 3), order='F')

    rst = 0.0
    # now add q vector differences into the control
    hkls = self.hkls
    qs = self.qs
    for i in xrange(qs.shape[0]):
      # calculate new Q vector based on perturbed unit cell
      q_tmp = np.dot(B_new, hkls[i])
      q_tmp = q_tmp/np.linalg.norm(q_tmp)
      rst += np.dot(q_tmp, qs[i])
    # get avg cos(theta)
    rst = 1.0 - rst/qs.shape[0]

    return rst


##################################
# MODULE LEVEL FUNCTIONS/METHODS #
##################################
def parse_xml(xmlfile,
              namespace={'step':
                         'http://sector34.xray.aps.anl.gov/34ide:indexResult'},
              verbose=True,
              keepEmptyVoxel=False):
  """
  DESCRIPTION
  -----------
  [VoxelStep(),...]= parse_xml(DAXM_DATA.xml,
                               namespace={$XML_NAMESPACE_DICT},
                               verbose=True)
    Parse the DAXM data from Beamline 34-I-DE to memory.
  PARAMETERS
  ----------
  xmlfile: str
    Path to the xml file requires data processing
  namespace: dictionary
    Containing dictionary of the namespace used in the xml file.
    For data from beamline 34-ID-E, use the default setting should
    work.
    NOTE:
      If the beamline changes there namespace, it is necessary to
      extract those namespace and update them with this argument.
  verbose: boolean
    Toggle output of parsing progress (terminal only)
  keepEmptyVoxel: boolean
    Keep non-indexed voxel in the return data set
  RETURNS
  -------
  voxels: list of VoxelStep
    List of instances of VoxelStep, each one representing indexed voxel
    in the xml data.
    NOTE:
      Not indexed file is screened out by checking the presence of a*
      for each voxel.
  NOTE
  ----
  """
  # read in the xml file using cElementtree
  tree = ET.parse(xmlfile)
  root = tree.getroot()
  voxels = []                  # empty container
  skipped = 0                   # keep track of how many voxel skipped
  ns = namespace
  sep_head = '\n' + '*'*60
  sep_tail = '*'*60 + "\n"
  # walk through each step
  if verbose:
    print sep_head
    print 'Extract data from XML file'
  for i in range(len(root)):
    step = root[i]
    # determine if voxel is indexed
    target_str = 'step:indexing/step:pattern/step:recip_lattice/step:astar'
    astar = step.find(target_str, ns)
    if astar is None:
      skipped += 1
      if keepEmptyVoxel:
        voxels.append('nan')
      continue
    # progress bar for parsing
    if verbose:
      state = float(i+1)/len(root)
      bar = '[' + '#'*int(state*10) + ' '*(10-int(state*10)) + ']'
      print '\r'+bar+'{:.2%}'.format(state),
    # STEP 1: EXTRACT TEXT STRING
    # |->motor/wire position
    xsample = step.find('step:Xsample', ns).text
    ysample = step.find('step:Ysample', ns).text
    zsample = step.find('step:Zsample', ns).text
    depth = step.find('step:depth', ns).text
    # |->H5 image name
    h5img = step.find('step:detector/step:inputImage', ns).text
    # |->peak position on CCD
    xpix = step.find('step:detector/step:peaksXY/step:Xpixel', ns).text
    ypix = step.find('step:detector/step:peaksXY/step:Ypixel', ns).text
    # |->diffraction vectors
    qx = step.find('step:detector/step:peaksXY/step:Qx', ns).text
    qy = step.find('step:detector/step:peaksXY/step:Qy', ns).text
    qz = step.find('step:detector/step:peaksXY/step:Qz', ns).text
    # |->reciprocal lattice vectors
    astar_str = 'step:indexing/step:pattern/step:recip_lattice/step:astar'
    bstar_str = 'step:indexing/step:pattern/step:recip_lattice/step:bstar'
    cstar_str = 'step:indexing/step:pattern/step:recip_lattice/step:cstar'
    astar = step.find(astar_str, ns).text
    bstar = step.find(bstar_str, ns).text
    cstar = step.find(cstar_str, ns).text
    # |->index results (hkl)
    h = step.find('step:indexing/step:pattern/step:hkl_s/step:h', ns).text
    k = step.find('step:indexing/step:pattern/step:hkl_s/step:k', ns).text
    l = step.find('step:indexing/step:pattern/step:hkl_s/step:l', ns).text
    # |->indexation goodess for the first set of patterns
    gd_str = 'step:indexing/step:pattern'
    goodness = step.find(gd_str, ns).attrib['goodness']
    # |->lattice constants (ideal)
    lc_str = 'step:indexing/step:xtl/step:latticeParameters'
    lc = step.find(lc_str, ns).text
    # STEP 2: PARSE DATA TO MEMORY
    voxel = VoxelStep()
    # |->motor/wire position
    voxel.Xsample = float(xsample)
    voxel.Ysample = float(ysample)
    voxel.Zsample = float(zsample)
    voxel.depth = float(depth)
    # |->record H5 image
    voxel.h5Img = h5img
    # |->record peak positions on CCD
    xpix = map(float, xpix.split())
    ypix = map(float, ypix.split())
    voxel.peaks = np.column_stack((xpix, ypix))
    # |->diffraction vectors
    qx = map(float, qx.split())
    qy = map(float, qy.split())
    qz = map(float, qz.split())
    voxel.qs = np.column_stack((qx, qy, qz))
    # |->diffraction pattern goodness
    voxel.goodness = goodness
    # |->reciprocal lattice vectors
    voxel.astar = map(float, astar.split())
    voxel.bstar = map(float, bstar.split())
    voxel.cstar = map(float, cstar.split())
    # |->index results (hkl)
    h = map(float, h.split())
    k = map(float, k.split())
    l = map(float, l.split())
    voxel.hkls = np.column_stack((h, k, l))
    # |->lattice constants (ideal)
    voxel.lc = map(float, lc.split())
    # validate the voxel
    voxel.validate()
    # STEP 3: PUSH DATA TO CONAINER ARRAY/LIST
    voxels.append(voxel)
  if verbose:
    print '\n', sep_tail
    print sep_head
    print "XML FILE: {}".format(xmlfile)
    print " Total number of voxels:\t\t{}".format(len(root))
    print " Valid voxel for DAXM analysis:\t{}".format(len(voxels))
    data_q = float(len(voxels))/len(root)
    print " Dataset goodness:\t\t\t{:.2%}".format(data_q)
    print sep_tail

  return voxels


def get_reciprocal_base(lc, degrees=True):
  """
  DESCRIPTION
  -----------
  reciprocal_basis = get_reciprocal_base(lc)
    wrapper function to return the reciprocal basis rather
    than standard basis
  PARAMETERS
  ----------
  lc: numpy.array/list/tuple [a,b,c,alpha,beta,gamma]
    Should contain necessary lattice constants that defines
    crystal structure.
  degree: boolean
    The angular lattice parameter are in degrees or radians.
  RETURNS
  -------
  rst: numpy.array
    A 3x3 numpy array formed by the reciprocal base vectors of
    given lattice constant. The base vectors are stack by column.
  """
  return get_base(lc, reciprocal=True, degrees=degrees)


def get_base(lc,
       reciprocal=False,
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
  degree: boolean
    The angular lattice parameter are in degrees or radians.
  RETURNS
  -------
  rst: numpy.array
    A 3x3 numpy array formed by the base vectors of given
    lattice constant. The base vectors are stack by column.
  """
  # unpack lattice constant
  a, b, c, alpha, beta, gamma = lc
  if degrees:
    alpha, beta, gamma = np.radians([alpha, beta, gamma])
  # just trying to make syntax cleaner
  # s_a = np.sin(alpha)
  c_a = np.cos(alpha)
  # s_b = np.sin(beta)
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
  rst = np.column_stack((v1, v2, v3))
  # calculating reciprocal lattice based on real lattice
  # ref: https://en.wikipedia.org/wiki/Reciprocal_lattice
  if reciprocal:
    # use pinv to avoid singluar matrix situation.
    rst = 2*np.pi*np.linalg.pinv(rst)
    rst = rst.T
  return rst


def base_hcp2cartesian(B_hcp,
             reciprocal=False):
  """
  DESCRIPTION
  -----------
  B_cartesian = base_hcp2cartesian(B_hcp)
    Convert [reciprocal] lattice basis from [a;b;c] to [x,y,z] where
    vec_x = vec_a
    vec_y = vec_b + 0.5*vec_a
    vec_z = vec_c
  PARAMETERS
  ----------
  B_hcp: np.array (3,3)
    Lattice basis where each column vector represents vec_a,
    vec_b and vec_c, i.e.
    B_hcp = np.column_stack((vec_a, vec_b, vec_c))
  reciprocal: boolean
    Whether the basis is reciprocal basis or not, if so convert
    to real space for calculation, then convert back.
  RETURNS
  -------
    Lattice basis where each column vector represents vec_x,
    vec_y and vec_z, i.e.
    B_cartesian = np.column_stack((vec_x, vec_y, vec_z))
  NOTE
  ----
  """
  if reciprocal:
    B_hcp = 1.0/(2*np.pi)*(np.linalg.inv(B_hcp)).T
  vec_a = B_hcp[:, 0]
  vec_b = B_hcp[:, 1]
  vec_c = B_hcp[:, 2]
  vec_x = vec_a
  vec_y = vec_b + 0.5*vec_a
  vec_z = vec_c
  B_cartesian = np.column_stack(((vec_x, vec_y, vec_z)))
  if reciprocal:
    B_cartesian = 2*np.pi*(np.linalg.inv(B_cartesian)).T
  return B_cartesian


def F2DeviatoricStrain(F, method='m2', debug=False):
  """
  DESCRIPTION
  -----------
  epsilon_D = F2DeviatoricStrain(F, method)
    Calculating deviatoric strain from full deformation gradient
    using specified method
  PARAMETERS
  ----------
  F: np.array (3,3)
    Full deformation gradient
  method: str ['tischler', 'm1', 'm2']
    Specify which method should be used for the calculation of
    deviatoric strain.
  RETURNS
  -------
  epsilon_D: np.array(3,3)
    A strain tensor without hydrostatic component.
  NOTE
  ----
  """
  method = method.lower()
  I = np.eye(3)  # identity matrix
  J = np.linalg.det(F)  # Jacobian of F
  # Start calculation
  if method == 'tischler':
    U = sqrtm(np.dot(F.T, F))  # stretch tensor U^2 = F.T*F
    epsilon = U - I
    epsilon_D = epsilon - 1./3*np.trace(epsilon)*I
  elif method == 'm1':
    U = sqrtm(np.dot(F.T, F))
    epsilon_D = U - J**(1./3.)*I
  elif method == 'm2':
    epsilon_D = 0.5*(np.dot(F.T, F) - J**(2./3.)*I)
  else:
    msg = "Unknown method for deviatoric calc: {}".format(method)
    raise ValueError(msg)
  # debug output
  if debug:
    print "method--> ", method
    print "strain:\n", epsilon_D, "\n"
    print "J:\n", J, "\n"
  return epsilon_D
