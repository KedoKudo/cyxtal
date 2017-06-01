#!/usr/bin/env python

# Sanity check with the current fitness function for strain refinement.
# 1. Random populate some deformation gradient (F) and calculate the strian
#    tensor.
# 2. Use the defomration gradient to distor a perfect lattice unit cell.
# 3. Collecting (hkl)-qv pair.
# 4. Use the strain refinement machinary to extract the strain tensor.
# 5. Measure the difference betweeen theoretical strain tensor and extracted
#    strain tensor

import numpy as np
from scipy.linalg import polar
from scipy.optimize import minimize
from cyxtal.ext_aps.parsers import F2DeviatoricStrain
from cyxtal.ext_aps.parsers import get_base


def gen_distorsion():
  """ generate random distorsion in the form of deformation gradient """
  F = np.random.random((3,3))
  R, U = polar(F, side='right')  # no rotation

  print 3-np.trace(R)
  return U, F2DeviatoricStrain(F)


def gen_qv(unitcell_base, hkl=None):
  """ generate the q vectors for given unitcell base """
  pass


if __name__ == "__main__":
  # generate the testing distorsion
  U, epsilon_ideal = gen_distorsion()

  # generate a zero strain base
  lc = [0.2965, 0.2965, 0.4747, 90, 90, 120]
  B_perfect = get_base(lc)
  # strain the perfect unit cell
  B_strained = np.dot(U,B_perfect)
