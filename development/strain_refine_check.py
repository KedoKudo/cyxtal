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