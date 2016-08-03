__author__='C.Z'

"""
   ________  ___  ___________    __
  / ____/\ \/ / |/ /_  __/   |  / /
 / /      \  /|   / / / / /| | / /
/ /___    / //   | / / / ___ |/ /___
\____/   /_//_/|_|/_/ /_/  |_/_____/

Copyright (c) 2016, C. Zhang.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification are permitted provided that the following conditions are met:

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
"""

# Base modules for computation
from cyxtal.cxtallite import symmetry
from cyxtal.cxtallite import Quaternion
from cyxtal.cxtallite import Xtallite
from cyxtal.cxtallite import slip_systems
from cyxtal.cxtallite import bravis2cartesian

from cyxtal.ctools import kmeans
from cyxtal.ctools import get_vonMisesStress
from cyxtal.ctools import get_vonMisesStrain

from cyxtal.geometry import Point
from cyxtal.geometry import Line
from cyxtal.geometry import Plane
from cyxtal.geometry import Point2D
from cyxtal.geometry import Line2D
from cyxtal.geometry import Polygon2D
from cyxtal.geometry import convex_hull

# Interface modules for vtk,damask
import ext_vtk
import ext_damask

from ext_damask.geom_gen import geom_fromRCB

# Auxiliary modules for APS, TSL, damask
from cyxtal.ext_aps.parsers import VoxelStep
from cyxtal.ext_aps.parsers import parse_xml
from cyxtal.ext_aps.parsers import get_base
from cyxtal.ext_aps.parsers import get_reciprocal_base
