#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-

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

DESCRIPTION
-----------
Provide various method to generate geom file for spectral solver@DAMASK.

METHOD
------
geom_fromRCB()
geom_from...()

NOTE
----
--> The default header for a reconstructed boundary file from TSL
# Column 1-3:    right hand average orientation (phi1, PHI, phi2 in radians)
# Column 4-6:    left hand average orientation (phi1, PHI, phi2 in radians)
# Column 7:      length (in microns)
# Column 8:      trace angle (in degrees)
# Column 9-12:   x,y coordinates of endpoints (in microns)
# Column 13-14:  IDs of right hand and left hand grains
"""

import numpy as np
from cyxtal import Point2D
from cyxtal import Polygon2D


def geom_fromRCB(rcbFile,
                 output=None,
                 viz=None,
                 rim=1,
                 thickness=1,
                 step=1,
                 debug=False):
    """
    DESCRIPTION
    -----------
    geom, textures = geom_fromRCB(RCBFILE,
                                  output='GEOM_FILE_NAME',
                                  rim=RIM_SIZE,
                                  thickness=NUM_VOXEL_Z_DIRECTION,
                                  step=MESH_RESOLUTION_IN_um)
        Generate a columnar microstructure based on given surface OIM
    data (RCBFILE: reconstructed boundary file), optional geom file and
    material configuration file can be auto generated during the process.
    PARAMETERS
    ----------
    RCBFILE:   str
        The path to reconstructed boundary file.
    output:    str
        File name for the geom file to be used with spectral solver.
    rim:       int
        Rim/Pan used to force periodic microstructure which is required by
        the spectral solver. Default value is set to 1, i.e.
    thickness: int
        Thickness along the sample z direction (depth). Since no subsurface
        information is available, the thickness of each grain is arbitrarily
        defined through this option.
    step:      int
        Spectral mesh resolution. Default value is set to 1, i.e. the smallest
        representation of the microstructure is a 1x1x1 um box.
    debug:     boolean
        Control debugging output to terminal.
    RETURNS
    -------
    geom:      numpy.array
        A 3D numpy array consists of microstructure IDs (grain ID).
    textures:  dict
        A dictionary keep tracks of grain ID and the corresponding crystal
        orientation in Bunge Euler angles.
    NOTES
    -----
    --> Standard header of the reconstructed grain boundary file from TSL
        software
    # Column 1-3:   right hand average orientation (phi1, PHI, phi2 in radians)
    # Column 4-6:   left hand average orientation (phi1, PHI, phi2 in radians)
    # Column 7:     length (in microns)
    # Column 8:     trace angle (in degrees)
    # Column 9-12:  x,y coordinates of endpoints (in microns)
    # Column 13-14: IDs of right hand and left hand grains
    --> Figuring out which grain are at the corner of the map:
    For each corner, we can computer the distance between the corner and the
    center of each grain. The grain with the smallest the distance to this
    corner should be one containing the corner vertex.
    """
    with open(rcbFile) as f:
        rawstr = f.readlines()[8:]
        data = [[float(item) for item in line.split()] for line in rawstr]
    # figure out patch size
    xmax = max(max(data[:, 8]), max(data[:, 10]))
    ymax = max(max(data[:, 9]), max(data[:, 11]))
    corners = [Point2D(0.0, 0.0),
               Point2D(0.0, ymax),
               Point2D(xmax, 0.0),
               Point2D(xmax, ymax)]
    # total number of grains
    # --> grain_1 is reserved for rim, thus offset all OIM ID by 1
    # --> grains[0] is an empty polygon, a place holder for rim
    ng = max(max(data[:, 12]), max(data[:, 13])) + 1
    grains = [Polygon2D() for i in range(ng)]
    # texture book keeping
    # --> textures = {ID: orientation}
    textures = {}
    textures[1] = None
    # parsing each grain boundary
    for row in data:
        l_O = tuple(np.rad2deg(row[0:3]))
        r_O = tuple(np.rad2deg(row[3:6]))
        vtx1 = Point2D(row[8], row[9])
        vtx2 = Point2D(row[10], row[11])
        l_gid = int(row[13]) + 1
        r_gid = int(row[12]) + 1
        # parse left grain
        if l_gid not in textures.keys():
            textures[l_gid] = l_O
        if r_gid not in textures.keys():
            textures[r_gid] = r_O
        # add vertex to polygon
        grains[l_gid].add_vertex(vtx1)
        grains[l_gid].add_vertex(vtx2)
        grains[r_gid].add_vertex(vtx1)
        grains[r_gid].add_vertex(vtx2)

    # generate some debugging info



def geom_fromSeed():
    """
    DESCRIPTION
    -----------
    PARAMETERS
    ----------
    RETURNS
    -------
    NOTES
    ----
    """
    pass
