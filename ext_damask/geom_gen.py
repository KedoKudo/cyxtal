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

import vtk
import sys
import numpy as np
from cyxtal import Point2D
from cyxtal import Polygon2D


def geom_fromRCB(rcbFile,
                 output=None,
                 rim=1,
                 thickness=None,
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
        Control debugging output to terminal. Also provide VTP file for
        visual inspection.
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
    --> Algorithm
    """
    with open(rcbFile) as f:
        rawstr = f.readlines()[8:]
        data = [[float(item) for item in line.split()] for line in rawstr]
    data = np.array(data)
    # figure out patch size
    # initialize columnar grain structure
    xmax = int(max(max(data[:, 8]), max(data[:, 10])))
    ymax = int(max(max(data[:, 9]), max(data[:, 11])))
    corners = [Point2D(0.0, 0.0),
               Point2D(0.0, ymax),
               Point2D(xmax, 0.0),
               Point2D(xmax, ymax)]
    # if patch thickness is not defined, use largest possible
    if thickness is None:
        thickness = max(ymax, xmax)
    geom = np.zeros((xmax, ymax, thickness))
    # ng: total number of grains (include offset and rim)
    # --> ng is type casted to be integer
    # --> gid=1 is reserved for rim, thus offset all OIM GID by 1
    # --> grains[0] is an empty polygon, a place holder since ID=0 is not
    #     valid in DAMASK.
    # --> grains[1] is rim, which is a empty polygon
    ng = max(max(data[:, 12]), max(data[:, 13])) + 1 + 1
    ng = int(ng)
    grains = [Polygon2D() for i in range(ng)]
    # texture book keeping
    # --> textures = {ID: orientation}
    textures = {}
    textures[1] = None
    # parsing each grain boundary
    # OIM ID is offset by +1 as GID=1 is reserved for rim
    tmp_xmax = max(max(data[:, 8]), max(data[:, 10]))
    for row in data:
        l_O = tuple(np.rad2deg(row[0:3]))
        r_O = tuple(np.rad2deg(row[3:6]))
        # Due to the coordinate setting, x coordinate has to be
        # flipped.
        vtx1 = Point2D(tmp_xmax - row[8], row[9])
        vtx2 = Point2D(tmp_xmax - row[10], row[11])
        l_gid = int(row[13]) + 1
        r_gid = int(row[12]) + 1
        # update texture info
        if l_gid not in textures.keys():
            textures[l_gid] = l_O
        if r_gid not in textures.keys():
            textures[r_gid] = r_O
        # add vertex to polygon
        grains[l_gid].add_vertex(vtx1)
        grains[l_gid].add_vertex(vtx2)
        grains[r_gid].add_vertex(vtx1)
        grains[r_gid].add_vertex(vtx2)
    # Grain information
    # --> ID, (phi1, PHI, phi2), (center.x, center.y)
    if debug:
        print "Identified grains:"
        for key in textures:
            if key == 1:
                print key, textures[key]
            else:
                print key, textures[key], grains[key].center
        print
    # Add corner to correct polygon (grain)
    # NOTE:
    #   Some of the Grain (Polygon2D) will be empty one due to
    #   missing info in the RCB file. This is usually related to
    #   the cleaning up.
    gids = textures.keys()[1:]
    # Figuring out which grain are at the corner of the map:
    # For each corner, the distance between the corner and the center of each
    # grain is computed. The grain with the smallest the distance to this
    # corner should be one containing this corner vertex.
    for corner in corners:
        dist = 1e9
        mygid = -1
        for gid in gids:
            tmp = corner.dist2point(grains[gid].center)
            if tmp < dist:
                mygid = gid
                dist = tmp
        grains[mygid].add_vertex(corner)
    # Fill the geom array with corresponding grain ID
    # NOTE:
    #   This part uses nested loop, which will not work well for large
    #   dataset.
    for i in range(xmax):
        for j in range(ymax):
            coord = Point2D(i, j)
            if debug:
                print "\r", coord,
                sys.stdout.flush()
            for gid in gids:
                if grains[gid].contains_point(coord):
                    geom[i, j, :] = gid
                    break
    # Add rim to geom
    x, y, z = geom.shape
    geom_withRim = np.ones((x+rim*2, y+rim*2, z+rim))
    geom_withRim[rim:x+rim, rim:y+rim, rim:] = geom
    # Output visualization file for debugging purpose
    if debug:
        print
        print "Exporting vtp file for visualization."
        geom_viz(geom_withRim, filename='geomviz.vtp')
    # Output geom file and material configuration file if specified

    return geom_withRim, textures


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


def geom_viz(geomData, filename='sample.vtp'):
    """
    DESCRIPTION
    -----------
    geom_viz(geomData, filename='geom.vtp')
    Generate a simple point centered vtp file to visualize grain ID.
    PARAMETERS
    ----------
    geomData: numpy.array
    Grain ID populated in a numpy array representing the extruded
    microstructure.
    filename: str
    Output VTP file name.
    RETURNS
    -------
    NOTES
    ----
    """
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()

    gids = vtk.vtkFloatArray()
    gids.SetNumberOfComponents(1)
    gids.SetName("GrainID")

    # iterating through CPFFT data
    cnt = 0
    x, y, z = geomData.shape
    for i in range(x):
        for j in range(y):
            for k in range(z):
                # set position
                points.InsertNextPoint(i, j, k)
                vertex = vtk.vtkVertex()
                vertex.GetPointIds().SetId(0, cnt)
                cnt += 1
                vertices.InsertNextCell(vertex)
                # set Grain ID
                gids.InsertNextTuple1(geomData[i, j, k])
    # finish the vtp object
    polydata.SetPoints(points)
    polydata.SetVerts(vertices)
    polydata.GetPointData().SetScalars(gids)
    polydata.Modified()

    if vtk.VTK_MAJOR_VERSION <= 5:
        polydata.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)

    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)

    writer.Write()
