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
python geom_fromRCB.py reconstrcuted_boundary_file.rcb.
    Generate a geom file for the spectral solver using reconstructed
    boundary file from TSL software.
"""

import sys
import argparse
import numpy as np
from cyxtal import Point2D
from cyxtal import Polygon2D

# import os, sys, argparse
# import h5py
# from   cyxtal.ext_damask import MPIE_spectralOut
# from   cyxtal.ext_damask import MPIE_marc


# ##
# # PARSER
# parser = argparse.ArgumentParser(prog='sepctral2HDF5',
#                                  epilog='require h5py and numpy',
#                                  description='convert binary results to hdf5.')
# parser.add_argument('sourceFile',
#                     help='Binary output file from DAMASK_spectral',
#                     default='run.spectralOut')
# parser.add_argument('-v', '--version',
#                     action='version',
#                     version="%(prog)s 0.1")
# parser.add_argument('-o', '--output',
#                     help='output file name.'
#                     default=None)
# parser.add_argument('-d', '--debug',
#                     action='store_true',
#                     default=False,
#                     help='toggle debug output on terminal.')
# parser.add_argument('-s', '--silent',
#                     action='store_true',
#                     default=False)
# args = parser.parse_args()
# if not args.silent:
#     print "*"*20
#     parser.parse_args()
#     print "*"


# ##
# # START PROCESSING
# rst_f = args.sourceFile
# try:
#     tmp_f = open(rst_f, "rb")
#     tmp_f.close()
# except:
#     raise ValueError("Invalid spectralOut file: {}".format(rst_f))

# # Invoke the auxMPIE interface for data extraction
# rst_ext = rst_f.split(".")[-1]
# if rst_ext == "spectralOut":
#     converter = MPIE_spectralOut(rst_f)
# elif rst_ext == "t16":
#     converter = MPIE_marc(rst_f)
# else:
#     raise ValueError("Unsupported type: *.{}".format(rst_ext))

# # Setting properties for output file
# outFileName = args.output
# if outFileName is None:
#     outFileName = rst_f.replace(".spectralOut", ".hdf5")
# converter.convert(outFileName)

# # Finishing up
# if not args.silent:
#     print "All done, output HDF5 file is: {}".format(outFileName)