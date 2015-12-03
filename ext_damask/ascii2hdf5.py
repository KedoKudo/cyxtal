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
    tableConvert:
        convert a single damask table into a hdf5 file
    tableMerge:
        merge several hdf5 table into one single large file

NOTE
----
    Ideally, the output from DAMASK should be directly in HDF5 format, which can help
    speed up the post processing phase. However, since no immediate plan is made in
    system wide implementation of HDF5 in DAMASK, we are still using the legacy combo
    binary+ascii. This script is design to convert ASCII table from DAMASK into hdf5
    table, which is faster when large amount of data is being processed.
"""

import h5py
import numpy as np

def tableConvert(damaskTalbe,
                 hdf5table=None, mode=None):
    """
    DESCRIPTION
    -----------
    tableConvert(PATH_TO_DAMASK_ASCII_TALBE,
                 hdf5table=MY_HDF5_NAME)
    create a
    """
    if hdf5table is None:
        hdf5table = damaskTalbe.replace(".txt", ".hdf5")

    mytable = h5py.file(hdf5table, "w")

def table