#!/usr/bin/env python
# encoding: utf-8
# filename: corientation.pyx


##
# NOTE: Start over of the corientation, this time
#       we will design the interface of this class

import math, random, os
import numpy as np
cimport numpy as np


cdef Symmetry(lattice):
    """
    """
    lattice = lattice.lower()
    if lattice == 'cubic':
        symQuats = [
                    [ 1.0,              0.0,              0.0,              0.0              ],
                    [ 0.0,              1.0,              0.0,              0.0              ],
                    [ 0.0,              0.0,              1.0,              0.0              ],
                    [ 0.0,              0.0,              0.0,              1.0              ],
                    [ 0.0,              0.0,              0.5*math.sqrt(2), 0.5*math.sqrt(2) ],
                    [ 0.0,              0.0,              0.5*math.sqrt(2),-0.5*math.sqrt(2) ],
                    [ 0.0,              0.5*math.sqrt(2), 0.0,              0.5*math.sqrt(2) ],
                    [ 0.0,              0.5*math.sqrt(2), 0.0,             -0.5*math.sqrt(2) ],
                    [ 0.0,              0.5*math.sqrt(2),-0.5*math.sqrt(2), 0.0              ],
                    [ 0.0,             -0.5*math.sqrt(2),-0.5*math.sqrt(2), 0.0              ],
                    [ 0.5,              0.5,              0.5,              0.5              ],
                    [-0.5,              0.5,              0.5,              0.5              ],
                    [-0.5,              0.5,              0.5,             -0.5              ],
                    [-0.5,              0.5,             -0.5,              0.5              ],
                    [-0.5,             -0.5,              0.5,              0.5              ],
                    [-0.5,             -0.5,              0.5,             -0.5              ],
                    [-0.5,             -0.5,             -0.5,              0.5              ],
                    [-0.5,              0.5,             -0.5,             -0.5              ],
                    [-0.5*math.sqrt(2), 0.0,              0.0,              0.5*math.sqrt(2) ],
                    [ 0.5*math.sqrt(2), 0.0,              0.0,              0.5*math.sqrt(2) ],
                    [-0.5*math.sqrt(2), 0.0,              0.5*math.sqrt(2), 0.0              ],
                    [-0.5*math.sqrt(2), 0.0,             -0.5*math.sqrt(2), 0.0              ],
                    [-0.5*math.sqrt(2), 0.5*math.sqrt(2), 0.0,              0.0              ],
                    [-0.5*math.sqrt(2),-0.5*math.sqrt(2), 0.0,              0.0              ],
                   ]
    elif lattice == 'hexagonal':
        pass
    return symQuats




cdef class Orientation:
    pass

cdef class crystallite:
    cdef public Orientation orientation

    def __init__(self, orientation, symmetry):
        pass