#!/usr/bin/env python
# -*- coding=utf-8 -*-


"""
Generate list of HKL indeices that can be used for virutal diffraction exp.

Usage:
    genHKLlist.py    (LATTICE)  [-v]
    genHKLlist.py    -h | --help
    genHKLlist.py    --version

Options:
    -h --help            Show this screen.
    --version            Show version.
    -v --verbose         Detalied output.
"""

import numpy as np
import pandas as pd
from docopt import docopt
from itertools import product


def genHKLS(lattice):
    """
    hkls, dists = genHKLS("FCC")
    NOTE: Analysis of real DAXM shows the hkl range should be between 
    -20 ~ 20, with a strong focuse on planes with lower index. 
    """
    if lattice.upper() == 'FCC':
        tmp = np.arange(-20, 20, 2)
        hkls_even = [me for me in product(tmp, repeat=3)]
        hkls_even.remove((0, 0, 0))
        tmp = np.arange(-19, 19, 2)
        hkls_odd = [me for me in product(tmp, repeat=3)]
        hkls = hkls_even + hkls_odd
    elif lattice.upper() == 'BCC':
        raise NotImplementedError
    elif lattice.upper() == 'HCP':
        raise NotImplementedError
    else:
        raise ValueError("Unknown lattice specified: {}".format(lattice))
    
    # do not prun out high index peak as the process would also elimenate 
    # parallel low index q point opposite direction.

    # assume that the intensity probability is inversely proportional to
    # interplanar spacing
    dists = map(lambda x: 1.0/np.linalg.norm(x), hkls)
    dists = [dist/sum(dists) for dist in dists]
    return hkls, dists


if __name__ == "__main__":
    # parse interface
    ARGS = docopt(__doc__, version="1.0.0")

    xtalLattice = "FCC" if ARGS["LATTICE"] is None else ARGS["LATTICE"]

    if ARGS["--verbose"]:
        print("Generate HKL list for {} xtal.".format(xtalLattice)) 

    # populate HKL based on pure index range.
    hkls, dists = genHKLS(xtalLattice)

    # write to file 
    with open('hkls_{}.txt'.format(xtalLattice.upper()), 'w') as f:
        print('use relative interplanar spacing as weight')
        txtstr = 'h\tk\tl\twgt\n'
        txtstr += '\n'.join(['\t'.join(map(str, me[0])) + '\t{}'.format(me[1]) 
                             for me in zip(hkls, dists)])
        f.write(txtstr)
