#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

ext = [ Extension("*", ["*.pyx"],
        include_dirs=[np.get_include()]) ]

setup(
      ext_modules=cythonize(ext)
)