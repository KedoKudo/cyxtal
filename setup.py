#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


ext = [ Extension("*", ["*.pyx"], include_dirs=[np.get_include()]
                  )
      ]

setup(
      name='cyxtal',
      version=0.1,
      description='crystal analysis package (cython)',
      author='C.Z',
      author_email='chenzhang8722@gmail.com',
      ext_modules=cythonize(ext)
)