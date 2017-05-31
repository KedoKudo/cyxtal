CYXTAL
=====
=

Copyright (c) 2015-2017, C. Zhang.
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


# PACKAGE DESCRIPTION
This package consists of several data processing tools supplementing CPFE/CPFFT
analysis with DAMASK in conjunction with APS-34-IDE.

##################
# FILE STRUCTURE #
##################
cyxtal/
    __init__.py
    corientation.pyd
    corientation.pyx
    ctools.pyd
    ctools.pyx
    ext_aps/
        parses.py
    ext_damask/
        nil
    ext_vtk/
        nil
    ext_ebsd/
        snil
    tests/
        test_ctools.py
        test_cxtallite.py
        test_exdamask_ascii2table.py

###############
# INSTRUCTION #
###############
To use cyxtal, add it to ${PYTHON}/site-packages or other location in
${PYTHONPATH}.
To use cython based module, run `make all` from command window (cython
and gcc compiler required).
To test modification, run `make test` from command window (nosetests
required). Alternatively, each module can be tested individually by executing
corresponding testing script located in ./tests .
