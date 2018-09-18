# CYXTAL v2.0

Copyright (c) 2015-2018, C. Zhang.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1) Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2) Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Quick start

This Python package consists of several modules that written in Cython for data processing related to crystallography.
To install this packge, first install Cython on your machine

```bash
# use pip
pip install Cython

# or use anaconda
conda install Cython
```

then clone this repository to your local machine with the following command

```bash
git clone https://github.com/KedoKudo/cyxtal.git
```

Now nagivate to the repository location and compile the Cython based C-extension using

```bash
# compile Cython code to c-extention
make all

# if desired, install to your Python site-package
make install
```

It is highly recommended to run `make test` before using this package for any data processing to ensure all modules are functioning properly.

## Pacakge overview

TODO: describe the package structure

## Examples

### Calculate average crystal orientations

### Calculate Schmid factors

### Calculate

## NOTE

The old version of cyxtal (v1.0) is available as a separate branch.
However, it is no longer maintained by the developer, and only works with **Python 2.x**.
The develop recomend users to update their code using the new API for better integration with the general **Python 3.x** environment.