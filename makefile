SHELL = /bin/sh
CC    = gcc-5

#########################
# COMPILE CYTHON MODULE #
#########################
all:
	python setup.py build_ext --inplace

test:
	nosetests -v tests

clean:
	rm -rvf build *.so