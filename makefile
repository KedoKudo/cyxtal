CC=gcc-5

#########################
# COMPILE CYTHON MODULE #
#########################
all:
	python setup.py build_ext --inplace

clean:
	rm -rvf build *.so