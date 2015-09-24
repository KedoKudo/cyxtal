CC=gcc-5

#########################
# COMPILE CYTHON MODULE #
#########################
all:
	python setup.py build_ext --inplace


############
# CLEAN UP #
############
clean:
	rm -rvf build
	rm *.c