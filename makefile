CC=gcc-5

#########################
# COMPILE CYTHON MODULE #
#########################
all:
	python setup.py build_ext --inplace


#######################
# COMPILE AND INSTALL #
#######################
install:
	python setup.py build_ext install


############
# CLEAN UP #
############
clean:
	rm -rvf build