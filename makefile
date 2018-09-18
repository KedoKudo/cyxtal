SHELL = /bin/sh
CC    = gcc

.PHONY: all install test doc sweep clean list

all:
	@echo "Build C-extensions from Cython"
	python setup.py build_ext --inplace

install:
	@echo "install Cyxtal to site-packages"

test:
	@echo "Perform predefined test"
	nosetests -v tests

doc:
	@echo "Generate local documentation using Doxgen"
	doxygen doxygen.config
	make -C documentation

sweep:
	@echo "remove build folders and intermdeia files"
	rm -rvf build

clean:
	@echo "Remove build folder and binraries"
	rm -rvf build *.so

list:
	@echo "LIST OF TARGETS:"
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null \
	| awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' \
	| sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$' | xargs