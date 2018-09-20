# use GNU compiler
cc = g++

# export env var
export CPPFLAGS = -Wall -Werror -fpic
export OMP_NUM_THREADS=2

# build targets 

orientation.o:
	$(cc) -c orientation.cpp

testomp:
	$(cc) -g \
	      -Xpreprocessor \
		  -fopenmp -lomp \
		  -o test_kinematics.out test_kinematics.cpp
	./test_kinematics.out

test: orientation.o
	$(cc) -c test_orientation.cpp
	$(cc) orientation.o test_orientation.o -o test_orientation.out
	./test_orientation.out

clean:
	rm -rvf *.o *.a *.so *.dSYM *.out