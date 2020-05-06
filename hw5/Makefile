EXECUTABLES = jacobi2D-mpi

all: clean $(EXECUTABLES)
#all: clean

jacobi2D-mpi: jacobi2D-mpi.cpp
	mpic++ -std=c++11 -O3 -march=native -o jacobi2D-mpi jacobi2D-mpi.cpp

clean: 
	rm -f jacobi2D-mpi