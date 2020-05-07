EXECUTABLES = jacobi2D-mpi ssort

all: clean $(EXECUTABLES)
#all: clean

jacobi2D-mpi: jacobi2D-mpi.cpp
	mpic++ -std=c++11 -O3 -o jacobi2D-mpi jacobi2D-mpi.cpp

ssort: ssort.cpp
	mpic++ -std=c++11 -O3 -o ssort ssort.cpp

clean: 
	rm -f jacobi2D-mpi ssort