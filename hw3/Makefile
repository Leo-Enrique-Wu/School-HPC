CXX = g++-9
EXECUTABLES = fast-sin omp-scan

all: clean $(EXECUTABLES)
#all: clean

fast-sin: fast-sin.cpp
	$(CXX) -std=c++11 -O3 -ftree-vectorize -mavx -march=native -fopenmp -o fast-sin fast-sin.cpp

omp-scan: omp-scan.cpp
	$(CXX) -std=c++11 -O3 -march=native -fopenmp -o omp-scan omp-scan.cpp

clean: 
	rm -f *.o fast-sin omp-scan