EXECUTABLES = val_test01_solved val_test02_solved MMult1 omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 jacobi2D-omp gs2D-omp

all: $(EXECUTABLES)

MMult1: MMult1.cpp
	g++ -std=c++11 -fopenmp -O3 -march=native -o MMult1  MMult1.cpp

val_test01_solved: val_test01_solved.cpp
	g++ -std=c++11 -O3 -o val_test01_solved val_test01_solved.cpp
	
val_test02_solved: val_test02_solved.cpp
	g++ -std=c++11 -O3 -o val_test02_solved val_test02_solved.cpp
	
omp_solved2: omp_solved2.c
	g++ -O3 -fopenmp -o omp_solved2 omp_solved2.c
	
omp_solved3: omp_solved3.c
	g++ -O3 -fopenmp -o omp_solved3 omp_solved3.c

omp_solved4: omp_solved4.c
	g++ -O3 -fopenmp -o omp_solved4 omp_solved4.c

omp_solved5: omp_solved5.c
	g++ -O3 -fopenmp -o omp_solved5 omp_solved5.c

omp_solved6: omp_solved6.c
	g++ -O3 -fopenmp -o omp_solved6 omp_solved6.c

jacobi2D-omp: jacobi2D-omp.cpp
	g++ -std=c++11 -O3 -o jacobi2D-omp -fopenmp jacobi2D-omp.cpp

gs2D-omp: gs2D-omp.cpp
	g++ -std=c++11 -O3 -o gs2D-omp -fopenmp gs2D-omp.cpp