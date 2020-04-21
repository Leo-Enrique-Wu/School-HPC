EXECUTABLES = InnerProduct jacobi2D

all: clean $(EXECUTABLES)
#all: clean

InnerProduct: InnerProduct.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -o InnerProduct InnerProduct.cu

jacobi2D: jacobi2D.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -o jacobi2D jacobi2D.cu

clean: 
	rm -f InnerProduct jacobi2D