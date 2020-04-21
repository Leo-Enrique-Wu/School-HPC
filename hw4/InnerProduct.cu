#include <math.h>
#include <stdio.h>
#include "utils.h"
#include <omp.h>
#include <string>

using std::string;

// g++ -std=c++11 -O3 -march=native -fopenmp -o InnerProduct InnerProduct.cpp
// g++-9 -std=c++11 -O3 -march=native -fopenmp -o InnerProduct InnerProduct.cpp
// nvcc -std=c++11 -Xcompiler -fopenmp -o InnerProduct InnerProduct.cu

// inner product between two matrixes
// matrix A: m * n. If m = 1, then it's a vector.
// matrix B: n * l. If l = 1, then it's a vector.

#define OMP_BLOCK_SIZE 128
#define GPU_GROUP_NUMBER 1024

long m = 400;
long n = 4000;
long l = 400;

int parallelThreadNum = 10;

void innerProductSeqencial(double *input_A, double*input_B, double *output_C){
  
  
    for(long i_m = 0; i_m < m; i_m++){
      for(long i_l = 0; i_l < l; i_l++){
      
        for(long i_n = 0; i_n < n; i_n++){
          
          long idx_C = (i_m * l) + i_l;
          long idx_A = (i_m * n) + i_n;
          long idx_B = (i_n * l) + i_l;
          output_C[idx_C] += (input_A[idx_A] * input_B[idx_B]);
          
        }
      
      }
    }
  
}

void innerProductCpuParallel(double *a, double*b, double *c){
  
   #pragma omp parallel num_threads(parallelThreadNum)
   #pragma omp for collapse(2)
   for (long i = 0; i < m; i+= OMP_BLOCK_SIZE) {
     
     for (long j = 0; j < l; j+= OMP_BLOCK_SIZE) {
       
       long blockCeiling_ii = ((i + OMP_BLOCK_SIZE) > m)? m : i + OMP_BLOCK_SIZE;
       long blockCeiling_jj = ((j + OMP_BLOCK_SIZE) > l)? l : j + OMP_BLOCK_SIZE;
       
       // initialize tempC(temp_range_i * temp_range_j)
       long temp_range_i = blockCeiling_ii - i;
       long temp_range_j = blockCeiling_jj - j;
       long tempC_range = temp_range_i * temp_range_j;
       // printf("tempC_range=%ld\n", tempC_range);
       double tempC[tempC_range];
       for(long c_idx = 0; c_idx < tempC_range; c_idx++) tempC[c_idx] = 0;
       
       // load C to block tempC
       for(long jj = j; jj < blockCeiling_jj; jj++){
         for(long ii = i; ii < blockCeiling_ii; ii++){
           
           long c_idx = (jj - j) + ((ii - i) * temp_range_j);
           tempC[c_idx] += c[jj + ii * l];
           
         }
       }
       
       for (long p = 0; p < n; p+= OMP_BLOCK_SIZE) {
         
         // printf("p=%ld\n", p);
  
         long blockCeiling_pp = ((p + OMP_BLOCK_SIZE) > n)? n : p + OMP_BLOCK_SIZE;
         long temp_range_p = blockCeiling_pp - p;
         
         // initialize tempA
         long tempA_range = temp_range_i * temp_range_p;
         double tempA[tempA_range];
         for(long a_idx = 0; a_idx < tempA_range; a_idx++) tempA[a_idx] = 0;
         
         // load block tempA
         for(long pp = p; pp < blockCeiling_pp; pp++){
           for(long ii = i; ii < blockCeiling_ii; ii++){
             
             long a_idx = (pp - p) + ((ii - i) * temp_range_p);
             tempA[a_idx] += a[pp + ii * n];
             
           }
         }
         // printf("tempA_range: %ld * %ld\n", temp_range_i, temp_range_p);
         
         // initialize tempB
         long tempB_range = temp_range_p * temp_range_j;
         double tempB[tempB_range];
         for(long b_idx = 0; b_idx < tempB_range; b_idx++) tempB[b_idx] = 0;
         
         // load block tempB
         for(long jj = j; jj < blockCeiling_jj; jj++){
           for(long pp = p; pp < blockCeiling_pp; pp++){
             
             long b_idx = (jj - j) + ((pp - p) * temp_range_j);
             tempB[b_idx] += b[jj + pp * l];
             
           }
         }
         // printf("tempB_range: %ld * %ld\n", temp_range_p, temp_range_j);
         
         // B x B mini matrix multiplications
         for(long temp_p = 0; temp_p < temp_range_p; temp_p++){
           for(long temp_j = 0; temp_j < temp_range_j; temp_j++){
             for(long temp_i = 0; temp_i < temp_range_i; temp_i++){
               
               double A_ip = tempA[temp_p + temp_i * temp_range_p];
               double B_pj = tempB[temp_j + temp_p * temp_range_j];
               double C_ij = tempC[temp_j + temp_i * temp_range_j];
               C_ij = C_ij + A_ip * B_pj;
               tempC[temp_j + temp_i * temp_range_j] = C_ij;
               
               /*
               long real_m = i + temp_i;
               long real_n = p + temp_p;
               long real_l = j + temp_i;
               */
               // printf("A[%ld][%ld] * B[%ld][%ld] = %10f * %10f = %10f\n", real_m, real_n, real_n, real_l, A_ip, B_pj, A_ip * B_pj);
               
             }
           }
         }
         
       }
       
       // store block tempC
       for(long jj = j; jj < blockCeiling_jj; jj++){
         for(long ii = i; ii < blockCeiling_ii; ii++){
           
           long c_idx = (jj - j) + ((ii - i) * temp_range_j);
           c[jj + ii * l] = tempC[c_idx];
           
         }
       }
     
     }
   }
  
}

__global__
void innerProduct_GPU_kernel(const double* a, const double* b, double* c, long m, long l, long n, long N){

  int idx_m = blockIdx.x;
  int idx_l = blockIdx.y;
  int idx = idx_m * l + idx_l;
  __shared__ double tempInnerProduct[GPU_GROUP_NUMBER];

  tempInnerProduct[threadIdx.x] = 0;
  long chuckSize = n / GPU_GROUP_NUMBER;
  long idx_n_start = threadIdx.x * chuckSize;
  long idx_n_end = (threadIdx.x == GPU_GROUP_NUMBER - 1)? n: (threadIdx.x + 1) * chuckSize;
  
  /*
  if(idx == 0){
    printf("idx_n_start=%ld, idx_n_end=%ld\n", idx_n_start, idx_n_end);
  }
  */
  
  for(long idx_n = idx_n_start; idx_n < idx_n_end; idx_n++){
    double a_idx_m_idx_n = a[(idx_m * n) + idx_n];
    double b_idx_n_idx_l = b[(idx_n * l) + idx_l];
    tempInnerProduct[threadIdx.x] += (a_idx_m_idx_n * b_idx_n_idx_l);
    // printf("a[%d][%ld] * b[%ld][%d] = %10f\n", idx_m, idx_n, idx_n, idx_l, (a_idx_m_idx_n * b_idx_n_idx_l));
  }
  /*
  if(idx == 0){
    printf("threadIdx.x=%d\n", threadIdx.x);
  }
  */
  __syncthreads ();
  
  
  for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if(threadIdx.x < s) {
      tempInnerProduct[threadIdx.x] += tempInnerProduct[threadIdx.x + s];
      /*
      if(idx == 0){
        printf("tempInnerProduct[%d] += tempInnerProduct[%d]\n", threadIdx.x, threadIdx.x + s);
      }
      */
    }
    __syncthreads ();
  }
  
  if(threadIdx.x == 0){
    c[idx] = tempInnerProduct[threadIdx.x];
  }
    
}

int main(int argc, char *argv[]) {

  // matrix presentation: row major
  
  // init
  // update parameters if have input parameters
  if(argc >= 5){
    
    string threadNumStr = argv[1];
    parallelThreadNum = stoi(threadNumStr);
    
    string mStr = argv[2];
    m = stoi(mStr);
    
    string nStr = argv[3];
    n = stoi(nStr);
    
    string lStr = argv[4];
    l = stoi(lStr);
    
  }
  printf("m=%ld, n=%ld, l=%ld\n", m, n, l);
  
  // allocate for input matrix A, B and output matrix C(for GPU) and C_ref(for CPU)
  double *A = (double*) aligned_malloc(m * n * sizeof(double));
  double *B = (double*) aligned_malloc(n * l * sizeof(double));
  // double *C = (double*) aligned_malloc(m * l * sizeof(double));
  double *C_ref = (double*) aligned_malloc(m * l * sizeof(double));
  double *C_GPU = (double*) aligned_malloc(m * l * sizeof(double));

  // initialize matrixes
  printf("Start to initialize\n");
  #pragma omp parallel num_threads(parallelThreadNum)
  {
    
    #pragma omp for
    for(long i_n = 0; i_n < n; i_n++){
    
      for(long i_m = 0; i_m < m; i_m++){
        long idx_A = (i_m * n) + i_n;
        A[idx_A] = drand48();
      }
    
      for(long i_l = 0; i_l < l; i_l++){
        long idx_B = (i_n * l) + i_l;
        B[idx_B] = drand48();
      }
    
    }
  
    #pragma omp for
    for(long i_C_row = 0; i_C_row < m; i_C_row++){
      for(long i_C_col = 0; i_C_col < l; i_C_col++){
        long idx_C = (i_C_row * l) + i_C_col;
        // C[idx_C] = 0;
        C_ref[idx_C] = 0;
        C_GPU[idx_C] = 0;
      }
    }
    
  }
  printf("End of initializing\n");
  
  /*
  printf("\nA:\n");
  for(long i_row = 0; i_row < m; i_row++){
    for(long i_col = 0; i_col < n; i_col++){
      long idx = (i_row * n) + i_col;
      printf("%10f", A[idx]);
      if(i_col == (n - 1)){
        printf("\n");
      }
    }
  }
  
  printf("\nB:\n");
  for(long i_row = 0; i_row < n; i_row++){
    for(long i_col = 0; i_col < l; i_col++){
      long idx = (i_row * l) + i_col;
      printf("%10f", B[idx]);
      if(i_col == (l - 1)){
        printf("\n");
      }
    }
  }
  printf("\n");
  */
  
  
  double start, end;
  /*
  double start = omp_get_wtime();
  innerProductSeqencial(A, B, C_ref);
  double end = omp_get_wtime();
  printf("innerProductSeqencial: %10f\n", end - start);
  */
  
  start = omp_get_wtime();
  innerProductCpuParallel(A, B, C_ref);
  end = omp_get_wtime();
  printf("innerProductCpuParallel: %10f\n", end - start);
  
  /*
  double max_err = 0;
  for (long i = 0; i < m * l; i++){
    max_err = std::max(max_err, fabs(C[i] - C_ref[i]));
  }
  printf("max_err: %10e\n", max_err);
  */
  
  /*
  for(long i_C_row = 0; i_C_row < m; i_C_row++){
    for(long i_C_col = 0; i_C_col < l; i_C_col++){
      long idx_C = (i_C_row * l) + i_C_col;
      printf("%10f", C[idx_C]);
      if(i_C_col == (l - 1)){
        printf("\n");
      }
    }
  }
  */
  
  // GPU version start
  // initialize
  double *A_d, *B_d, *C_GPU_d;
  cudaMalloc(&A_d, m * n * sizeof(double));
  cudaMalloc(&B_d, n * l * sizeof(double));
  cudaMalloc(&C_GPU_d, m * l * sizeof(double));
  long N = 10000;
  
  start = omp_get_wtime();
  // double tempStart = omp_get_wtime();
  cudaMemcpy(A_d, A, m * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, n * l * sizeof(double), cudaMemcpyHostToDevice);
  // double tempEnd = omp_get_wtime();
  // printf("innerProduct_GPU_kernel-cudaMemcpy(HostToDevice): %10f\n", tempEnd - tempStart);
  
  // tempStart = omp_get_wtime();
  dim3 GridDim(m , l);
  innerProduct_GPU_kernel<<<GridDim, GPU_GROUP_NUMBER>>>(A_d, B_d, C_GPU_d, m, l, n, N);
  cudaDeviceSynchronize();
  // tempEnd = omp_get_wtime();
  // printf("innerProduct_GPU_kernel: %10f\n", tempEnd - tempStart);
  
  // tempStart = omp_get_wtime();
  cudaMemcpy(C_GPU, C_GPU_d, m * l * sizeof(double), cudaMemcpyDeviceToHost);
  // tempEnd = omp_get_wtime();
  // printf("innerProduct_GPU_kernel-cudaMemcpy(DeviceToHost): %10f\n", tempEnd - tempStart);
  
  end = omp_get_wtime();
  printf("innerProduct_GPU_kernel: %10f\n", end - start);
  
  double doubleNum = ((double) 2 * (l * m * n) + (m * l));
  double memoryBandwidth = (doubleNum * sizeof(double) / 1e9 ) / (end - start);
  printf("GPU Bandwidth = %f GB/s\n", memoryBandwidth);
  
  double err = 0;
  // printf("m*l = %ld\n", m * l);
  for (long i = 0; i < m * l; i++) {
    // printf("i=%ld, C_GPU[i]=%10f, C_ref[i]=%10f\n", i, C_GPU[i], C_ref[i]);
    err += (fabs(C_GPU[i] - C_ref[i]));
  }
  printf("Total Error = %10e\n", err);
  
  double max_err = 0;
  for (long i = 0; i < m * l; i++){
    max_err = std::max(max_err, fabs(C_GPU[i] - C_ref[i]));
  }
  printf("max_err: %10e\n", max_err);

  // free memory before terminate
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_GPU_d);
  
  aligned_free(A);
  aligned_free(B);
  // aligned_free(C);
  aligned_free(C_ref);
  aligned_free(C_GPU);
  
}

