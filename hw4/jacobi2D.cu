#include <iostream>
#include <string>
#include <stdio.h>
#include <sstream>
#include <cmath>
#include "utils.h"
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using std::string;

// if after iteration is over this number, we can't get the residual decreased by a factor n, then stop the program.
const int stopAfterIterNum = 5000;

// common utils
std::ostringstream strs;

// g++ -std=c++11 -O3 -o jacobi2D-omp jacobi2D-omp.cpp
// g++-9 -std=c++11 -O3 -o jacobi2D -fopenmp jacobi2D.cpp
// nvcc -std=c++11 -Xcompiler -fopenmp -o jacobi2D jacobi2D.cu
// Without parallel: N=10, run time =   0.011500 second(s).

int calcRowMajorIndex(int i, int j, int columnSize){
  int idx = (j + ((i - 1) * columnSize)) - 1;
  return idx;
}

__device__ int calcRowMajorIndex_d(int i, int j, int columnSize) {
  int idx = j + (i * columnSize);
  // printf("i=%d, j=%d, columnSize=%d, idx=%d\n", i, j, columnSize, idx);
  return idx;
}

void calculate_next_u(int n, long range, double h, double* arr_u_k, double* arr_f, double* arr_u_k_plus_1){
  
  // use the Jacobi method
  // calculate arr_u_1
  
  /*
  for(int i = 1; i <= n; i++){
    for(int j = 1; j <= n; j++){
      int idx_i_j = calcRowMajorIndex(i, j, n);
      printf("u_k[i=%d][j=%d]=%10f\n", i, j, arr_u_k[idx_i_j]);
    }
  }
  */
  
  for(int i = 1; i <= n; i++){
     
    #pragma omp parallel for
      for(int j = 1; j <= n; j++){
        
        double f_i_j = 0;
        double u_k_im1_j = 0;
        double u_k_i_jm1 = 0;
        double u_k_ip1_j = 0;
        double u_k_i_jp1 = 0;
        
        int idx_i_j = calcRowMajorIndex(i, j, n);
        f_i_j = arr_f[idx_i_j];
        
        if(i - 1 > 0){
          int idx_im1_j = calcRowMajorIndex(i - 1, j, n);
          u_k_im1_j = arr_u_k[idx_im1_j];
          // printf("arr_u_k[%ld]=%10f\n", idx_im1_j, u_k_im1_j);
        }
        
        if(j - 1 > 0){
          int idx_i_jm1 = calcRowMajorIndex(i, j - 1, n);
          u_k_i_jm1 = arr_u_k[idx_i_jm1];
        }
        
        if(i < n){
          int idx_ip1_j = calcRowMajorIndex(i + 1, j, n);
          u_k_ip1_j = arr_u_k[idx_ip1_j];
          // printf("arr_u_k[%ld]=%10f\n", idx_ip1_j, u_k_ip1_j);
        }
        
        if(j < n){
          int idx_i_jp1 = calcRowMajorIndex(i, j + 1, n);
          u_k_i_jp1 = arr_u_k[idx_i_jp1];
          // printf("arr_u_k[%ld]=%10f\n", idx_i_jp1, u_k_i_jp1);
        }
        
        arr_u_k_plus_1[idx_i_j] = (std::pow(h, 2) * f_i_j + u_k_im1_j + u_k_i_jm1 + u_k_ip1_j + u_k_i_jp1) / 4;
        
        // printf("i=%d, j=%d, u(k+1)_i_j=%10f, f_i_j=%10f, u_k_im1_j=%10f, u_k_i_jm1=%10f, u_k_ip1_j=%10f, u_k_i_jp1=%10f\n", i, j, arr_u_k_plus_1[idx_i_j], f_i_j, u_k_im1_j, u_k_i_jm1, u_k_ip1_j, u_k_i_jp1);
      
      }
    
  }
  
}

double calculateResidualNorm(int n, long range, double h, double* arr_u, double* arr_f){
  
  // calculate residual matrix A * u(k) - f
  double norm;
  double residualSqSum = 0;
  
  
  for(int i = 1; i <= n; i++){
    
    #pragma omp parallel for reduction(+: residualSqSum)
    for(int j = 1; j <= n; j++){
      
      double f_i_j = 0;
      double u_k_im1_j = 0;
      double u_k_i_jm1 = 0;
      double u_k_i_j = 0;
      double u_k_ip1_j = 0;
      double u_k_i_jp1 = 0;
      
      int idx_i_j = calcRowMajorIndex(i, j, n);
      f_i_j = arr_f[idx_i_j];
      u_k_i_j = arr_u[idx_i_j];
      // printf("[i=%d][j=%d] u_k_i_j=%10f\n", i, j, u_k_i_j);
      
      if((i - 1) > 0){
        int idx_im1_j = calcRowMajorIndex(i - 1, j, n);
        u_k_im1_j = arr_u[idx_im1_j];
        // printf("[i=%d][j=%d] u_k_im1_j=%10f\n", i, j, u_k_im1_j);
      }
      
      if((j - 1) > 0){
        int idx_i_jm1 = calcRowMajorIndex(i, j - 1, n);
        u_k_i_jm1 = arr_u[idx_i_jm1];
        // printf("[i=%d][j=%d] u_k_i_jm1=%10f\n", i, j, u_k_i_jm1);
      }
      
      if(i < n){
        int idx_ip1_j = calcRowMajorIndex(i + 1, j, n);
        u_k_ip1_j = arr_u[idx_ip1_j];
        // printf("[i=%d][j=%d] u_k_ip1_j=%10f\n", i, j, u_k_ip1_j);
      }
      
      if(j < n){
        int idx_i_jp1 = calcRowMajorIndex(i, j + 1, n);
        u_k_i_jp1 = arr_u[idx_i_jp1];
        // printf("[i=%d][j=%d] u_k_i_jp1=%10f\n", i, j, u_k_i_jp1);
      }
      
      // residual = f_i_j - ((- u_im1_j - u_i_jm1 + 4 u_i_j - u_ip1_j - u_i_jp1)/(h ^ 2))
      double a_mult_u = (-1 * u_k_im1_j - u_k_i_jm1 + 4 * u_k_i_j - u_k_ip1_j - u_k_i_jp1) / (std::pow(h, 2.0));
      double residual = f_i_j - a_mult_u;
      // printf("res[%ld][%ld]=%10f\n", i, j, residual);
      residualSqSum += std::pow(residual, 2);
      
    }
  }
  
  norm = std::sqrt(residualSqSum);
  return norm;
  
}

void processNextIter(int iterNumber, double initialNorm, int n, long range, double h, double* arr_u_k, double* arr_f, double* arr_u_k_plus_1){
  
  calculate_next_u(n, range, h, arr_u_k, arr_f, arr_u_k_plus_1);
  double thisNorm = calculateResidualNorm(n, range, h, arr_u_k_plus_1, arr_f);
  double decreasingFactor = initialNorm / thisNorm;
  
  /*
  std::cout << "Iter[";
  std::cout << iterNumber;
  std::cout << "]:norm=";
  std::cout << thisNorm;
  std::cout << ", decreasingFactor=";
  std::cout << decreasingFactor;
  std::cout << "\n";
  */
  
  bool greaterThanStopCond = false;
  if(decreasingFactor > std::pow(10, 6)){
    greaterThanStopCond = true;
  }
  
  // terminate the iteration when the initial residual is decreased by a factor of 106 or after 5000 iterations.
  if(greaterThanStopCond || (iterNumber >= stopAfterIterNum)){
    printf("Iter[%d]:norm=%10f, decreasingFactor=%10f\n", iterNumber, decreasingFactor);
    return;
  }else{
    
    iterNumber++;
    double *swap = arr_u_k;
    arr_u_k = arr_u_k_plus_1;
    arr_u_k_plus_1 = swap;
    processNextIter(iterNumber, initialNorm, n, range, h, arr_u_k, arr_f, arr_u_k_plus_1);
    return;
    
  }
  
}

__global__
void calculate_next_u_d(int n, long range, double h, double* arr_u_k, double* arr_f, double* arr_u_k_plus_1){
  
  // use the Jacobi method
  // calculate arr_u_1
  
  int i = blockIdx.x;
  int j = threadIdx.x;
        
        double f_i_j = 0;
        double u_k_im1_j = 0;
        double u_k_i_jm1 = 0;
        double u_k_ip1_j = 0;
        double u_k_i_jp1 = 0;
        
        int idx_i_j = calcRowMajorIndex_d(i, j, n);
        f_i_j = arr_f[idx_i_j];
        // printf("f[%d][%d]=%10f\n", i, j , f_i_j);
        
        if(i > 0){
          int idx_im1_j = calcRowMajorIndex_d(i - 1, j, n);
          u_k_im1_j = arr_u_k[idx_im1_j];
        }
        
        if(j > 0){
          int idx_i_jm1 = calcRowMajorIndex_d(i, j - 1, n);
          u_k_i_jm1 = arr_u_k[idx_i_jm1];
        }
        
        if(i < (n - 1)){
          int idx_ip1_j = calcRowMajorIndex_d(i + 1, j, n);
          u_k_ip1_j = arr_u_k[idx_ip1_j];
          // printf("arr_u_k[%ld]=%10f\n", idx_ip1_j, u_k_ip1_j);
        }
        
        if(j < (n - 1)){
          int idx_i_jp1 = calcRowMajorIndex_d(i, j + 1, n);
          u_k_i_jp1 = arr_u_k[idx_i_jp1];
          // printf("arr_u_k[%ld]=%10f\n", idx_i_jp1, u_k_i_jp1);
        }
        
        arr_u_k_plus_1[idx_i_j] = (std::pow(h, 2) * f_i_j + u_k_im1_j + u_k_i_jm1 + u_k_ip1_j + u_k_i_jp1) / 4;
        
        // printf("i=%d, j=%d, u(k+1)_i_j=%10f, f_i_j=%10f, u_k_im1_j=%10f, u_k_i_jm1=%10f, u_k_ip1_j=%10f, u_k_i_jp1=%10f\n", i, j, arr_u_k_plus_1[idx_i_j], f_i_j, u_k_im1_j, u_k_i_jm1, u_k_ip1_j, u_k_i_jp1);
      
      
  
}

__global__ void calculateResidualSqSumPerRow_d(int n, long range, double h, double* arr_u, double* arr_f, double* residualSqSum, int residualSqSumRange){
  
  // calculate residual matrix A * u(k) - f
  int i = blockIdx.x;
  int j = threadIdx.x;
  extern __shared__ double tempResidualSqSum[];
  tempResidualSqSum[threadIdx.x] = 0;
      
      double f_i_j = 0;
      double u_k_im1_j = 0;
      double u_k_i_jm1 = 0;
      double u_k_i_j = 0;
      double u_k_ip1_j = 0;
      double u_k_i_jp1 = 0;
      
      int idx_i_j = calcRowMajorIndex_d(i, j, n);
      // printf("i=%d, j=%d, idx_i_j=%d\n", i, j, idx_i_j);
      f_i_j = arr_f[idx_i_j];
      // printf("[i=%d][j=%d] f_i_j=%10f\n", i, j, f_i_j);
      u_k_i_j = arr_u[idx_i_j];
      // printf("[i=%d][j=%d] u_k_i_j=%10f\n", i, j, u_k_i_j);
      
      if(i > 0){
        int idx_im1_j = calcRowMajorIndex_d(i - 1, j, n);
        u_k_im1_j = arr_u[idx_im1_j];
        // printf("[i=%d][j=%d] u_k_im1_j=%10f\n", i, j, u_k_im1_j);
      }
      
      if(j > 0){
        int idx_i_jm1 = calcRowMajorIndex_d(i, j - 1, n);
        u_k_i_jm1 = arr_u[idx_i_jm1];
        // printf("[i=%d][j=%d] u_k_i_jm1=%10f\n", i, j, u_k_i_jm1);
      }
      
      if(i < (n - 1)){
        int idx_ip1_j = calcRowMajorIndex_d(i + 1, j, n);
        u_k_ip1_j = arr_u[idx_ip1_j];
        // printf("[i=%d][j=%d] u_k_ip1_j=%10f\n", i, j, u_k_ip1_j);
      }
      
      if(j < (n - 1)){
        int idx_i_jp1 = calcRowMajorIndex_d(i, j + 1, n);
        u_k_i_jp1 = arr_u[idx_i_jp1];
        // printf("[i=%d][j=%d] u_k_i_jp1=%10f\n", i, j, u_k_i_jp1);
      }
      
      
      // residual = f_i_j - ((- u_im1_j - u_i_jm1 + 4 u_i_j - u_ip1_j - u_i_jp1)/(h ^ 2))
      double a_mult_u = (-1 * u_k_im1_j - u_k_i_jm1 + 4 * u_k_i_j - u_k_ip1_j - u_k_i_jp1) / (std::pow(h, 2.0));
      double residual = f_i_j - a_mult_u;
      // printf("res[%d][%d]=%10f\n", i, j, residual);
      tempResidualSqSum[j] = std::pow(residual, 2);
      __syncthreads();
      
  for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if(threadIdx.x < s) {
      // printf("[%d] tempResidualSqSum[%d]=%ld\n", i, threadIdx.x, tempResidualSqSum[threadIdx.x]);
      tempResidualSqSum[threadIdx.x] += tempResidualSqSum[threadIdx.x + s];
      // printf("[%d] +tempResidualSqSum[%d]=%ld\n", i, threadIdx.x + s, tempResidualSqSum[threadIdx.x + s]);
      // printf("[%d] => tempResidualSqSum[%d]=%ld\n", i, threadIdx.x, tempResidualSqSum[threadIdx.x]);
    }
    __syncthreads();
  }
  
  // printf("1threadIdx.x=%d\n", threadIdx.x);
  if(threadIdx.x == 0){
    // printf("2threadIdx.x=%d\n", threadIdx.x);
    residualSqSum[i] = tempResidualSqSum[threadIdx.x];
    // printf("residualSqSum[%d]=%10f\n", i, residualSqSum[i]);
  }
      // printf("residualSqSum=%10f\n", residualSqSum);
      // printf("i=%ld, j=%ld, residual=%10f\n", i, j ,residual);
      // printf("f_i_j=%10f, u_k_i_j=%10f, u_k_im1_j=%10f, u_k_i_jm1=%10f, u_k_ip1_j=%10f, u_k_i_jp1=%10f\n", f_i_j, u_k_i_j, u_k_im1_j, u_k_i_jm1, u_k_ip1_j, u_k_i_jp1);
      
    // }
  // }
  
  
}

__global__ void calculateResidualNorm_d(double* residualSqSum, double* norm_d){
  
  // printf("threadIdx.x=%d\n", threadIdx.x);
  extern __shared__ double tempResidualSqSum[];
  // __shared__ double tempResidualSqSum[1024];
  tempResidualSqSum[threadIdx.x] = residualSqSum[threadIdx.x];
  __syncthreads();
  
  for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if(threadIdx.x < s) {
      tempResidualSqSum[threadIdx.x] += tempResidualSqSum[threadIdx.x + s];
    }
    __syncthreads();
  }
  
  
  if(threadIdx.x == 0){
    norm_d[0] = std::sqrt(tempResidualSqSum[threadIdx.x]);
    // printf("norm_d[0]=%10.10f\n", norm_d[0]);
  }
  
}

double calculateResidualNorm_GPU(int n, long range, double h, double* arr_u, double* arr_f, double* residualSqSum, int residualSqSumRange, double* norm_d, double* norm){
  
  double normValue = 0;
  
  calculateResidualSqSumPerRow_d<<<n, n, residualSqSumRange * sizeof (double)>>>(n, range, h, arr_u, arr_f, residualSqSum, residualSqSumRange);
  // calculateResidualSqSumPerRow_d<<<n, n>>>(n, range, h, arr_u, arr_f, residualSqSum);
  cudaDeviceSynchronize();
  
  // printf("Start to calculateResidualNorm_d\n");
  calculateResidualNorm_d<<<1, n, residualSqSumRange * sizeof (double)>>>(residualSqSum, norm_d);
  // calculateResidualNorm_d<<<1, n>>>(residualSqSum, norm_d);
  cudaDeviceSynchronize();
  // printf("-norm[0]=%10f\n", norm[0]);
  cudaMemcpy(norm, norm_d, 1 * sizeof(double), cudaMemcpyDeviceToHost);
  // printf("+norm[0]=%10f\n", norm[0]);
  normValue = norm[0];
  
  return normValue;
  
}

void processNextIter_GPU(int iterNumber, double initialNorm, int n, long range, double h, double* arr_u_k, double* arr_f, double* arr_u_k_plus_1, double* residualSqSum, int residualSqSumRange, double* norm_d, double* norm){
  
  calculate_next_u_d<<<n, n>>>(n, range, h, arr_u_k, arr_f, arr_u_k_plus_1);
  cudaDeviceSynchronize();
  double thisNorm = calculateResidualNorm_GPU(n, range, h, arr_u_k_plus_1, arr_f, residualSqSum, residualSqSumRange, norm_d, norm);
  double decreasingFactor = initialNorm / thisNorm;
  
  /*
  std::cout << "Iter[";
  std::cout << iterNumber;
  std::cout << "]:norm=";
  std::cout << thisNorm;
  std::cout << ", decreasingFactor=";
  std::cout << decreasingFactor;
  std::cout << "\n";
  */
  
  bool greaterThanStopCond = false;
  if(decreasingFactor > std::pow(10, 6)){
    greaterThanStopCond = true;
  }
  
  // terminate the iteration when the initial residual is decreased by a factor of 106 or after 5000 iterations.
  if(greaterThanStopCond || (iterNumber >= stopAfterIterNum)){
    
    printf("Iter[%d]:norm=%10f, decreasingFactor=%10f\n", iterNumber, decreasingFactor);
    return;
    
  }else{
    
    iterNumber++;
    double *swap = arr_u_k;
    arr_u_k = arr_u_k_plus_1;
    arr_u_k_plus_1 = swap;
    processNextIter_GPU(iterNumber, initialNorm, n, range, h, arr_u_k, arr_f, arr_u_k_plus_1, residualSqSum, residualSqSumRange, norm_d, norm);
    return;
    
  }
  
}

void jacobi2D_GPU(int n, long range, double h, double* arr_u_0, double* arr_f_d, double* arr_u_result, double* arr_u_0_d, double* arr_u_result_d){
  
  double *norm_d;
  cudaMalloc(&norm_d, 1 * sizeof(double));
  int iterNumber = 0;
  double *norm = (double*) aligned_malloc(1 * sizeof(double));
  
  int residualSqSumRange = std::pow(2, ceil(log2((double)n)));
  printf("n=%d, residualSqSumRange=%d\n", n, residualSqSumRange);
  double *residualSqSum;
  cudaMalloc(&residualSqSum, residualSqSumRange * sizeof(double));
  
  cudaMemcpy(arr_u_0_d, arr_u_0, range * sizeof(double), cudaMemcpyHostToDevice);
  
  // printf("Start to calculateResidualNorm_GPU\n");
  double norm_0 = calculateResidualNorm_GPU(n, range, h, arr_u_0_d, arr_f_d, residualSqSum, residualSqSumRange, norm_d, norm);
  // printf("norm_0=%10.10f\n", norm_0);
  iterNumber++;
  processNextIter_GPU(iterNumber, norm_0, n, range, h, arr_u_0_d, arr_f_d, arr_u_result_d, residualSqSum, residualSqSumRange, norm_d, norm);
  
  cudaMemcpy(arr_u_result, arr_u_result_d, range * sizeof(double), cudaMemcpyDeviceToHost);
  
  cudaFree(norm_d);
  cudaFree(residualSqSum);
  
  aligned_free(norm);
  
}

int main(int argc, char *argv[]){

  // common settings
  std::cout.precision(10);
  
  // get input param N
  int n = 256;
  int threadNumber = 4;
  
  if(argc >= 3){
  
    string nstr = argv[1];
    n = stoi(nstr);
    
    string threadNumberStr = argv[2];
    int threadNumber = stoi(threadNumberStr);
    
  }
  
  #ifdef _OPENMP
  omp_set_num_threads(threadNumber);
  #endif
  
  // initialization
  // 1. set array f = {f_11, f12, ..., f_1N, f_21, ..., f_N1, ..., f_NN}
  double start, end;
  long range = n * n;
  double* arr_f = (double*) aligned_malloc(range * sizeof(double));
  for(long i = 0; i < range; i++){
    arr_f[i] = 1;
  }
  
  // calculate h according to the input N
  double h = (double)1 / (n + 1);
  strs << h;
  std::cout << "h=" + strs.str() + "\n";
  strs.clear();
  strs.str("");
    
  // set a initialzation vector u^0
  int iterNumber = 0;
  double* arr_u_0 = (double*) aligned_malloc(range * sizeof(double));
  double* arr_u_result = (double*) aligned_malloc(range * sizeof(double));
  double* arr_u_result_ref = (double*) aligned_malloc(range * sizeof(double));
  for(long i = 0; i < range; i++){
    arr_u_0[i] = 0;
  }
  
  double norm_0 = calculateResidualNorm(n, range, h, arr_u_0, arr_f);
  iterNumber++;
  
  start = omp_get_wtime();
  processNextIter(iterNumber, norm_0, n, range, h, arr_u_0, arr_f, arr_u_result_ref);
  end = omp_get_wtime(); // unit: second
  
  /*
  std::cout << "arr_u_result:\n";
  for(long i = 0; i < range; i++){
    std::cout << arr_u_result[i];
    std::cout << "\n";
  }
  */
  
  strs << n;
  std::cout << "N=" + strs.str() +"\n";
  strs.clear();
  strs.str("");
  
  printf("jacobi2D_omp: run time = %10f second(s).\n", (end - start));
  
  // re-initialize
  for(long i = 0; i < range; i++){
    arr_u_0[i] = 0;
  }
  
  // initialize for GPU version
  double *arr_u_0_d, *arr_u_result_d, *arr_f_d;
  cudaMalloc(&arr_u_0_d, range * sizeof(double));
  cudaMalloc(&arr_u_result_d, range * sizeof(double));
  cudaMalloc(&arr_f_d, range * sizeof(double));
  
  start = omp_get_wtime();
  cudaMemcpy(arr_f_d, arr_f, range * sizeof(double), cudaMemcpyHostToDevice);
  jacobi2D_GPU(n, range, h, arr_u_0, arr_f_d, arr_u_result, arr_u_0_d, arr_u_result_d);
  end = omp_get_wtime();
  printf("jacobi2D_GPU: run time = %10f second(s).\n", (end - start));
  
  double err = 0;
  for (long i = 0; i < range; i++) {
    err += (fabs(arr_u_result[i] - arr_u_result_ref[i]));
  }
  printf("Total Error = %10e\n", err);
  
  double max_err = 0;
  for (long i = 0; i < range; i++){
    max_err = std::max(max_err, fabs(arr_u_result[i] - arr_u_result_ref[i]));
  }
  printf("max_err: %10e\n", max_err);
  
  // free memory before terminate
  cudaFree(arr_u_0_d);
  cudaFree(arr_u_result_d);
  
  aligned_free(arr_f);
  aligned_free(arr_u_0);
  aligned_free(arr_u_result);
  aligned_free(arr_u_result_ref);

}
