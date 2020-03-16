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
// g++ -std=c++11 -O3 -o jacobi2D-omp -fopenmp jacobi2D-omp.cpp
// Without parallel: N=10, run time =   0.011500 second(s).

long calcRowMajorIndex(long i, long j, long columnSize){
  int idx = (j + ((i - 1) * columnSize)) - 1;
  return idx;
}

double* calculate_next_u(long n, long range, double h, double* arr_u_k, double* arr_f){
  
  // use the Jacobi method
  // calculate arr_u_1
  double *arr_u_k_plus_1 = new double[range];
  
  
  for(long i = 1; i <= n; i++){
     
    #pragma omp parallel for
      for(long j = 1; j <= n; j++){
        
        double f_i_j = 0;
        double u_k_im1_j = 0;
        double u_k_i_jm1 = 0;
        double u_k_ip1_j = 0;
        double u_k_i_jp1 = 0;
        
        long idx_i_j = calcRowMajorIndex(i, j, n);
        f_i_j = arr_f[idx_i_j];
        
        if(i - 1 > 0){
          long idx_im1_j = calcRowMajorIndex(i - 1, j, n);
          u_k_im1_j = arr_u_k[idx_im1_j];
        }
        
        if(j - 1 > 0){
          long idx_i_jm1 = calcRowMajorIndex(i, j - 1, n);
          u_k_i_jm1 = arr_u_k[idx_i_jm1];
        }
        
        if(i < n){
          long idx_ip1_j = calcRowMajorIndex(i + 1, j, n);
          u_k_ip1_j = arr_u_k[idx_ip1_j];
          // printf("arr_u_k[%ld]=%10f\n", idx_ip1_j, u_k_ip1_j);
        }
        
        if(j < n){
          long idx_i_jp1 = calcRowMajorIndex(i, j + 1, n);
          u_k_i_jp1 = arr_u_k[idx_i_jp1];
          // printf("arr_u_k[%ld]=%10f\n", idx_i_jp1, u_k_i_jp1);
        }
        
        arr_u_k_plus_1[idx_i_j] = (std::pow(h, 2) * f_i_j + u_k_im1_j + u_k_i_jm1 + u_k_ip1_j + u_k_i_jp1) / 4;
        
        // printf("i=%ld, j=%ld, u(k+1)_i_j=%10f\n", i, j, arr_u_k_plus_1[idx_i_j]);
      
      }
    
  }
  
  return arr_u_k_plus_1;
  
}

double calculateResidualNorm(long n, long range, double h, double* arr_u, double* arr_f){
  
  // calculate residual matrix A * u(k) - f
  double norm;
  double residualSqSum = 0;
  
  
  for(long i = 1; i <= n; i++){
    
    #pragma omp parallel for reduction(+: residualSqSum)
    for(long j = 1; j <= n; j++){
      
      double f_i_j = 0;
      double u_k_im1_j = 0;
      double u_k_i_jm1 = 0;
      double u_k_i_j = 0;
      double u_k_ip1_j = 0;
      double u_k_i_jp1 = 0;
      
      // printf("i=%ld, j=%ld\n", i, j);
      
      int idx_i_j = calcRowMajorIndex(i, j, n);
      f_i_j = arr_f[idx_i_j];
      u_k_i_j = arr_u[idx_i_j];
      // printf("u_k_i_j=%10f\n", u_k_i_j);
      
      if((i - 1) > 0){
        int idx_im1_j = calcRowMajorIndex(i - 1, j, n);
        u_k_im1_j = arr_u[idx_im1_j];
        // printf("u_k_im1_j=%10f\n", u_k_im1_j);
      }
      
      if((j - 1) > 0){
        int idx_i_jm1 = calcRowMajorIndex(i, j - 1, n);
        u_k_i_jm1 = arr_u[idx_i_jm1];
        // printf("u_k_i_jm1=%10f\n", u_k_i_jm1);
      }
      
      if(i < n){
        int idx_ip1_j = calcRowMajorIndex(i + 1, j, n);
        u_k_ip1_j = arr_u[idx_ip1_j];
        // printf("u_k_ip1_j=%10f\n", u_k_ip1_j);
      }
      
      if(j < n){
        int idx_i_jp1 = calcRowMajorIndex(i, j + 1, n);
        u_k_i_jp1 = arr_u[idx_i_jp1];
        // printf("u_k_i_jp1=%10f\n", u_k_i_jp1);
      }
      
      // residual = f_i_j - ((- u_im1_j - u_i_jm1 + 4 u_i_j - u_ip1_j - u_i_jp1)/(h ^ 2))
      double a_mult_u = (-1 * u_k_im1_j - u_k_i_jm1 + 4 * u_k_i_j - u_k_ip1_j - u_k_i_jp1) / (std::pow(h, 2.0));
      // printf("a_mult_u=%10f\n", a_mult_u);
      double residual = f_i_j - a_mult_u;
      residualSqSum += std::pow(residual, 2);
      // printf("residualSqSum=%10f\n", residualSqSum);
      // printf("i=%ld, j=%ld, residual=%10f\n", i, j ,residual);
      // printf("f_i_j=%10f, u_k_i_j=%10f, u_k_im1_j=%10f, u_k_i_jm1=%10f, u_k_ip1_j=%10f, u_k_i_jp1=%10f\n", f_i_j, u_k_i_j, u_k_im1_j, u_k_i_jm1, u_k_ip1_j, u_k_i_jp1);
      
    }
  }
  
  norm = std::sqrt(residualSqSum);
  return norm;
  
}

double* processNextIter(int iterNumber, double initialNorm, int n, long range, double h, double* arr_u_k, double* arr_f){
  
  double* arr_u_k_plus_1;
  arr_u_k_plus_1 = calculate_next_u(n, range, h, arr_u_k, arr_f);
  
  double thisNorm = calculateResidualNorm(n, range, h, arr_u_k_plus_1, arr_f);
  double decreasingFactor = initialNorm / thisNorm;
  
  std::cout << "Iter[";
  std::cout << iterNumber;
  std::cout << "]:norm=";
  std::cout << thisNorm;
  std::cout << ", decreasingFactor=";
  std::cout << decreasingFactor;
  std::cout << "\n";
  
  bool greaterThanStopCond = false;
  if(decreasingFactor > std::pow(10, 6)){
    greaterThanStopCond = true;
  }
  
  // terminate the iteration when the initial residual is decreased by a factor of 106 or after 5000 iterations.
  if(greaterThanStopCond || (iterNumber >= stopAfterIterNum)){
    return arr_u_k_plus_1;
  }else{
    
    iterNumber++;
    double* arr_u_result = new double[n];
    arr_u_result = processNextIter(iterNumber, initialNorm, n, range, h, arr_u_k_plus_1, arr_f);
    return arr_u_result;
    
  }
  
}



// g++ -std=c++11 -O3 -o jacobi2D-omp jacobi2D-omp.cpp
int main(int argc, char *argv[]){

  // common settings
  std::cout.precision(10);
  
  Timer t;
  double time;
  
  // get input param N
  /*
  string nstr = argv[1];
  long n = stoi(nstr);
  */
  long n = 300;
  
  /*
  string threadNumberStr = argv[2];
  int threadNumber = stoi(threadNumberStr);
  */
  int threadNumber = 3;
  
  #ifdef _OPENMP
  omp_set_num_threads(threadNumber);
  #endif
  
  // initialization
  // 1. set array f = {f_11, f12, ..., f_1N, f_21, ..., f_N1, ..., f_NN}
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
  long iterNumber = 0;
  double* arr_u_0 = (double*) aligned_malloc(range * sizeof(double));
  for(long i = 0; i < range; i++){
    arr_u_0[i] = 0;
  }
  
  double norm_0 = calculateResidualNorm(n, range, h, arr_u_0, arr_f);
  
  iterNumber++;
  double* arr_u_result = new double[range];
  
  t.tic();
  arr_u_result = processNextIter(iterNumber, norm_0, n, range, h, arr_u_0, arr_f);
  time = t.toc(); // unit: second
  
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
  
  printf("run time = %10f second(s).\n", time);
  
  aligned_free(arr_f);
  aligned_free(arr_u_0);

}
