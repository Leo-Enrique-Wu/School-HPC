// g++ -fopenmp -O3 -march=native MMult1.cpp && ./a.out
// g++ -std=c++11 -O3 -march=native -o MMult1  MMult1.cpp
// g++ -std=c++11 -fopenmp -O3 -march=native -o MMult1  MMult1.cpp

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

// 128 optimal
// O3 is better than O2
#define BLOCK_SIZE 128

// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
void MMult0(long m, long n, long k, double *a, double *b, double *c) {
  
  // jpi
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}

void MMult0RearrangeLoopOrder(long m, long n, long k, double *a, double *b, double *c) {
  
  /*
  // jip: 950-3.73
  for (long j = 0; j < n; j++) {
    for (long i = 0; i < m; i++) {
      for (long p = 0; p < k; p++) {
  */
  
  /*
  // jpi: 950-0.793
  //   better than jip
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
  */
  
  /*
  // ijp: 950-2.63
  for (long i = 0; i < m; i++) {
    for (long j = 0; j < n; j++) {
      for (long p = 0; p < k; p++) {
   */
  
  /*
  // ipj: 950-20.16
  for (long i = 0; i < m; i++) {
    for (long p = 0; p < k; p++) {
      for (long j = 0; j < n; j++) {
  */
        
  /*
  // pij: 950-19.48
  for (long p = 0; p < k; p++) {
    for (long i = 0; i < m; i++) {
      for (long j = 0; j < n; j++) {
  */
        
  // pji: 950-0.75
  for (long p = 0; p < k; p++) {
    for (long j = 0; j < n; j++) {
      for (long i = 0; i < m; i++) {
        
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
        
      }
    }
  }
  
}

void MMult1BlockVersion(long m, long n, long k, double *a, double *b, double *c) {
  
  #pragma omp parallel for
  for (long i = 0; i < m; i+= BLOCK_SIZE) {
    
    for (long j = 0; j < n; j+= BLOCK_SIZE) {
      
      long blockCeiling_ii = ((i + BLOCK_SIZE) > m)? m : i + BLOCK_SIZE;
      long blockCeiling_jj = ((j + BLOCK_SIZE) > n)? n : j + BLOCK_SIZE;
      
      // initialize tempC
      long temp_range_i = blockCeiling_ii - i;
      long temp_range_j = blockCeiling_jj - j;
      long tempC_range = temp_range_i * temp_range_j;
      double tempC[tempC_range];
      for(long c_idx = 0; c_idx < tempC_range; c_idx++) tempC[c_idx] = 0;
      
      // load C to block tempC
      for(long jj = j; jj < blockCeiling_jj; jj++){
        for(long ii = i; ii < blockCeiling_ii; ii++){
          
          long c_idx = (ii - i) + ((jj - j) * temp_range_i);
          tempC[c_idx] += c[ii + jj * m];
          
        }
      }
      
      for (long p = 0; p < k; p+= BLOCK_SIZE) {
 
        long blockCeiling_pp = ((p + BLOCK_SIZE) > k)? k : p + BLOCK_SIZE;
        long temp_range_p = blockCeiling_pp - p;
        
        // initialize tempA
        long tempA_range = temp_range_i * temp_range_p;
        double tempA[tempA_range];
        for(long a_idx = 0; a_idx < tempA_range; a_idx++) tempA[a_idx] = 0;
        
        // load block tempA
        for(long pp = p; pp < blockCeiling_pp; pp++){
          for(long ii = i; ii < blockCeiling_ii; ii++){
            
            long a_idx = (ii - i) + ((pp - p) * temp_range_i);
            tempA[a_idx] += a[ii + pp * m];
            
          }
        }
        
        // initialize tempB
        long tempB_range = temp_range_p * temp_range_j;
        double tempB[tempB_range];
        for(long b_idx = 0; b_idx < tempB_range; b_idx++) tempB[b_idx] = 0;
        
        // load block tempB
        for(long jj = j; jj < blockCeiling_jj; jj++){
          for(long pp = p; pp < blockCeiling_pp; pp++){
            
            long b_idx = (pp - p) + ((jj - j) * temp_range_p);
            tempB[b_idx] += b[pp + jj * k];
            
          }
        }
        
        // B x B mini matrix multiplications
        for(long temp_p = 0; temp_p < temp_range_p; temp_p++){
          for(long temp_j = 0; temp_j < temp_range_j; temp_j++){
            for(long temp_i = 0; temp_i < temp_range_i; temp_i++){
              
              double A_ip = tempA[temp_i + temp_p * temp_range_i];
              double B_pj = tempB[temp_p + temp_j * temp_range_p];
              double C_ij = tempC[temp_i + temp_j * temp_range_i];
              C_ij = C_ij + A_ip * B_pj;
              tempC[temp_i + temp_j * temp_range_i] = C_ij;
              
            }
          }
        }
        
      }
      
      
      // store block tempC
      for(long jj = j; jj < blockCeiling_jj; jj++){
        for(long ii = i; ii < blockCeiling_ii; ii++){
          
          long c_idx = (ii - i) + ((jj - j) * temp_range_i);
          c[ii + jj * m] = tempC[c_idx];
          
        }
      }
    
    }
  }
  
  /*
  for (long j = 0; j < n; j+= BLOCK_SIZE) {
    for (long p = 0; p < k; p+= BLOCK_SIZE) {
      for (long i = 0; i < m; i+= BLOCK_SIZE) {
        
        // B x B mini matrix multiplications
        long blockCeiling_jj = ((j + BLOCK_SIZE) > n)? n : j + BLOCK_SIZE;
        long blockCeiling_pp = ((p + BLOCK_SIZE) > k)? k : p + BLOCK_SIZE;
        long blockCeiling_ii = ((i + BLOCK_SIZE) > m)? m : i + BLOCK_SIZE;
        
        long tempC_range = blockCeiling_ii * blockCeiling_jj
        double tempC_j[tempC_range];
        for(long c_idx = 0; c_idx < tempC_range; c_idx++) tempC_j[c_idx] = 0;
        
        for(long jj = j; jj < blockCeiling_jj; jj++){
          for(long pp = p; pp < blockCeiling_pp; pp++){
            
            double B_pj = b[pp + jj * k];
            
            for(long ii = i; ii < blockCeiling_ii; ii++){
              c[ii + jj * m] += a[ii + pp * m] * B_pj;
            }
            
          }
        }
        
      }
    }
  }
  */
  
}

void MMult1(long m, long n, long k, double *a, double *b, double *c) {
  
  MMult1BlockVersion(m, n, k, a, b, c);
  
}

int main(int argc, char** argv) {
  
  
  const long PFIRST = BLOCK_SIZE;
  const long PLAST = 2000;
  const long PINC = std::max(50/BLOCK_SIZE,1) * BLOCK_SIZE; // multiple of BLOCK_SIZE
  
  /*
  const long PFIRST = 50;
  const long PLAST = 2000;
  const long PINC = 100;
   */
  
  printf("BLOCK_SIZE:%d\n", BLOCK_SIZE);
  
  
  printf(" Dimension       Time    Gflop/s       GB/s        Error\n");
  
  for (long p = PFIRST; p < PLAST; p += PINC) {
    
    long m = p, n = p, k = p;
    long NREPEATS = 1e9/(m*n*k)+1;
    // long NREPEATS = 1;
    // When p increases, NREPEATS decreases.
    double* a = (double*) aligned_malloc(m * k * sizeof(double)); // m x k
    double* b = (double*) aligned_malloc(k * n * sizeof(double)); // k x n
    double* c = (double*) aligned_malloc(m * n * sizeof(double)); // m x n
    double* c_ref = (double*) aligned_malloc(m * n * sizeof(double)); // m x n
    
    // Initialize matrices
    for (long i = 0; i < m*k; i++) a[i] = drand48();
    for (long i = 0; i < k*n; i++) b[i] = drand48();
    for (long i = 0; i < m*n; i++) c_ref[i] = 0;
    for (long i = 0; i < m*n; i++) c[i] = 0;
    
    for (long rep = 0; rep < NREPEATS; rep++) { // Compute reference solution
      MMult0(m, n, k, a, b, c_ref);
    }
    
    Timer t;
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
      MMult1(m, n, k, a, b, c);
    }
    double time = t.toc();
    
    double flops = (((m * n * k) * 2 * NREPEATS) / 1e9) / time;
      // "/ 1e9": because the measure unit is "G"flop/s
    
    long blockNum_m = m / BLOCK_SIZE;
    long blockNum_n = n / BLOCK_SIZE;
    long blockNum_k = k / BLOCK_SIZE;
    
    long loadAndStoreC = blockNum_m * blockNum_n * BLOCK_SIZE * BLOCK_SIZE * 2;
    long loadFromAandB = blockNum_m * blockNum_n * blockNum_k * BLOCK_SIZE * BLOCK_SIZE * 2;
    
    double bandwidth = ((loadAndStoreC + loadFromAandB) * NREPEATS) * sizeof(double) / 1e9 / time;
    
    printf("%10ld %10f %10f %10f", p, time, flops, bandwidth);

    double max_err = 0;
    for (long i = 0; i < m*n; i++) max_err = std::max(max_err, fabs(c[i] - c_ref[i]));
    printf(" %10e\n", max_err);

    aligned_free(a);
    aligned_free(b);
    aligned_free(c);
    
  }

  return 0;
}

// * Using MMult0 as a reference, implement MMult1 and try to rearrange loops to
// maximize performance. Measure performance for different loop arrangements and
// try to reason why you get the best performance for a particular order?
//
//
// * You will notice that the performance degrades for larger matrix sizes that
// do not fit in the cache. To improve the performance for larger matrices,
// implement a one level blocking scheme by using BLOCK_SIZE macro as the block
// size. By partitioning big matrices into smaller blocks that fit in the cache
// and multiplying these blocks together at a time, we can reduce the number of
// accesses to main memory. This resolves the main memory bandwidth bottleneck
// for large matrices and improves performance.
//
// NOTE: You can assume that the matrix dimensions are multiples of BLOCK_SIZE.
//
//
// * Experiment with different values for BLOCK_SIZE (use multiples of 4) and
// measure performance.  What is the optimal value for BLOCK_SIZE?
//
//
// * Now parallelize your matrix-matrix multiplication code using OpenMP.
//
//
// * What percentage of the peak FLOP-rate do you achieve with your code?
//
//
// NOTE: Compile your code using the flag -march=native. This tells the compiler
// to generate the best output using the instruction set supported by your CPU
// architecture. Also, try using either of -O2 or -O3 optimization level flags.
// Be aware that -O2 can sometimes generate better output than using -O3 for
// programmer optimized code.
