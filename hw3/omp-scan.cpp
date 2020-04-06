#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <string>

using std::string;

int parallelThreadNum = 3;

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
// g++-9 -std=c++11 -O3 -march=native -fopenmp -o omp-scan omp-scan.cpp
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = A[0];
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  
  // k(1) - k(p + 1): (p + 1) indexes
  long k[parallelThreadNum + 1];
  
  #pragma omp parallel num_threads(parallelThreadNum)
  {
    
    #pragma omp for
    // calculate k(j)
    for(int i = 0; i < (parallelThreadNum + 1); i++){
      
      // i = j - 1
      // k(j) = 1 + ((n + 1) - 1)/p * (j - 1)
      k[i] = 0 + floor(((double)n / parallelThreadNum) * i);
      // printf("k[%d]=%ld\n", i, k[i]);
      
    }
    
    const int thread_id = omp_get_thread_num();
    int j = thread_id % parallelThreadNum;
    
    // scan local in parallel by the local first element
    long localStart = k[j];
    prefix_sum[localStart] = A[localStart];
    // printf("[Thread-%d] prefix_sum[%ld]=%ld\n", j, localStart, prefix_sum[localStart]);
    for(long i = localStart + 1; i < k[j + 1]; i++){
      prefix_sum[i] = prefix_sum[i - 1] + A[i];
      // printf("[Thread-%d] prefix_sum[%ld]=%ld\n", j, i, prefix_sum[i]);
    }
    // printf("[Thread-%d] completed local task.\n", j);
    
    // use the last element in the first parition
    // to calculate each parition's last element
    // in seqencial
    #pragma omp barrier
    #pragma omp single
    {
      for(int par = 1; par < parallelThreadNum; par++){
        
        long previousParLastEltIdx = k[par] - 1;
        long previousParLastElt = prefix_sum[previousParLastEltIdx];
        // printf("[Thread-%d] prefix_sum[%ld]=%ld\n", par, previousParLastEltIdx, previousParLastElt);
        long thisParLastEltIdx = k[par + 1] - 1;
        // printf("[Thread-%d] prefix_sum[%ld]=%ld\n", par, thisParLastEltIdx, prefix_sum[thisParLastEltIdx]);
        prefix_sum[thisParLastEltIdx] += previousParLastElt;
        // printf("[Thread-%d] Update prefix_sum[%ld]=%ld\n", par, thisParLastEltIdx, prefix_sum[thisParLastEltIdx]);
        
      }
    }
    
    // use previous parition's last element to
    // calculate rest elements in parallel
    // printf("[Thread-%d] 2nd part.\n", j);
    if(j != 0){
      long previousParLastEltIdx = k[j] - 1;
      long previousParLastElt = prefix_sum[previousParLastEltIdx];
      for(long i = localStart; i < k[j + 1] - 1; i++){
        prefix_sum[i] += previousParLastElt;
        // printf("[Thread-%d] prefix_sum[%ld]=%ld\n", j, i, prefix_sum[i]);
      }
    }
  
  }
  
}

int main(int argc, char *argv[]) {
  
  if(argc >= 2){
    string parallelThreadNumStr = argv[1];
    parallelThreadNum = stoi(parallelThreadNumStr);
  }
  printf("parallelThreadNum = %d.\n", parallelThreadNum);
  
  // ori: long N = 100000000;
  long N = 100000000;
  if(argc >= 3){
    string NStr = argv[2];
    N = stoi(NStr);
  }
  printf("N = %ld.\n", N);
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  /*
  for (long i = 0; i < N; i++) {
    A[i] = rand();
    printf("A[%ld]=%ld\n", i, A[i]);
  }
  */

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);
  /*
  for(int i = 0; i < N; i++){
    printf("B0[%ld]=%ld\n", i, B0[i]);
  }
  */
  
  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);
  /*
  for(int i = 0; i < N; i++){
    printf("B1[%ld]=%ld\n", i, B1[i]);
  }
  */

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
