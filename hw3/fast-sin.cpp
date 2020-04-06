#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "intrin-wrapper.h"

// Headers for intrinsics
#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif

// g++-9 -std=c++11 -O3 -ftree-vectorize -mavx -march=native -fopenmp -o fast-sin fast-sin.cpp
// coefficients in the Taylor series expansion of sin(x)
static constexpr double c3 = -1 / (((double) 2) * 3);
static constexpr double c5 = 1 / (((double) 2) * 3 * 4 * 5);
static constexpr double c7 = -1 / (((double) 2) * 3 * 4 * 5 * 6 * 7);
static constexpr double c9 = 1 / (((double) 2) * 3 * 4 * 5 * 6 * 7 * 8 * 9);
static constexpr double c11 = -1
    / (((double) 2) * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11);
static constexpr double c13 = 1
    / (((double) 2) * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11 * 12 * 13);
static constexpr double c15 = -1
    / (((double) 2) * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11 * 12 * 13 * 14 * 15);
static constexpr double c17 = 1
    / (((double) 2) * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11 * 12 * 13 * 14 * 15 * 16 * 17);
static constexpr double c19 = -1
    / (((double) 2) * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11 * 12 * 13 * 14 * 15 * 16 * 17 * 18 * 19);
static constexpr double c21 = 1
    / (((double) 2) * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11 * 12 * 13 * 14 * 15 * 16 * 17 * 18 * 19 * 20 * 21);
static constexpr double c23= -1
    / (((double) 2) * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11 * 12 * 13 * 14 * 15 * 16 * 17 * 18 * 19 * 20 * 21 * 22 * 23);
// sin(x) = x + c3*x^3 + c5*x^5 + c7*x^7 + x9*x^9 + c11*x^11

void sin4_reference(double *sinx, const double *x) {
  for (long i = 0; i < 4; i++)
    sinx[i] = sin(x[i]);
}

void sin4_taylor(double *sinx, const double *x) {
  
  // For n = floor(x / 2pi), then -pi <= x - (2n + 1) * pi <= pi
  // Also, sin(x) = -1 * sin(x + (2n + 1) * pi) for n is an integer
  
  for (int i = 0; i < 4; i++) {
    
    double x1 = x[i];
    bool isTransform = false;
    if(x1 < -M_PI || x1 > M_PI){
      isTransform = true;
      double ori_x = x[i]; // x
      double n = floor(ori_x / (2 * M_PI));
      x1 = ori_x - (2 * n + 1) * M_PI;
      // printf("x=%10f, n=%10f, x1=%10f.\n", ori_x, n, x1);
    }
    // sin(x) = -sin(x1);
    
    
    double x2 = x1 * x1;
    double x3 = x1 * x2;
    double x5 = x3 * x2;
    double x7 = x5 * x2;
    double x9 = x7 * x2;
    double x11 = x9 * x2;
    double x13 = x11 * x2;
    double x15 = x13 * x2;
    double x17 = x15 * x2;
    double x19 = x17 * x2;
    double x21 = x19 * x2;
    double x23 = x21 * x2;

    double s = x1;
    s += x3 * c3;
    s += x5 * c5;
    s += x7 * c7;
    s += x9 * c9;
    s += x11 * c11;
    s += x13 * c13;
    s += x15 * c15;
    s += x17 * c17;
    s += x19 * c19;
    s += x21 * c21;
    s += x23 * c23;
    if(isTransform){
      sinx[i] = -s;
    }else{
      sinx[i] = s;
    }
    
  }
}

void sin4_intrin(double *sinx, const double *x) {
  // The definition of intrinsic functions can be found at:
  // https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
#if defined(__AVX__)

  __m256d x1, x2, x3, x5, x7, x9, x11, x13, x15, x17, x19, x21, x23;
  const __m256d avx_M_PI = {M_PI};
  
  x1  = _mm256_load_pd(x);
  __m256d n = _mm256_floor_pd(_mm256_div_pd(x1, _mm256_mul_pd(_mm256_set1_pd((double)2), _mm256_set1_pd(M_PI))));
  x1 = _mm256_sub_pd(x1,_mm256_mul_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd((double)2), n), _mm256_set1_pd((double)1)), _mm256_set1_pd(M_PI)));
  // x1 = ori_x - (2 * n + 1) * M_PI;
  
  
  x2  = _mm256_mul_pd(x1, x1);
  x3  = _mm256_mul_pd(x1, x2);
  x5  = _mm256_mul_pd(x3, x2);
  x7  = _mm256_mul_pd(x5, x2);
  x9  = _mm256_mul_pd(x7, x2);
  x11  = _mm256_mul_pd(x9, x2);
  x13  = _mm256_mul_pd(x11, x2);
  x15  = _mm256_mul_pd(x13, x2);
  x17  = _mm256_mul_pd(x15, x2);
  x19  = _mm256_mul_pd(x17, x2);
  x21  = _mm256_mul_pd(x19, x2);
  x23  = _mm256_mul_pd(x21, x2);

  __m256d s = x1;
  s = _mm256_add_pd(s, _mm256_mul_pd(x3 , _mm256_set1_pd(c3)));
  s = _mm256_add_pd(s, _mm256_mul_pd(x5 , _mm256_set1_pd(c5)));
  s = _mm256_add_pd(s, _mm256_mul_pd(x7 , _mm256_set1_pd(c7)));
  s = _mm256_add_pd(s, _mm256_mul_pd(x9 , _mm256_set1_pd(c9)));
  s = _mm256_add_pd(s, _mm256_mul_pd(x11 , _mm256_set1_pd(c11)));
  s = _mm256_add_pd(s, _mm256_mul_pd(x13 , _mm256_set1_pd(c13)));
  s = _mm256_add_pd(s, _mm256_mul_pd(x15 , _mm256_set1_pd(c15)));
  s = _mm256_add_pd(s, _mm256_mul_pd(x17 , _mm256_set1_pd(c17)));
  s = _mm256_add_pd(s, _mm256_mul_pd(x19 , _mm256_set1_pd(c19)));
  s = _mm256_add_pd(s, _mm256_mul_pd(x21 , _mm256_set1_pd(c21)));
  s = _mm256_add_pd(s, _mm256_mul_pd(x23 , _mm256_set1_pd(c23)));
  
  s = _mm256_mul_pd(s, _mm256_set1_pd((double)-1));
  _mm256_store_pd(sinx, s);
  
#elif defined(__SSE2__)
  constexpr int sse_length = 2;
  for (int i = 0; i < 4; i += sse_length) {
    __m128d x1, x2, x3;
    x1 = _mm_load_pd(x + i);
    x2 = _mm_mul_pd(x1, x1);
    x3 = _mm_mul_pd(x1, x2);

    __m128d s = x1;
    s = _mm_add_pd(s, _mm_mul_pd(x3, _mm_set1_pd(c3)));
    _mm_store_pd(sinx + i, s);
  }
#else
  sin4_reference(sinx, x);
#endif
}

void sin4_vector(double *sinx, const double *x) {
  
  // The Vec class is defined in the file intrin-wrapper.h
  typedef Vec<double, 4> Vec4;
  Vec4 x1, x2, x3, x5, x7, x9, x11, x13, x15, x17, x19, x21, x23;
  x1 = Vec4::LoadAligned(x);
  
  // n = floor(x / 2pi)
  Vec4 x_div_2pi = x1 / (2 * M_PI);
  Vec4 n = floor(x_div_2pi);
  // x1 = ori_x - (2 * n + 1) * M_PI;
  x1 = x1 - (2 * n + 1) * M_PI;
  
  x2 = x1 * x1;
  x3 = x1 * x2;
  x5 = x3 * x2;
  x7 = x5 * x2;
  x9 = x7 * x2;
  x11 = x9 * x2;
  x13 = x11 * x2;
  x15 = x13 * x2;
  x17 = x15 * x2;
  x19 = x17 * x2;
  x21 = x19 * x2;
  x23 = x21 * x2;

  Vec4 s = x1;
  s += x3 * c3;
  s += x5 * c5;
  s += x7 * c7;
  s += x9 * c9;
  s += x11 * c11;
  s += x13 * c13;
  s += x15 * c15;
  s += x17 * c17;
  s += x19 * c19;
  s += x21 * c21;
  s += x23 * c23;
  s = -s;
  s.StoreAligned(sinx);
  
}

double err(double *x, double *y, long N) {
  double error = 0;
  for (long i = 0; i < N; i++)
    error = std::max(error, fabs(x[i] - y[i]));
  return error;
}

int main() {

  Timer tt;
  long N = 1000000; // 1_000_000
  // long N = 4;
  double *x = (double*) aligned_malloc(N * sizeof(double));
  double *sinx_ref = (double*) aligned_malloc(N * sizeof(double));
  double *sinx_taylor = (double*) aligned_malloc(N * sizeof(double));
  double *sinx_intrin = (double*) aligned_malloc(N * sizeof(double));
  double *sinx_vector = (double*) aligned_malloc(N * sizeof(double));

  // Initialize
  // Initialize an array x with random numbers between [-pi/4,pi/4]
  // Initialize arrays, sinx_ref, sinx_taylor, sinx_intrin and sinx_vector, filled with 0
  for (long i = 0; i < N; i++) {
    // x[i] = (drand48() - 0.5) * M_PI / 2; // [-pi/4,pi/4]
    // x[i] = (drand48() - 0.5) * M_PI; // [-pi/2,pi/2]
    x[i] = (drand48() - 0.5) * (rand() % 100);
    // x[i] = (drand48() - 0.5) * 2 * 100; // [-100,100]
    sinx_ref[i] = 0;
    sinx_taylor[i] = 0;
    sinx_intrin[i] = 0;
    sinx_vector[i] = 0;
  }

  #if defined(__AVX__)
    printf("Calculate sin4_intrin using AVX.\n");
  #elif defined(__SSE2__)
    printf("Calculate sin4_intrin using SSE2.\n");
  #else
    printf("Calculate sin4_intrin using the built-in C/C++ function.\n");
  #endif

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i += 4) {
      sin4_reference(sinx_ref + i, x + i);
    }
  }
  printf("Reference time: %6.4f\n", tt.toc());

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i += 4) {
      sin4_taylor(sinx_taylor + i, x + i);
    }
  }
  printf("Taylor time:    %6.4f      Error: %e\n", tt.toc(),
      err(sinx_ref, sinx_taylor, N));

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i += 4) {
      sin4_intrin(sinx_intrin + i, x + i);
    }
  }
  printf("Intrin time:    %6.4f      Error: %e\n", tt.toc(),
      err(sinx_ref, sinx_intrin, N));

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i += 4) {
      sin4_vector(sinx_vector + i, x + i);
    }
  }
  printf("Vector time:    %6.4f      Error: %e\n", tt.toc(),
      err(sinx_ref, sinx_vector, N));

  aligned_free(x);
  aligned_free(sinx_ref);
  aligned_free(sinx_taylor);
  aligned_free(sinx_intrin);
  aligned_free(sinx_vector);
}

