/******************************************************************************
* FILE: omp_bug4.c
* DESCRIPTION:
*   This very simple program causes a segmentation fault.
* AUTHOR: Blaise Barney  01/09/04
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1048

// Chia-Hao Wu: Has 1 bug.
int main (int argc, char *argv[]) 
{
int nthreads, tid, i, j;
  
  // Bug 1: The original let every thread keep a private and large array, but the computer might do not have that much space. There are two ways to fix this problem. One is to set the environment to enlarge the stack size. The other is to use dynamic allocated array and let each thread to allocate memory by themselves. 
double **a;

/* Fork a team of threads with explicit variable scoping */
#pragma omp parallel shared(nthreads) private(i,j,tid,a)
  {

    double **a = (double **)malloc(N*sizeof(double*));
    for(int i = 0; i < N; i++)
      a[i] = (double *)malloc(N * sizeof(double));
    
  /* Obtain/print thread info */
  tid = omp_get_thread_num();
  if (tid == 0) 
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d starting...\n", tid);

  /* Each thread works on its own private copy of the array */
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      a[i][j] = tid + i + j;

  /* For confirmation */
  printf("Thread %d done. Last element= %f\n",tid,a[N-1][N-1]);
    
    free(a);

  }  /* All threads join master thread and disband */

}

