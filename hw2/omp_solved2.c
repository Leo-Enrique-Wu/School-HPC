/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug. 
* AUTHOR: Blaise Barney 
* LAST REVISED: 04/06/05 
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// Chia-Hao Wu: Total has 3 bugs
int main (int argc, char *argv[]) 
{
int nthreads, i, tid;
  
  // Bug 1: Type float will cause the inaccuracy. Change the type to double.
double total;

/*** Spawn parallel region ***/
  // Bug 2: The variables, tid and i, should be private for each thread.
  //        If do not state expicitly, those threads will share those variables.
  //        That will cause it print two or more thread starting with same tid.
  //        Also, once a thread increment the index, it also modify other threads'
  //        index value.
#pragma omp parallel private(tid, i)
  {
  /* Obtain thread number */
  tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */
  total = 0.0;
    // Bug 3: Because total is a shared varible, if it is not thread-safe to
    //        update the varible, it will face race condition.
#pragma omp for schedule(dynamic,10) reduction(+: total)
  for (i=0; i<1000000; i++) 
     total = total + i*1.0;

  printf ("Thread %d is done! Total= %e\n",tid,total);

  } /*** End of parallel region ***/
}
