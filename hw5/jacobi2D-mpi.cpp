// mpic++ -std=c++11 -O3 -march=native -o jacobi2D-mpi jacobi2D-mpi.cpp
// mpirun --mca btl_vader_backing_directory /tmp --oversubscribe -np 16 ./jacobi2D-mpi

#include <iostream>
#include <string>
#include <stdio.h>
#include <sstream>
#include <cmath>
#include "utils.h"
#include <stdio.h>
#include <mpi.h>

#define DIM 2

#define index(ic, nc) ((ic)[0] + (nc)[0] * (ic)[1])

#define iterate(ic, minnc, maxnc) \
for((ic)[0] = (minnc)[0]; (ic)[0] < (maxnc)[0]; (ic)[0]++) \
for((ic)[1] = (minnc)[1]; (ic)[1] < (maxnc)[1]; (ic)[1]++)

#define inverseIndex(i, nc, ic) \
((ic)[0] = (i) % (nc)[0], (ic)[1] = (i) / (nc)[0])

typedef struct {
  
  int mpiRank; // process number of the local process
  int numProcs; // number of processes started
  int ip[DIM]; // position of process in the process mesh
  int np[DIM]; // size of process mesh, also number of subdomains
  int ip_lower[DIM]; // process number of the neighbor processes
  int ip_upper[DIM];
  
  int ic_start[DIM]; // width of broader neighborhood, corresponds to the first local index in the interior of the subdomain
  int ic_stop[DIM]; // first local index in the upper border neighborhood
  int ic_number[DIM]; // number of cells in subdomain, including border neighborhood
  
} SubDomain;

using std::string;

// if after iteration is over this number, we can't get the residual decreased by a factor n, then stop the program.
int stopAfterIterNum = 5000;

const int MPI_TAG_SEND_TO_LEFT = 123;
const int MPI_TAG_REC_FROM_RIGHT = MPI_TAG_SEND_TO_LEFT;
const int MPI_TAG_SEND_TO_RIGHT = 124;
const int MPI_TAG_REC_FROM_LEFT = MPI_TAG_SEND_TO_RIGHT;
const int MPI_TAG_SEND_TO_UP = 125;
const int MPI_TAG_REC_FROM_DOWN = MPI_TAG_SEND_TO_UP;
const int MPI_TAG_SEND_TO_DOWN = 126;
const int MPI_TAG_REC_FROM_UP = MPI_TAG_SEND_TO_DOWN;

double *sendToLeftBuffer = NULL;
double *sendToRightBuffer = NULL;
double *sendToUpBuffer = NULL;
double *sendToDownBuffer = NULL;
double *recFromLeftBuffer = NULL;
double *recFromRightBuffer = NULL;
double *recFromUpBuffer = NULL;
double *recFromDownBuffer = NULL;

// common utils
std::ostringstream strs;

long calcRowMajorIndex(long i, long j, long columnSize){
  long idx = j + (i * columnSize);
  return idx;
}

void calculate_next_u(SubDomain *s, double* arr_u_k, double* arr_u_k_plus_1, double h){
  
  long n = s->ic_number[0];
  long row_start_idx = s->ic_start[0];
  long row_stop_idx = s->ic_stop[0];
  long col_start_idx = s->ic_start[1];
  long col_stop_idx = s->ic_stop[1];
  
  // use the Jacobi method
  MPI_Status sendToLeftStatus, sendToRightStatus, sendToUpStatus, sendToDownStatus, recFromLeftStatus, recFromRightStatus, recFromUpStatus, recFromDownStatus;
  MPI_Request sendToLeftRequest, sendToRightRequest, sendToUpRequest, sendToDownRequest, recFromLeftRequest, recFromRightRequest, recFromUpRequest, recFromDownRequest;
  
  long parRowStartIdx, parRowStopIdx, parColStartIdx, parColStopIdx; // the index of the starting point and stopping point which do not need to wait for communicating finished
  
  long rowNum = s->ic_stop[1] - s->ic_start[1];
  long colNum = s->ic_stop[0] - s->ic_start[0];
  
  // communicate with(send & recieve) adjacent subdomain to get ghost points(with no block)
  // Send to left(if ip[0] != 0): ( ic_start[0], ic_start[1] ~ (ic_stop[1] - 1) )
  // from left(if ip[0] != 0): put in ( 0, ic_start[1] ~ (ic_stop[1] - 1) )
  if(s->ip[0] != 0){
    
    parRowStartIdx = s->ic_start[1] + 1;
    
    for(long k = 0; k < rowNum; k++){
      
      long i = s->ic_start[0];
      long j = s->ic_start[1] + k;
      long idx_i_j = calcRowMajorIndex(i, j, n);
      sendToLeftBuffer[k] = arr_u_k[idx_i_j];
      
    }
    
    MPI_Isend(sendToLeftBuffer, rowNum, MPI_DOUBLE, s->ip_lower[0], MPI_TAG_SEND_TO_LEFT, MPI_COMM_WORLD, &sendToLeftRequest);
    MPI_Irecv(recFromLeftBuffer, rowNum, MPI_DOUBLE, s->ip_lower[0], MPI_TAG_REC_FROM_LEFT, MPI_COMM_WORLD, &recFromLeftRequest);
    
  }else{
    parRowStartIdx = s->ic_start[1];
  }
  
  // to right(if ip[0] != np[0] - 1): ( (ic_stop[0] - 1), ic_start[1] ~ (ic_stop[1] - 1) )
  // from right(if ip[0] != np[0] - 1): put in ( ic_stop[0], ic_start[1] ~ (ic_stop[1] - 1) )
  if(s->ip[0] != s->np[0] - 1){
    
    parRowStopIdx = s->ic_stop[0] - 1;
    
    for(long k = 0; k < rowNum; k++){
      
      long i = s->ic_stop[0] - 1;
      long j = s->ic_start[1] + k;
      long idx_i_j = calcRowMajorIndex(i, j, n);
      sendToRightBuffer[k] = arr_u_k[idx_i_j];
      
    }
    
    MPI_Isend(sendToRightBuffer, rowNum, MPI_DOUBLE, s->ip_upper[0], MPI_TAG_SEND_TO_RIGHT, MPI_COMM_WORLD, &sendToRightRequest);
    MPI_Irecv(recFromRightBuffer, rowNum, MPI_DOUBLE, s->ip_upper[0], MPI_TAG_REC_FROM_RIGHT, MPI_COMM_WORLD, &recFromRightRequest);
    
  }else{
    parRowStopIdx = s->ic_stop[0];
  }
  
  // to up(if ip[1] != np[1] - 1): ( ic_start[0] ~ (ic_stop[0] - 1), (ic_stop[1] - 1) )
  // from up(if ip[1] != np[1] - 1): put in ( ic_start[0] ~ (ic_stop[0] - 1), ic_stop[1]  )
  if(s->ip[1] != s->np[1] - 1){
    
    parColStopIdx = s->ic_stop[1] - 1;
    
    for(long k = 0; k < colNum; k++){
      
      long i = s->ic_start[0] + k;
      long j = s->ic_stop[1] - 1;
      long idx_i_j = calcRowMajorIndex(i, j, n);
      sendToUpBuffer[k] = arr_u_k[idx_i_j];
      
    }
    
    MPI_Isend(sendToUpBuffer, colNum, MPI_DOUBLE, s->ip_upper[1], MPI_TAG_SEND_TO_UP, MPI_COMM_WORLD, &sendToUpRequest);
    MPI_Irecv(recFromUpBuffer, colNum, MPI_DOUBLE, s->ip_upper[1], MPI_TAG_REC_FROM_UP, MPI_COMM_WORLD, &recFromUpRequest);
    
  }else{
    parColStopIdx = s->ic_stop[1];
  }
  
  // Send to down(if ip[1] != 0): ( ic_start[0] ~ (ic_stop[0] - 1), ic_start[1] )
  // Rec from down(if ip[1] != 0): put in ( ic_start[0] ~ (ic_stop[0] - 1), 0 )
  if(s->ip[1] != 0){
    
    parColStartIdx = s->ic_start[1] + 1;
    
    for(long k = 0; k < colNum; k++){
      
      long i = s->ic_start[0] + k;
      long j = s->ic_start[1];
      long idx_i_j = calcRowMajorIndex(i, j, n);
      sendToDownBuffer[k] = arr_u_k[idx_i_j];
      
    }
    
    MPI_Isend(sendToDownBuffer, colNum, MPI_DOUBLE, s->ip_lower[1], MPI_TAG_SEND_TO_DOWN, MPI_COMM_WORLD, &sendToDownRequest);
    MPI_Irecv(recFromDownBuffer, colNum, MPI_DOUBLE, s->ip_lower[1], MPI_TAG_REC_FROM_DOWN, MPI_COMM_WORLD, &recFromDownRequest);
    
  }else{
    parColStartIdx = s->ic_start[1];
  }
  
  for(long i = s->ic_start[0] + 1; i < s->ic_stop[0] - 1; i++){
    for(long j = s->ic_start[1] + 1; j < s->ic_stop[1] - 1; j++){
        
        double f_i_j = 1;
        double u_k_im1_j = 0;
        double u_k_i_jm1 = 0;
        double u_k_ip1_j = 0;
        double u_k_i_jp1 = 0;
        
        if(s->ip[0] != 0 || i > row_start_idx){
          long idx_im1_j = calcRowMajorIndex(i - 1, j, n);
          u_k_im1_j = arr_u_k[idx_im1_j];
        }
        
        if(s->ip[1] != 0 || j > col_start_idx){
          long idx_i_jm1 = calcRowMajorIndex(i, j - 1, n);
          u_k_i_jm1 = arr_u_k[idx_i_jm1];
        }
        
        if(s->ip[0] != (s->np[0] - 1) || i < row_stop_idx - 1){
          long idx_ip1_j = calcRowMajorIndex(i + 1, j, n);
          u_k_ip1_j = arr_u_k[idx_ip1_j];
          // printf("arr_u_k[%ld]=%10f\n", idx_ip1_j, u_k_ip1_j);
        }
        
        if(s->ip[1] != (s->np[1] - 1) || j < col_stop_idx - 1){
          long idx_i_jp1 = calcRowMajorIndex(i, j + 1, n);
          u_k_i_jp1 = arr_u_k[idx_i_jp1];
          // printf("arr_u_k[%ld]=%10f\n", idx_i_jp1, u_k_i_jp1);
        }
        
        long idx_i_j = calcRowMajorIndex(i, j, n);
        arr_u_k_plus_1[idx_i_j] = (std::pow(h, 2) * f_i_j + u_k_im1_j + u_k_i_jm1 + u_k_ip1_j + u_k_i_jp1) / 4;
        
        // printf("i=%ld, j=%ld, u(k+1)_i_j=%10f\n", i, j, arr_u_k_plus_1[idx_i_j]);
      
      }
    }
  
  // wait for ghost points and finished the calculation for the boundary
  // ghost from the left
  if(s->ip[0] != 0){
    
    MPI_Wait(&recFromLeftRequest, &recFromLeftStatus);
    
    for(long k = 0; k < rowNum; k++){
      
      // receive ghost points
      long i = 0;
      long j = s->ic_start[1] + k;
      long idx_i_j = calcRowMajorIndex(i, j, n);
      arr_u_k[idx_i_j] = recFromLeftBuffer[k];
      
    }
    
  }
  
  // ghost from the right
  if(s->ip[0] != s->np[0] - 1){
    
    MPI_Wait(&recFromRightRequest, &recFromRightStatus);
    
    for(long k = 0; k < rowNum; k++){
      
      // receive ghost points
      long i = s->ic_stop[0];
      long j = s->ic_start[1] + k;
      long idx_i_j = calcRowMajorIndex(i, j, n);
      double temp = recFromRightBuffer[k];
      arr_u_k[idx_i_j] = temp;
      
    }
    // free(recFromRightBuffer);
  }
  
  // ghost from the up
  if(s->ip[1] != s->np[1] - 1){
    
    MPI_Wait(&recFromUpRequest, &recFromUpStatus);
    
    for(long k = 0; k < colNum; k++){
      
      // receive ghost points
      long i = s->ic_start[0] + k;
      long j = s->ic_stop[1];
      long idx_i_j = calcRowMajorIndex(i, j, n);
      arr_u_k[idx_i_j] = recFromUpBuffer[k];
      
    }
    
  }
  
  // ghost from the down
  if(s->ip[1] != 0){
    
    MPI_Wait(&recFromDownRequest, &recFromDownStatus);
    
    for(long k = 0; k < colNum; k++){
      
      // receive ghost points
      long i = s->ic_start[0] + k;
      long j = 0;
      long idx_i_j = calcRowMajorIndex(i, j, n);
      arr_u_k[idx_i_j] = recFromDownBuffer[k];
      
    }
    
  }
  
  // update boundary points
  for(long i = s->ic_start[0]; i < s->ic_stop[0]; i++){
    for(long j = s->ic_start[1]; j < s->ic_stop[1]; j++){
      if(i == s->ic_start[0] || i == s->ic_stop[0] - 1 || j == s->ic_start[1] || j == s->ic_stop[1] - 1){
        
        // update boundary points
        double f_i_j = 1;
        double u_k_im1_j = 0;
        double u_k_i_jm1 = 0;
        double u_k_ip1_j = 0;
        double u_k_i_jp1 = 0;
        
        if(s->ip[0] != 0 || i > row_start_idx){
          long idx_im1_j = calcRowMajorIndex(i - 1, j, n);
          u_k_im1_j = arr_u_k[idx_im1_j];
        }
        
        if(s->ip[1] != 0 || j > col_start_idx){
          long idx_i_jm1 = calcRowMajorIndex(i, j - 1, n);
          u_k_i_jm1 = arr_u_k[idx_i_jm1];
        }
        
        if(s->ip[0] != (s->np[0] - 1) || i < row_stop_idx - 1){
          long idx_ip1_j = calcRowMajorIndex(i + 1, j, n);
          u_k_ip1_j = arr_u_k[idx_ip1_j];
          // printf("arr_u_k[%ld]=%10f\n", idx_ip1_j, u_k_ip1_j);
        }
        
        if(s->ip[1] != (s->np[1] - 1) || j < col_stop_idx - 1){
          long idx_i_jp1 = calcRowMajorIndex(i, j + 1, n);
          u_k_i_jp1 = arr_u_k[idx_i_jp1];
          // printf("arr_u_k[%ld]=%10f\n", idx_i_jp1, u_k_i_jp1);
        }
        
        long idx_i_j = calcRowMajorIndex(i, j, n);
        arr_u_k_plus_1[idx_i_j] = (std::pow(h, 2) * f_i_j + u_k_im1_j + u_k_i_jm1 + u_k_ip1_j + u_k_i_jp1) / 4;
        
        // printf("i=%ld, j=%ld, u(k+1)_i_j=%10f\n", i, j, arr_u_k_plus_1[idx_i_j]);
        
      }
    }
  }
  
}

double calculateResidualNorm(SubDomain *s, double* arr_u, double h){
  
  // calculate residual matrix A * u(k) - f
  double norm = 0;
  double localResidualSqSum = 0;
  double globalResidualSqSum = 0;
  
  long n = s->ic_number[0];
  long row_start_idx = s->ic_start[0];
  long row_stop_idx = s->ic_stop[0];
  long col_start_idx = s->ic_start[1];
  long col_stop_idx = s->ic_stop[1];
  
  // use the Jacobi method
  MPI_Status sendToLeftStatus, sendToRightStatus, sendToUpStatus, sendToDownStatus, recFromLeftStatus, recFromRightStatus, recFromUpStatus, recFromDownStatus;
  MPI_Request sendToLeftRequest, sendToRightRequest, sendToUpRequest, sendToDownRequest, recFromLeftRequest, recFromRightRequest, recFromUpRequest, recFromDownRequest;
  
  // long parRowStartIdx, parRowStopIdx, parColStartIdx, parColStopIdx; // the index of the starting point and stopping point which do not need to wait for communicating finished
  
  long rowNum = s->ic_stop[1] - s->ic_start[1];
  long colNum = s->ic_stop[0] - s->ic_start[0];
  
  // communicate with(send & recieve) adjacent subdomain to get ghost points(with no block)
  // Send to left(if ip[0] != 0): ( ic_start[0], ic_start[1] ~ (ic_stop[1] - 1) )
  // from left(if ip[0] != 0): put in ( 0, ic_start[1] ~ (ic_stop[1] - 1) )
  if(s->ip[0] != 0){
    
    for(long k = 0; k < rowNum; k++){
      
      long i = s->ic_start[0];
      long j = s->ic_start[1] + k;
      long idx_i_j = calcRowMajorIndex(i, j, n);
      sendToLeftBuffer[k] = arr_u[idx_i_j];
      // printf("[rank=%d] sendToLeftBuffer[%ld] = arr_u[%ld] = %10f\n", s->mpiRank, k, idx_i_j, arr_u[idx_i_j]);
      
    }
    
    MPI_Isend(sendToLeftBuffer, rowNum, MPI_DOUBLE, s->ip_lower[0], MPI_TAG_SEND_TO_LEFT, MPI_COMM_WORLD, &sendToLeftRequest);
    MPI_Irecv(recFromLeftBuffer, rowNum, MPI_DOUBLE, s->ip_lower[0], MPI_TAG_REC_FROM_LEFT, MPI_COMM_WORLD, &recFromLeftRequest);
    
  }
  
  // to right(if ip[0] != np[0] - 1): ( (ic_stop[0] - 1), ic_start[1] ~ (ic_stop[1] - 1) )
  // from right(if ip[0] != np[0] - 1): put in ( ic_stop[0], ic_start[1] ~ (ic_stop[1] - 1) )
  if(s->ip[0] != s->np[0] - 1){
    
    for(long k = 0; k < rowNum; k++){
      
      long i = s->ic_stop[0] - 1;
      long j = s->ic_start[1] + k;
      long idx_i_j = calcRowMajorIndex(i, j, n);
      sendToRightBuffer[k] = arr_u[idx_i_j];
      
    }
    
    MPI_Isend(sendToRightBuffer, rowNum, MPI_DOUBLE, s->ip_upper[0], MPI_TAG_SEND_TO_RIGHT, MPI_COMM_WORLD, &sendToRightRequest);
    MPI_Irecv(recFromRightBuffer, rowNum, MPI_DOUBLE, s->ip_upper[0], MPI_TAG_REC_FROM_RIGHT, MPI_COMM_WORLD, &recFromRightRequest);
    
  }
  
  // to up(if ip[1] != np[1] - 1): ( ic_start[0] ~ (ic_stop[0] - 1), (ic_stop[1] - 1) )
  // from up(if ip[1] != np[1] - 1): put in ( ic_start[0] ~ (ic_stop[0] - 1), ic_stop[1]  )
  if(s->ip[1] != s->np[1] - 1){
    
    for(long k = 0; k < colNum; k++){
      
      long i = s->ic_start[0] + k;
      long j = s->ic_stop[1] - 1;
      long idx_i_j = calcRowMajorIndex(i, j, n);
      sendToUpBuffer[k] = arr_u[idx_i_j];
      
    }
    
    MPI_Isend(sendToUpBuffer, colNum, MPI_DOUBLE, s->ip_upper[1], MPI_TAG_SEND_TO_UP, MPI_COMM_WORLD, &sendToUpRequest);
    MPI_Irecv(recFromUpBuffer, colNum, MPI_DOUBLE, s->ip_upper[1], MPI_TAG_REC_FROM_UP, MPI_COMM_WORLD, &recFromUpRequest);
    
  }
  
  // Send to down(if ip[1] != 0): ( ic_start[0] ~ (ic_stop[0] - 1), ic_start[1] )
  // Rec from down(if ip[1] != 0): put in ( ic_start[0] ~ (ic_stop[0] - 1), 0 )
  if(s->ip[1] != 0){
    
    for(long k = 0; k < colNum; k++){
      
      long i = s->ic_start[0] + k;
      long j = s->ic_start[1];
      long idx_i_j = calcRowMajorIndex(i, j, n);
      sendToDownBuffer[k] = arr_u[idx_i_j];
      // printf("[rank=%d] sendToDownBuffer[%ld] = arr_u[%ld] = %10f\n", s->mpiRank, k, idx_i_j, arr_u[idx_i_j]);
      
    }
    
    MPI_Isend(sendToDownBuffer, colNum, MPI_DOUBLE, s->ip_lower[1], MPI_TAG_SEND_TO_DOWN, MPI_COMM_WORLD, &sendToDownRequest);
    MPI_Irecv(recFromDownBuffer, colNum, MPI_DOUBLE, s->ip_lower[1], MPI_TAG_REC_FROM_DOWN, MPI_COMM_WORLD, &recFromDownRequest);
    
  }
  
  /*
  if(s->mpiRank == 0){
    printf("parRowStartIdx=%ld, parRowStopIdx=%ld, parColStartIdx=%ld, parColStopIdx=%ld\n", parRowStartIdx, parRowStopIdx, parColStartIdx, parColStopIdx);
  }
  */
  
  for(long i = s->ic_start[0] + 1; i < s->ic_stop[0] - 1; i++){
    for(long j = s->ic_start[1] + 1; j < s->ic_stop[1] - 1; j++){
      
      double f_i_j = 1;
      double u_k_im1_j = 0;
      double u_k_i_jm1 = 0;
      double u_k_i_j = 0;
      double u_k_ip1_j = 0;
      double u_k_i_jp1 = 0;
      
      long idx_i_j = calcRowMajorIndex(i, j, n);
      u_k_i_j = arr_u[idx_i_j];
      
      // if(s->mpiRank == 0){
      // printf("[rank=%d] (i, j) = (%ld, %ld), u_k_i_j=arr_u[%ld]=%10f\n", s->mpiRank, i, j, idx_i_j, u_k_i_j);
      // }
      
      if(s->ip[0] != 0 || i > row_start_idx){
        long idx_im1_j = calcRowMajorIndex(i - 1, j, n);
        u_k_im1_j = arr_u[idx_im1_j];
        // if(s->mpiRank == 0){
        // printf("[rank=%d] (i, j) = (%ld, %ld), u_k_im1_j=arr_u[%ld]=%10f\n", s->mpiRank, i, j, idx_im1_j, u_k_im1_j);
        // }
      }
      
      if(s->ip[1] != 0 || j > col_start_idx){
        long idx_i_jm1 = calcRowMajorIndex(i, j - 1, n);
        u_k_i_jm1 = arr_u[idx_i_jm1];
        // if(s->mpiRank == 0){
        // printf("[rank=%d] (i, j) = (%ld, %ld), u_k_i_jm1=arr_u[%ld]=%10f\n", s->mpiRank, i, j, idx_i_jm1, u_k_i_jm1);
        // }
      }
      
      if(s->ip[0] != (s->np[0] - 1) || i < row_stop_idx - 1){
        long idx_ip1_j = calcRowMajorIndex(i + 1, j, n);
        u_k_ip1_j = arr_u[idx_ip1_j];
        // if(s->mpiRank == 0){
        // printf("[rank=%d] (i, j) = (%ld, %ld), u_k_ip1_j=arr_u[%ld]=%10f\n", s->mpiRank, i, j, idx_ip1_j, u_k_ip1_j);
        // }
      }
      
      if(s->ip[1] != (s->np[1] - 1) || j < col_stop_idx - 1){
        long idx_i_jp1 = calcRowMajorIndex(i, j + 1, n);
        u_k_i_jp1 = arr_u[idx_i_jp1];
        // if(s->mpiRank == 0){
        // printf("[rank=%d] (i, j) = (%ld, %ld), u_k_i_jp1=arr_u[%ld]=%10f\n", s->mpiRank, i, j, idx_i_jp1, u_k_i_jp1);
        // }
      }
      
      // residual = f_i_j - ((- u_im1_j - u_i_jm1 + 4 u_i_j - u_ip1_j - u_i_jp1)/(h ^ 2))
      double a_mult_u = (-1 * u_k_im1_j - u_k_i_jm1 + 4 * u_k_i_j - u_k_ip1_j - u_k_i_jp1) / (std::pow(h, 2.0));
      // printf("a_mult_u=%10f\n", a_mult_u);
      double residual = f_i_j - a_mult_u;
      localResidualSqSum += std::pow(residual, 2);
      // if(s->mpiRank == 0){
      // printf("inner part [rank=%d] residualSqSum=%10f\n", s->mpiRank, localResidualSqSum);
      // printf("[rank=%d] i=%ld, j=%ld, residual=%10f\n", s->mpiRank, i, j ,residual);
      // printf("[rank=%d] f_i_j=%10f, u_k_i_j=%10f, u_k_im1_j=%10f, u_k_i_jm1=%10f, u_k_ip1_j=%10f, u_k_i_jp1=%10f\n", s->mpiRank, f_i_j, u_k_i_j, u_k_im1_j, u_k_i_jm1, u_k_ip1_j, u_k_i_jp1);
      // }
      
    }
  }
  
  // wait for ghost points and finished the calculation for the boundary
  // ghost from the left
  if(s->ip[0] != 0){
    
    MPI_Wait(&recFromLeftRequest, &recFromLeftStatus);
    
    for(long k = 0; k < rowNum; k++){
      // receive ghost points
      long i = 0;
      long j = s->ic_start[1] + k;
      long idx_i_j = calcRowMajorIndex(i, j, n);
      arr_u[idx_i_j] = recFromLeftBuffer[k];
    }
    
  }
  
  // ghost from the right
  if(s->ip[0] != s->np[0] - 1){
    
    MPI_Wait(&recFromRightRequest, &recFromRightStatus);
    
    for(long k = 0; k < rowNum; k++){
      
      // receive ghost points
      long i = s->ic_stop[0];
      long j = s->ic_start[1] + k;
      long idx_i_j = calcRowMajorIndex(i, j, n);
      double temp = recFromRightBuffer[k];
      arr_u[idx_i_j] = temp;
      
    }
    // free(recFromRightBuffer);
  }
  
  // ghost from the up
  if(s->ip[1] != s->np[1] - 1){
    
    MPI_Wait(&recFromUpRequest, &recFromUpStatus);
    
    for(long k = 0; k < colNum; k++){
      
      // receive ghost points
      long i = s->ic_start[0] + k;
      long j = s->ic_stop[1];
      long idx_i_j = calcRowMajorIndex(i, j, n);
      arr_u[idx_i_j] = recFromUpBuffer[k];
      
    }
    
  }
  
  // ghost from the down
  if(s->ip[1] != 0){
    
    MPI_Wait(&recFromDownRequest, &recFromDownStatus);
    
    for(long k = 0; k < colNum; k++){
      
      // receive ghost points
      long i = s->ic_start[0] + k;
      long j = 0;
      long idx_i_j = calcRowMajorIndex(i, j, n);
      arr_u[idx_i_j] = recFromDownBuffer[k];
      
    }
    
  }
  
  // calculate the boarder
  // update boundary points
  for(long i = s->ic_start[0]; i < s->ic_stop[0]; i++){
    for(long j = s->ic_start[1]; j < s->ic_stop[1]; j++){
      if(i == s->ic_start[0] || i == s->ic_stop[0] - 1 || j == s->ic_start[1] || j == s->ic_stop[1] - 1){
        
        double f_i_j = 1;
        double u_k_im1_j = 0;
        double u_k_i_jm1 = 0;
        double u_k_i_j = 0;
        double u_k_ip1_j = 0;
        double u_k_i_jp1 = 0;
        
        long idx_i_j = calcRowMajorIndex(i, j, n);
        u_k_i_j = arr_u[idx_i_j];
        // if(s->mpiRank == 0){
        // printf("[rank=%d] (i, j) = (%ld, %ld), u_k_i_j=arr_u[%ld]=%10f\n", s->mpiRank, i, j, idx_i_j, u_k_i_j);
        // }
        
        if(s->ip[0] != 0 || i > row_start_idx){
          long idx_im1_j = calcRowMajorIndex(i - 1, j, n);
          u_k_im1_j = arr_u[idx_im1_j];
          // if(s->mpiRank == 0){
          // printf("[rank=%d] (i, j) = (%ld, %ld), u_k_im1_j=arr_u[%ld]=%10f\n", s->mpiRank, i, j, idx_im1_j, u_k_im1_j);
          // }
        }
        
        if(s->ip[1] != 0 || j > col_start_idx){
          long idx_i_jm1 = calcRowMajorIndex(i, j - 1, n);
          u_k_i_jm1 = arr_u[idx_i_jm1];
          // if(s->mpiRank == 0){
          // printf("[rank=%d] (i, j) = (%ld, %ld), u_k_i_jm1=arr_u[%ld]=%10f\n", s->mpiRank, i, j, idx_i_jm1, u_k_i_jm1);
          // }
        }
        
        if(s->ip[0] != (s->np[0] - 1) || i < row_stop_idx - 1){
          long idx_ip1_j = calcRowMajorIndex(i + 1, j, n);
          u_k_ip1_j = arr_u[idx_ip1_j];
          // if(s->mpiRank == 0){
          // printf("[rank=%d] (i, j) = (%ld, %ld), u_k_ip1_j=arr_u[%ld]=%10f\n", s->mpiRank, i, j, idx_ip1_j, u_k_ip1_j);
          // }
        }
        
        if(s->ip[1] != (s->np[1] - 1) || j < col_stop_idx - 1){
          long idx_i_jp1 = calcRowMajorIndex(i, j + 1, n);
          u_k_i_jp1 = arr_u[idx_i_jp1];
          // if(s->mpiRank == 0){
          // printf("[rank=%d] (i, j) = (%ld, %ld), u_k_i_jp1=arr_u[%ld]=%10f\n", s->mpiRank, i, j, idx_i_jp1, u_k_i_jp1);
          // }
        }
        
        // residual = f_i_j - ((- u_im1_j - u_i_jm1 + 4 u_i_j - u_ip1_j - u_i_jp1)/(h ^ 2))
        double a_mult_u = (-1 * u_k_im1_j - u_k_i_jm1 + 4 * u_k_i_j - u_k_ip1_j - u_k_i_jp1) / (std::pow(h, 2.0));
        // printf("a_mult_u=%10f\n", a_mult_u);
        double residual = f_i_j - a_mult_u;
        localResidualSqSum += std::pow(residual, 2);
        // if(s->mpiRank == 0){
        // printf("[rank=%d] residualSqSum=%10f\n", s->mpiRank, localResidualSqSum);
        // printf("[rank=%d] i=%ld, j=%ld, residual=%10f\n", s->mpiRank, i, j ,residual);
        // printf("[rank=%d] f_i_j=%10f, u_k_i_j=%10f, u_k_im1_j=%10f, u_k_i_jm1=%10f, u_k_ip1_j=%10f, u_k_i_jp1=%10f\n", s->mpiRank, f_i_j, u_k_i_j, u_k_im1_j, u_k_i_jm1, u_k_ip1_j, u_k_i_jp1);
        // }
        
      }
    }
  }
  
  // if(s->mpiRank == 0){
  // printf("[rank=%d] localResidualSqSum=%10f\n", s->mpiRank, localResidualSqSum);
  // }
  MPI_Allreduce(&localResidualSqSum, &globalResidualSqSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(globalResidualSqSum);
  
}

double* processNextIter(int iterNumber, double initialNorm, SubDomain *s, double h, double* arr_u_k, double* arr_u_k_plus_1){
  
  calculate_next_u(s, arr_u_k, arr_u_k_plus_1, h);
  
  double thisNorm = calculateResidualNorm(s, arr_u_k_plus_1, h);
  double decreasingFactor = initialNorm / thisNorm;
  if(s->mpiRank == 0){
    printf("Iter[%d]: norm=%10f, decreasingFactor=%10f\n", iterNumber, thisNorm, decreasingFactor);
  }
  
  bool greaterThanStopCond = false;
  if(decreasingFactor > std::pow(10, 6)){
    greaterThanStopCond = true;
  }
  
  // terminate the iteration when the initial residual is decreased by a factor of 106 or after 5000 iterations.
  if(greaterThanStopCond || (iterNumber >= stopAfterIterNum)){
    return arr_u_k_plus_1;
  }else{
    
    iterNumber++;
    double *swap = arr_u_k;
    arr_u_k = arr_u_k_plus_1;
    arr_u_k_plus_1 = swap;
    double* arr_u_result = processNextIter(iterNumber, initialNorm, s, h, arr_u_k, arr_u_k_plus_1);
    return arr_u_result;
    
  }
  
}

int main(int argc, char *argv[]){

  SubDomain s;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &s.mpiRank);
  MPI_Comm_size(MPI_COMM_WORLD, &s.numProcs);
  
  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  
  // check the value of total process number is fill the contion p = 4^j
  // : check sqrt(p) is an even integer
  double powerOf4 = log2(s.numProcs) / 2;
  if(s.mpiRank == 0){
    printf("powerOf4=%10f\n", powerOf4);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  if(s.mpiRank == 0 && (powerOf4 != floor(powerOf4))){
    printf("p is not 4 ^ j: p=%d\n", s.numProcs);
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  
  for(int d = 0; d < DIM; d++){
    s.np[d] = pow(2, powerOf4);
  }
  
  // determine position of myrank in the process mesh np[d]
  int ipTemp[DIM];
  inverseIndex(s.mpiRank, s.np, s.ip);
  for(int d = 0; d < DIM; d++){
    ipTemp[d] = s.ip[d];
  }
                
  // determine neighborhood processes
  for(int d = 0; d < DIM; d++){
    ipTemp[d] = (s.ip[d] - 1 + s.np[d]) % (s.np[d]);
    s.ip_lower[d] = index(ipTemp, s.np);
    ipTemp[d] = (s.ip[d] + 1 + s.np[d]) % (s.np[d]);
    s.ip_upper[d] = index(ipTemp, s.np);
    ipTemp[d] = s.ip[d];
  }
  
  // printf("Rank %d/%d(ip=(%d, %d)) running on %s.\n", s.mpiRank, s.numProcs, s.ip[0], s.ip[1], processor_name);
  MPI_Barrier(MPI_COMM_WORLD);
  
  // get input params
  long N_local = 4;
  if(argc >= 2){
    N_local = atoi(argv[1]);
  }
  
  if(argc >= 3){
    stopAfterIterNum = atoi(argv[2]);
  }
  
  long range = 1;
  for(int d = 0; d < DIM; d++){
    s.ic_start[d] = 1;
    s.ic_stop[d] = s.ic_start[d] + N_local;
    s.ic_number[d] = N_local + 2; // +2 : space for ghost points
    range *= s.ic_number[d];
  }
  
  // calculate h according to the input N
  long int_powerOf4 = (long)powerOf4;
  long N = pow(2, int_powerOf4) * N_local;
  double h = (double)1 / (N + 1);
  if(s.mpiRank == 0){
    printf("N=%ld, h=%10f, stopAfterIterNum=%d\n", N, h, stopAfterIterNum);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  // initialization
  // set a initialzation vector u^0
  int iterNumber = 0;
  double* arr_u1 = (double*) aligned_malloc(range * sizeof(double));
  double* arr_u2 = (double*) aligned_malloc(range * sizeof(double));
  for(long i = 0; i < range; i++){
    arr_u1[i] = 0;
    arr_u2[i] = 0;
  }
  
  long rowNum = s.ic_stop[1] - s.ic_start[1];
  long colNum = s.ic_stop[0] - s.ic_start[0];
  sendToLeftBuffer = (double*)malloc(rowNum * sizeof(double));
  recFromLeftBuffer = (double*)malloc(rowNum * sizeof(double));
  sendToRightBuffer = (double*)malloc(rowNum * sizeof(double));
  recFromRightBuffer = (double*)malloc(rowNum * sizeof(double));
  sendToUpBuffer = (double*)malloc(colNum * sizeof(double));
  recFromUpBuffer = (double*)malloc(colNum * sizeof(double));
  sendToDownBuffer = (double*)malloc(colNum * sizeof(double));
  recFromDownBuffer = (double*)malloc(colNum * sizeof(double));
  
  double norm_0 = calculateResidualNorm(&s, arr_u1, h);
  if(s.mpiRank == 0){
    printf("norm_0=%10f\n", norm_0);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  iterNumber++;
  
  double tt = MPI_Wtime();
  double *arr_u_result = processNextIter(iterNumber, norm_0, &s, h, arr_u1, arr_u2);
  double elapsed = MPI_Wtime() - tt;
  // printf("[rank=%d] done\n", s.mpiRank);
  MPI_Barrier(MPI_COMM_WORLD);
  
  /*
  long rowIdxBase = s.ip[0] * N_local;
  long colIdxBase = s.ip[1] * N_local;
  long n = s.ic_number[0];
  for(int i = 0; i < s.numProcs; i++){
    // printf("[rank=%d] rowIdxBase=%ld, colIdxBase=%ld\n", s.mpiRank, rowIdxBase, colIdxBase);
    if(s.mpiRank == i){
      printf("[rank=%d] arr_u_result:\n", s.mpiRank);
      for(long i = s.ic_start[0]; i < s.ic_stop[0]; i++){
        for(long j = s.ic_start[1]; j < s.ic_stop[1]; j++){
          long idx_i_j = calcRowMajorIndex(i, j, n);
          printf("u(%ld, %ld)=%10f\n", rowIdxBase + (i - s.ic_start[0]), colIdxBase + (j - s.ic_start[1]), arr_u_result[idx_i_j]);
        }
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  */
  
  if(s.mpiRank == 0){
    printf("N=%ld, run time = %10f second(s).\n", N, elapsed);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  aligned_free(arr_u1);
  aligned_free(arr_u2);
  
  free(sendToLeftBuffer);
  free(recFromLeftBuffer);
  free(sendToRightBuffer);
  free(recFromRightBuffer);
  free(sendToUpBuffer);
  free(recFromUpBuffer);
  free(sendToDownBuffer);
  free(recFromDownBuffer);
  
  MPI_Finalize();
  return 0;

}
