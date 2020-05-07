// mpic++ -std=c++11 -O3 -march=native -o ssort ssort.cpp
// mpirun --mca btl_vader_backing_directory /tmp --oversubscribe -np 4 ./ssort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>
#include <string>
#include <cmath>

using std::string;

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N = 10;
  if(argc >= 2){
    N = atoi(argv[1]);
  }
  if(rank == 0){
    printf("N=%d, p=%d\n", N, p);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
    // vec[i] = rand() % 100 + 1;
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);
  MPI_Barrier(MPI_COMM_WORLD);

  double tt = MPI_Wtime();
  
  // sort locally
  std::sort(vec, vec+N);

  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
  int numBesidesSamples = N - (p - 1);
  double rawPartitionSize = (double)numBesidesSamples / p;
  int *sampleVecBuffer = (int*)malloc((p - 1) * sizeof(int));
  
  /*
  if(rank == 0){
    string compareResult = (rawPartitionSize == floor(rawPartitionSize))? "true": "false";
    printf("rawPartitionSize == floor(rawPartitionSize)? %s\n", compareResult.c_str());
  }
  */
  
  if(rawPartitionSize == floor(rawPartitionSize)){
    
    int rawPartitionSize_int = (int)rawPartitionSize;
    
    // rawPartitionSize is an integer
    // EX: N = 7, p = 4, p - 1 = 3, rawPartitionSize = 1
    // x o x o x o x
    for(int k = 0; k < p - 1; k++){
      int splitterIdx = (rawPartitionSize_int + 1) * (k + 1) - 1;
      sampleVecBuffer[k] = vec[splitterIdx];
    }
    
  }else{
    
    int partitionSize1 = (int)floor((double)numBesidesSamples / p);
    int partitionSize2 = (int)ceil((double)numBesidesSamples / p);
    /*
    if(rank == 0){
      printf("p=%d, numBesidesSamples=%d, partitionSize1=%d, partitionSize2=%d\n", p, numBesidesSamples, partitionSize1, partitionSize2);
    }
    */
    
    // calculate the number of partiion with size partitionSize1(x) and partitionSize2(y)
    // x + y = p
    // (partitionSize1 * x) + (partitionSize2 * y) = N - (p - 1)
    int parNumSize2 = ((N - (p - 1)) - (partitionSize1 * p)) / (partitionSize2 - partitionSize1);
    int parNumSize1 = p - parNumSize2;
    /*
    if(rank == 0){
      printf("partitionSize1=%d, parNumSize1=%d, partitionSize2=%d, parNumSize2=%d\n", partitionSize1, parNumSize1, partitionSize2, parNumSize2);
    }
     */
    
    int splitterIdx = -1;
    for(int k = 0; k < p - 1; k++){
      
      int parMemberSize;
      if(parNumSize1 > 0){
        parMemberSize = partitionSize1;
        parNumSize1--;
      }else{
        parMemberSize = partitionSize2;
        parNumSize2--;
      }
      
      splitterIdx += (parMemberSize + 1);
      sampleVecBuffer[k] = vec[splitterIdx];
      
    }
    
  }
  
  /*
  MPI_Barrier(MPI_COMM_WORLD);
  for(int i = 0; i < p; i++){
    if(rank == i){
    
      printf("[rank=%d] vec: ", rank);
      for (int i = 0; i < N; i++) {
        printf("%d",vec[i]);
        if(i != N - 1){
          printf(" ");
        }
      }
      printf("\n");
    
      printf("sampleVecBuffer: ");
      for (int i = 0; i < p - 1; i++) {
        printf("%d",sampleVecBuffer[i]);
        if(i != N - 1){
          printf(" ");
        }
      }
      printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  */
  
  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  int *sampleVecRecvBuffer;
  if(rank == 0){
    sampleVecRecvBuffer = (int*)malloc(p * (p - 1) * sizeof(int));
    /*
    for (int i = 0; i < p * (p - 1); ++i) {
      sampleVecRecvBuffer[i] = 0;
    }
    */
  }
  // MPI_Barrier(MPI_COMM_WORLD);
  
  MPI_Gather(sampleVecBuffer, p - 1, MPI_INT, sampleVecRecvBuffer, p - 1, MPI_INT, 0, MPI_COMM_WORLD);
  // MPI_Barrier(MPI_COMM_WORLD);
  
  /*
  if(rank == 0){
    printf("sampleVecRecvBuffer: ");
    for (int i = 0; i < p * (p - 1); i++) {
      printf("%d", sampleVecRecvBuffer[i]);
      if(i != p * (p - 1) - 1){
        printf(" ");
      }
    }
    printf("\n");
  }
  */
  
  free(sampleVecBuffer);

  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  
  int *globalSplitter = (int*)malloc((p - 1) * sizeof(int));
  if(rank == 0){
    
    // sort locally
    std::sort(sampleVecRecvBuffer, sampleVecRecvBuffer + (p * (p - 1)));
  
    /*
    printf("After sorting, sampleVecRecvBuffer: ");
    for (int i = 0; i < p * (p - 1); ++i) {
      printf("%d", sampleVecRecvBuffer[i]);
      if(i != p * (p - 1) - 1){
        printf(" ");
      }
    }
    printf("\n");
    */
    
    int numBesidesGlobalSplitters = (p * (p - 1)) - (p - 1);
    double rawRecvPartitionSize = (double)numBesidesGlobalSplitters / p;
    
    /*
    string compareRecvResult = (rawRecvPartitionSize == floor(rawRecvPartitionSize))? "true": "false";
    printf("rawRecvPartitionSize == floor(rawRecvPartitionSize)? %s\n", compareRecvResult.c_str());
    */
    
    if(rawRecvPartitionSize == floor(rawRecvPartitionSize)){
      
      int rawRecvPartitionSize_int = (int)rawRecvPartitionSize;
      
      // rawPartitionSize is an integer
      // EX: N = 7, p = 4, p - 1 = 3, rawPartitionSize = 1
      // x o x o x o x
      for(int k = 0; k < p - 1; k++){
        int splitterIdx = (rawRecvPartitionSize_int + 1) * (k + 1) - 1;
        globalSplitter[k] = sampleVecRecvBuffer[splitterIdx];
      }
      
    }else{
      
      int globalSampleParSize1 = (int)floor((double)numBesidesGlobalSplitters / p);
      int globalSampleParSize2 = (int)ceil((double)numBesidesGlobalSplitters / p);
      /*
      if(rank == 0){
        printf("p=%d, numBesidesGlobalSplitters=%d, globalSampleParSize1=%d, globalSampleParSize2=%d\n", p, numBesidesGlobalSplitters, globalSampleParSize1, globalSampleParSize2);
      }
      */
      
      // calculate the number of partiion with size partitionSize1(x) and partitionSize2(y)
      // x + y = p
      // (partitionSize1 * x) + (partitionSize2 * y) = N - (p - 1)
      int globalSampleParNumSize2 = (numBesidesGlobalSplitters - (globalSampleParSize1 * p)) / (globalSampleParSize2 - globalSampleParSize1);
      int globalSampleParNumSize1 = p - globalSampleParNumSize2;
      /*
      if(rank == 0){
        printf("globalSampleParNumSize1=%d, globalSampleParNumSize2=%d\n", globalSampleParNumSize1, globalSampleParNumSize2);
      }
      */
      
      int globalSplitterIdx = -1;
      for(int k = 0; k < p - 1; k++){
        
        int parMemberSize;
        if(globalSampleParNumSize1 > 0){
          parMemberSize = globalSampleParSize1;
          globalSampleParNumSize1--;
        }else{
          parMemberSize = globalSampleParSize2;
          globalSampleParNumSize2--;
        }
        
        globalSplitterIdx += (parMemberSize + 1);
        globalSplitter[k] = sampleVecRecvBuffer[globalSplitterIdx];
        
      }
      
    }
  
    /*
    printf("globalSplitter: ");
    for(int i = 0; i < p - 1; i++){
      printf("%d", globalSplitter[i]);
      if(i != p - 2){
        printf(" ");
      }
    }
    printf("\n");
    */
    
    free(sampleVecRecvBuffer);
    
  }

  // root process broadcasts splitters to all other processes
  MPI_Bcast(globalSplitter, p - 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  /*
  for(int i = 0; i < p; i++){
    if(rank == i){
    
      printf("[rank=%d] globalSplitter: ", rank);
      for (int i = 0; i < p - 1; ++i) {
        printf("%d", globalSplitter[i]);
        if(i != p - 2){
          printf(" ");
        }
      }
      printf("\n");
      
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  */
  
  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  //
  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;
  
  int *sdispls = (int*)malloc(p * sizeof(int));
  int *sendcounts = (int*)malloc(p * sizeof(int));
  
  sdispls[0] = 0;
  for (int i = 1; i < p; i++) {
    
    sdispls[i] = std::lower_bound(vec, vec+N, globalSplitter[i - 1]) - vec;
    sendcounts[i - 1] = sdispls[i] - sdispls[i - 1];
    
    if(i == p - 1){
      sendcounts[i] = N - sdispls[i];
    }
    
  }
  
  /*
  for(int i = 0; i < p; i++){
    if(rank == i){
    
      printf("[rank=%d] sdispls: ", rank);
      for (int i = 0; i < p; i++) {
        printf("%d", sdispls[i]);
        if(i != p - 1){
          printf(" ");
        }
      }
      printf("\n");
      
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  for(int i = 0; i < p; i++){
    if(rank == i){
    
      printf("[rank=%d] sendcounts: ", rank);
      for (int i = 0; i < p; i++) {
        printf("%d", sendcounts[i]);
        if(i != p - 1){
          printf(" ");
        }
      }
      printf("\n");
      
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  */
  
  free(globalSplitter);
  
  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data
  
  int *recvcounts = (int*)malloc(p * sizeof(int));
  MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
  // MPI_Barrier(MPI_COMM_WORLD);
  
  /*
  for(int i = 0; i < p; i++){
    if(rank == i){
    
      printf("[rank=%d] recvcounts: ", rank);
      for (int i = 0; i < p; ++i) {
        printf("%d", recvcounts[i]);
        if(i != p - 1){
          printf(" ");
        }
      }
      printf("\n");
      
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  */
  
  /*
  int MPI_Alltoallv(const void *sendbuf, const int sendcounts[],
  const int sdispls[], MPI_Datatype sendtype,
  void *recvbuf, const int recvcounts[],
  const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm)
  */
  
  int totalRecvNum = 0;
  int *rdispls = (int*)malloc(p * sizeof(int));
  for(int i = 0; i < p; i++){
    
    totalRecvNum += recvcounts[i];
    
    if(i == 0){
      rdispls[i] = 0;
    }else{
      rdispls[i] = rdispls[i - 1] + recvcounts[i - 1];
    }
    
  }
  int *globalSortedVec = (int*)malloc(totalRecvNum * sizeof(int));
  
  /*
  for(int i = 0; i < p; i++){
    if(rank == i){
      printf("[rank=%d] totalRecvNum=%d\n", rank, totalRecvNum);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  for(int i = 0; i < p; i++){
    if(rank == i){
    
      printf("[rank=%d] rdispls: ", rank);
      for (int i = 0; i < p; i++) {
        printf("%d", rdispls[i]);
        if(i != p - 1){
          printf(" ");
        }
      }
      printf("\n");
      
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  */
  
  MPI_Alltoallv(vec, sendcounts, sdispls, MPI_INT, globalSortedVec, recvcounts, rdispls, MPI_INT, MPI_COMM_WORLD);
  
  /*
  for(int i = 0; i < p; i++){
    if(rank == i){
    
      printf("[rank=%d] globalSortedVec: ", rank);
      for (int i = 0; i < totalRecvNum; i++) {
        printf("%d", globalSortedVec[i]);
        if(i != totalRecvNum - 1){
          printf(" ");
        }
      }
      printf("\n");
      
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  */
  
  free(sdispls);
  free(sendcounts);
  free(recvcounts);
  
  // do a local sort of the received data

  std::sort(globalSortedVec, globalSortedVec + totalRecvNum);
  MPI_Barrier(MPI_COMM_WORLD);
  
  double elapsed = MPI_Wtime() - tt;
  if(rank == 0){
    printf("N=%d, elapsed time=%10f\n", N, elapsed);
  }
  
  /*
  for(int i = 0; i < p; i++){
    if(rank == i){
    
      printf("After sorting, [rank=%d] globalSortedVec: ", rank);
      for (int i = 0; i < totalRecvNum; i++) {
        printf("%d", globalSortedVec[i]);
        if(i != totalRecvNum - 1){
          printf(" ");
        }
      }
      printf("\n");
      
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  */
  
  // every process writes its result to a file

  { // Write output to a file
    FILE* fd = NULL;
    char filename[256];
    snprintf(filename, 256, "output%02d.txt", rank);
    fd = fopen(filename,"w+");

    if(NULL == fd) {
      printf("Error opening file \n");
      return 1;
    }

    for(int n = 0; n < totalRecvNum; n++){
      if(n != 0){
        fprintf(fd, " %d", globalSortedVec[n]);
      }else{
        fprintf(fd, "%d", globalSortedVec[n]);
      }
    }

    fclose(fd);
  }
  
  free(vec);
  free(globalSortedVec);
  MPI_Finalize();
  return 0;
  
}
