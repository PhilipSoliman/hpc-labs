/******************************************************************************
* FILE: mm.c
* DESCRIPTION:  
*   This program calculates the product of matrix a[nra][nca] and b[nca][ncb],
*   the result is stored in matrix c[nra][ncb].
*   The max dimension of the matrix is constraint with static array declaration,
*   for a larger matrix you may consider dynamic allocation of the arrays, 
*   but it makes a parallel code much more complicated (think of communication),
*   so this is only optional.
*   
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 1000

int main (int argc, char *argv[]) 
{
  printf("Matrix Multiplication \n");

  int tid, nthreads, i, j, k;
  int NRA=N;
  int NCA=N;
  int NCB=N;

  double  a[NRA][NCA],           /* matrix A to be multiplied */
          b[NCA][NCB],           /* matrix B to be multiplied */
          c[NRA][NCB],           /* result matrix C */
          c_seq[NRA][NCB],       /* result matrix C (sequential) */
          timeSeq, timePar,      /* timing variables */
          startTime;    /* timing variables */

  // Initialize matrices
  for (i=0; i<NRA; i++)
    for (j=0; j<NCA; j++)
      a[i][j]= i+j;
  
  for (i=0; i<NCA; i++)
    for (j=0; j<NCB; j++)
      b[i][j]= i*j;
  
  for (i=0; i<NRA; i++)
    for (j=0; j<NCB; j++)
        c[i][j]= 0;
        c_seq[i][j]= 0;

  // Variables for the process rank and number of processes
  int myRank, numProcs, i, j;
  MPI_Status status;

  // Initialize MPI, find out MPI communicator size and process rank
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  // perform sequential matrix multiplication
  startTime = MPI_Wtime();
  if (myRank == 0){
    for (i=0; i<NRA; i++)    
      {
      for(j=0; j<NCB; j++)       
        for (k=0; k<NCA; k++)
          c_seq[i][j] += a[i][k] * b[k][j];
      } 
  }
  timeSeq = MPI_Wtime()-startTime;

  if (numProcs < 2)
  {
      printf("Error: Run the program with at least 2 MPI tasks!\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
      exit(1);
  }

  /*perform parallel matrix multiplication
    Steps:
    1) Rank 0 sends the rows of matrix a and columns of matrix b to different processes (MPI_Send);
    2) All other ranks receive the rows of matrix a and columns of matrix b from 0 (MPI_Recv);
    3) Each process computes a part of the result matrix c; 
    4) Use MPI_reduceAll to collect the results in the master process, which sums up the partial 
      results into the final result matrix c.
    5) Implement time measurement (MPI_Wtime). 
    6) Compare the results with the sequential version. 
  */

  startTime = MPI_Wtime();
  // step 1:
  if (myRank == 0)
  {
    // send rows of matrix a
    for (i=1; i<numProcs; i++)
    {
      MPI_Send(a[i], NCA, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
    }
    // send columns of matrix b
    for (i=1; i<numProcs; i++)
    {
      MPI_Send(b, NCA*NCB, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
    }
  }

  // step 2: TODO: check if this is correct
  if (myRank > 0)
  {
    // receive rows of matrix a
    MPI_Recv(a, NCA, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    // receive columns of matrix b
    MPI_Recv(b, NCA*NCB, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
  }

  // step 3: TODO: check if this is correct
  for (i=0; i<NRA; i++)    
  {
    for(j=0; j<NCB; j++)       
      for (k=0; k<NCA; k++)
        c[i][j] += a[i][k] * b[k][j];
  }

  // step 4: TODO: check if this is correct
  if (myRank == 0)
  {
    // receive partial results from other processes
    for (i=1; i<numProcs; i++)
    {
      MPI_Recv(c[i], NCA*NCB, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
    }
  }
  else
  {
    // send partial results to master process
    MPI_Send(c, NCA*NCB, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
  timePar = MPI_Wtime()-startTime;

  // step 6: 
  if (myRank == 0)
  {
    // compare results
    for (i=0; i<NRA; i++)
    {
      for (j=0; j<NCB; j++)
      {
        if (c[i][j] != c_seq[i][j])
        {
          printf("Error: Result mismatch!\n");
          MPI_Abort(MPI_COMM_WORLD, 1);
          exit(1);
        }
      }
    }
  }

  // print results
  if (myRank == 0)
  {
    printf("Matrix multiplication completed!\n");
    printf("\tMatrix size: %d\n", N);
    printf("\tNumber of processes: %d\n", numProcs);
    printf("\tSequential version: %f seconds\n", timeSeq);
    printf("\tParallel version: %f seconds\n", timePar);
    printf("\tSpeedup: %f\n", timeSeq/timePar);
  }
  
  // Finalize MPI
  MPI_Finalize();
  return 0; 
}


