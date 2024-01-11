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
#include <string.h>
#include <mpi.h>
#include "mpifuncs.h"
#include "dataxtract.h"

void printMatrix(void *matrixPtr, int rows, int cols, int showRows, int showCols);

#define MIN_MATRIX_SIZE_P2 7
#define MAX_MATRIX_SIZE_P2 10 // 2^10 = 1024 do not go beyond 10,!
#define ASSIGNMENT_FOLDER "/home/psoliman/HPC/hpc-labs/out/assignment_0b/" // trailing slash is important


int main (int argc, char *argv[]) {

  int tid, nThreads, nRows, nRowsMax, i, j, k,
      NRA, NCA, NCB, N, N_max;
  double timeSeq, timePar,      /* timing variables */
        startTime, timeArray[3][MAX_MATRIX_SIZE_P2-MIN_MATRIX_SIZE_P2+1];      

  N_max = 1<<MAX_MATRIX_SIZE_P2;
  int a[N_max][N_max],           /* matrix A to be multiplied */
    b[N_max][N_max],           /* matrix B to be multiplied */
    c[N_max][N_max],           /* result matrix C */
    c_seq[N_max][N_max];       /* result matrix C (sequential) */

  // Initialize MPI, find out MPI communicator size and process rank
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &tid);
  MPI_Comm_size(MPI_COMM_WORLD, &nThreads);

  MPI_Status status;
  
  for (int n = MIN_MATRIX_SIZE_P2; n <= MAX_MATRIX_SIZE_P2; n++) {
    N=1<<n;
    NRA=N;
    NCA=N;
    NCB=N;

    // (re-)initialize matrices
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

    startTime = MPI_Wtime();
    if (tid == 0){
      for (i=0; i<NRA; i++) {
        for(j=0; j<NCB; j++)       
          for (k=0; k<NCA; k++)
            c_seq[i][j] += a[i][k] * b[k][j];
        } 
    }
    timeSeq = MPI_Wtime()-startTime;

    startTime = MPI_Wtime();

    // step 1: determine portion of matrix a to be multiplied with matrix b by each process.
    nRowsMax = ceil(NRA/nThreads);
    if (tid >= 0 && tid < nThreads-1){
      nRows = nRowsMax;
    } else if (tid == nThreads-1) {
      nRows = (NRA-nRowsMax*tid);
    }

    // step 2:
    int c_temp[nRows][NCB];
    for (i=0; i<nRows; i++)
      for (j=0; j<NCB; j++)
        c_temp[i][j] = 0;
        
    for(i=0; i<nRows; i++){       
      for (j=0; j<NCB; j++){
        for (k=0; k<NCA; k++){
          c_temp[i][j] += a[i+tid*nRowsMax][k] * b[k][j];
        }
      }
    }

    // step 3a: sum up and gather the intermediate results (only send temporary results from the processes to the master process)
    // MPI_Reduce(c_temp, &c[tid*nRowsMax][0], nRows*NCB, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // MPI_Gather(&c_temp, nRows*NCB, MPI_INT, &c[tid*nRowsMax], nRows*NCB, MPI_INT, 0, MPI_COMM_WORLD);
    if (tid != 0) {
      MPI_Send(&c_temp, nRows*NCB, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else {
      // copy results from process 0
      for (i=0; i<nRowsMax; i++) {
        for (j=0; j<NCB; j++) {
          c[i][j] = c_temp[i][j];
        }
      }
      // receive results from other processes
      for (i = 1; i < nThreads; i++) {
        MPI_Recv(&c[i*nRowsMax], nRows*NCB, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
      }
    }

    timePar = MPI_Wtime()-startTime;

    // step 4: compare the results with the sequential version
    if (tid == 0) {

      printf("\nMatrix C (sequential):\n");
      printMatrix(&c_seq, NRA, NCB, 4, 4);
      printf("\nMatrix C (parallel):\n");
      printMatrix(&c, NRA, NCB, 4, 4);

      // print results
      printf("Matrix multiplication completed!\n");
      printf("\tMatrix size: %d\n", N);
      printf("\tNumber of processes: %d\n", nThreads);
      printf("\tSequential version: %f seconds\n", timeSeq);
      printf("\tParallel version: %f seconds\n", timePar);
      printf("\tSpeedup: %f\n", timeSeq/timePar);

      // save results to time array
      timeArray[0][n-MIN_MATRIX_SIZE_P2] = timeSeq;
      timeArray[1][n-MIN_MATRIX_SIZE_P2] = timePar;
      timeArray[2][n-MIN_MATRIX_SIZE_P2] = timeSeq/timePar;
    }

    // synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // save time array to file
  if (tid == 0) {
    printf("Saving sequential-, parallel times and speedup to file...\n");
    char *arrayFileName = (char*)malloc((strlen("time_array_nproc=") + 2)* sizeof(char));
    sprintf(arrayFileName, "time_array_nproc=%i", nThreads);
    char fullPath[sizeof(ASSIGNMENT_FOLDER)+sizeof(arrayFileName)] = ASSIGNMENT_FOLDER;
    strcat(fullPath, arrayFileName); 
    saveArray(&timeArray, 3, MAX_MATRIX_SIZE_P2-MIN_MATRIX_SIZE_P2+1, fullPath);
    printf("Saving done!\n");
  }

  // Finalize MPI
  MPI_Finalize();

  return 0; 
}

void printMatrix(void *matrixPtr, int rows, int cols, int showRows, int showCols) {
  int maxWidth = 30;

  // recreating matrix
  int (*matrix)[rows*cols] = matrixPtr;

  // Print first rows and columns 
  for (int i = 0; i < showRows && i < rows; i++) {
    printf("|");
    for (int j = 0; j < showCols && j < cols; j++) {
      printf("%-*d", maxWidth, matrix[i][j]);
    }
    if (showCols < cols) {
      printf("  ...");
    }
    printf("|\n");
  }

  // Print a horizontal line
  printf("|");
  for (int j = 0; j < showCols && j < cols; j++) {
    for (int k = 0; k < maxWidth; k++) {
      printf("-");
    }
  }
  if (showCols < cols) {
      printf("-----");
  }
  printf("|\n");

  // Print the last rows and columns
  // for (int i = rows - showRows; i < rows; i++) {
  //   printf("|");
  //   if (showCols < cols) {
  //     printf("...  ");
  //   }
  //   for (int j = cols - showCols; j < cols; j++) {
  //     printf("%-*d", maxWidth, matrix[i][j]);
  //   }
  //   printf("|\n");
  // }

}

