/*
////////////////////////////////////////////////////////////////////
This is a specific example on how to run a MPI program successfully.
////////////////////////////////////////////////////////////////////
*/

/*
--Log onto DelftBlue
*/

#include "mpi.h"
#include <stdio.h>

int np, rank;

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  printf("Task %d of %d tasks says: Hello world!\n", rank, np);

  MPI_Finalize();
  return 0;
}

/*
====================================================================
REMARKS
This is all done in the sh, please modify the number of processes in the .sh file

--Load the two modules
====================================================================
module load 2022r2 openmpi
====================================================================

--Compile the helloworld program
====================================================================
mpicc -o helloworld helloworld.c
====================================================================

--TODO Run the helloworld program with 2 processes, 4 cores for each process
====================================================================
modify the helloworld.sh file sbatch directive
====================================================================

--Welcome to this fancy MPI world!
*/