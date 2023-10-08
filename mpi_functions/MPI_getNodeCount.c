#include <stdio.h>
#include "mpifuncs.h"
#include <mpi.h>


/*
Function to get the number of nodes from MPI_COMM_WORLD
obtained from: https://stackoverflow.com/questions/34115227/how-to-get-the-number-of-physical-machine-in-mpi
*/ 
int MPI_getNodeCount(void)
{
    printf("Getting number of nodes:\n");

    int rank, is_rank0, nodes;
    MPI_Comm shmcomm;

    printf("\tSplitting MPI_COMM_WORLD into shared memory communicator...\n");
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &shmcomm);

    printf("\tGetting rank of each process in shared memory communicator...\n");
    MPI_Comm_rank(shmcomm, &rank);

    printf("\tChecking if rank is 0...\n");
    is_rank0 = (rank == 0) ? 1 : 0;

    printf("\tSumming up all ranks...\n");
    MPI_Allreduce(&is_rank0, &nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    printf("\tNumber of nodes: %i\n", nodes);

    printf("\tFreeing shared memory communicator...\n");
    MPI_Comm_free(&shmcomm);

    printf("\tDone!\n");
    return nodes;
}

