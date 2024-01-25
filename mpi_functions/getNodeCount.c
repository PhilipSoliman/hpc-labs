#include <stdio.h>
#include <mpi.h>
#include "mpifuncs.h"

/*
Function to get the number of nodes from MPI_COMM_WORLD
obtained from: https://stackoverflow.com/questions/34115227/how-to-get-the-number-of-physical-machine-in-mpi
*/
int MPI_getNodeCount(void)
{
    int rank, is_rank0, nodes;
    MPI_Comm shmcomm;

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &shmcomm);

    MPI_Comm_rank(shmcomm, &rank);

    is_rank0 = (rank == 0) ? 1 : 0;

    MPI_Allreduce(&is_rank0, &nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    MPI_Comm_free(&shmcomm);

    return nodes;
}
