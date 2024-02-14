#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "dataxtract.h"
#include "mpifuncs.h"

// Maximum array size 2^20 = 1048576 elements
#define MAX_ARRAY_SIZE_LEFT_SHIFT 20
#define MAX_ARRAY_SIZE 1<<MAX_ARRAY_SIZE_LEFT_SHIFT
#define ASSIGNMENT_FOLDER "/home/psoliman/HPC/hpc-labs/intro/pingPong_times/"

int main(int argc, char **argv)
{
    // Variables for the process rank and number of processes
    int myRank, numProcs, i, j;
    MPI_Status status;

    // Initialize MPI, find out MPI communicator size and process rank
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    printf("Rank %2.1i: Hello, world!\n", myRank);

    // Allocate memory for the array
    int *myArray = malloc(sizeof(int) * MAX_ARRAY_SIZE);
    if (myArray == NULL)
    {
        printf("Not enough memory\n");
        exit(1);
    }
    printf("(%i)succesfully allocated memory!\n", myRank);
    // Initialize myArray
    for (i = 0; i < MAX_ARRAY_SIZE; i++)
        myArray[i] = 1;

    int numberOfElementsToSend;
    int numberOfElementsReceived;

    // PART C
    if (numProcs < 2)
    {
        printf("(%i)Error: Run the program with at least 2 MPI tasks!\n", myRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // number of nodes
    int numberOfNodes = MPI_getNodeCount();

    // initialise arrays to save message length, duration and combined duration
    int messageLengthArray[MAX_ARRAY_SIZE_LEFT_SHIFT +1];
    double timeArray[MAX_ARRAY_SIZE_LEFT_SHIFT +1];

    printf("(%i) successfully initialised arrays\n", myRank);
    // double combinedTimeArray[MAX_ARRAY_SIZE_LEFT_SHIFT + 1];
    double startTime, endTime;

    // ping pong loop
    for (j = 0; j < MAX_ARRAY_SIZE_LEFT_SHIFT +1; j++)
    {
        // message size obtained using left-shift operator. Also save it.
        numberOfElementsToSend = 1<<j;

        if (myRank == 0)
        {
            // save message length
            messageLengthArray[j] = sizeof(int) * numberOfElementsToSend;

            printf("Rank %2.1i: Sending %i elements\n",
                   myRank, numberOfElementsToSend);

            myArray[0] = myArray[1] + 1; // activate in cache (avoids possible delay when sending the 1st element)

            startTime = MPI_Wtime();
            for (i = 0; i < 5; i++)
            {
                MPI_Send(myArray, numberOfElementsToSend, MPI_INT, 1, 0,
                         MPI_COMM_WORLD);

                // Probe message in order to obtain the amount of data
                MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                MPI_Get_count(&status, MPI_INT, &numberOfElementsReceived);

                MPI_Recv(myArray, numberOfElementsReceived, MPI_INT, 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } // end of for-loop

            endTime = MPI_Wtime();

            printf("Rank %2.1i: Received %i elements\n",
                   myRank, numberOfElementsReceived);

            // average communication time of 1 send-receive (total 5*2 times)
            printf("Ping Pong took %f seconds\n", (endTime - startTime) / 10);

            // save average time
            timeArray[j] = (endTime - startTime) / 10;
        }
        else if (myRank == 1)
        {
            // Probe message in order to obtain the amount of data
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_INT, &numberOfElementsReceived);

            for (i = 0; i < 5; i++)
            {

                MPI_Recv(myArray, numberOfElementsReceived, MPI_INT, 0, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // printf("Rank %2.1i: Received %i elements\n",
                //     myRank, numberOfElementsReceived);

                // printf("Rank %2.1i: Sending back %i elements\n",
                //     myRank, numberOfElementsToSend);

                MPI_Send(myArray, numberOfElementsToSend, MPI_INT, 0, 0,
                         MPI_COMM_WORLD);
            } // end of for-loop
        }
    }

    // combined ping pong loop send & receive operation using MPI_Sendrecv
    // if (0)
    // {
    //     // create second array to use different buffer for send and receive operations
    //     int *mySecondArray = (int *)malloc(sizeof(int) * MAX_ARRAY_SIZE);
    //     if (mySecondArray == NULL)
    //     {
    //         printf("Not enough memory\n");
    //         exit(1);
    //     }
    //     for (i = 0; i < MAX_ARRAY_SIZE; i++)
    //         mySecondArray[i] = 1;

    //     for (j = 0; j <= MAX_ARRAY_SIZE_LEFT_SHIFT; j++)
    //     {
    //         numberOfElementsToSend = 1 << j;
    //         if (myRank == 0)
    //         {
    //             startTime = MPI_Wtime();
    //             for (i = 0; i < 5; i++)
    //             {
    //                 MPI_Sendrecv(myArray, numberOfElementsToSend, MPI_INT, 1, 0,
    //                              mySecondArray, numberOfElementsToSend, MPI_INT, 1, 0,
    //                              MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //             }
    //             endTime = MPI_Wtime();

    //             // average communication time of 1 combined send-receive (total 5*2 times)
    //             printf("Combined ping Pong took %f seconds\n", (endTime - startTime) / 10);

    //             // save average time
    //             combinedTimeArray[j] = (endTime - startTime) / 10;
    //         }

    //         if (myRank == 1)
    //         {
    //             for (i = 0; i < 5; i++)
    //             {
    //                 MPI_Sendrecv(mySecondArray, numberOfElementsToSend, MPI_INT, 0, 0,
    //                              myArray, numberOfElementsToSend, MPI_INT, 0, 0,
    //                              MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //             }
    //         }
    //     }
    // }

    // free memory
    free(myArray);

    // Finalize MPI
    MPI_Finalize();

    if (myRank == 0)
    {
        // combine message length and duration arrays
        double timeDataArray[MAX_ARRAY_SIZE_LEFT_SHIFT +1][2];
        for (i = 0; i < MAX_ARRAY_SIZE_LEFT_SHIFT +1; i++)
        {
            timeDataArray[i][0] = messageLengthArray[i];
            timeDataArray[i][1] = timeArray[i];
        }
        // int arraySize = 2 * (MAX_ARRAY_SIZE_LEFT_SHIFT + 1);

        // save data to file
        printf("Saving message length and duration data to file...\n");
        char arrayFileName[50];
        sprintf(arrayFileName, "pingPong_times_nnodes=%i.dat", numberOfNodes);
        char fullPath[sizeof(ASSIGNMENT_FOLDER) + sizeof(arrayFileName)] = ASSIGNMENT_FOLDER;
        strcat(fullPath, arrayFileName);
        FILE *f;
        if ((f = fopen(fullPath, "w")) == NULL)
            printf("Error opening file!\n");
        if (fwrite(timeDataArray, sizeof(double), MAX_ARRAY_SIZE_LEFT_SHIFT +1 * 2, f) != 2)
            printf("Error writing to file!\n");
    }

    return 0;
}
