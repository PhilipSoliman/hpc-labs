
#include <stdio.h>
#include <libgen.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include "dataxtract.h"


void saveArray(void* myArrayPtr, int nRows, int nCols, char fullPath[]){
    int arraySize = nRows * nCols;
    printf("Saving array of size %d to %s\n", arraySize, fullPath);
    FILE *fp;

    // recreating array
    double (*myArray)[arraySize] = myArrayPtr;

    // creating copy of fullPath, as dirname and/or basename may modify the string
    char fullPathCopy[sizeof(*fullPath) + 100];

    // creating directory path and file name
    const char * fileName = basename(fullPath);
    const char * directoryPath = dirname(fullPath);    
    sprintf(fullPathCopy, "%s/%s", directoryPath, fileName);

    // creating directory (read/write/search permissions for owner and group, and with read/search permissions for others)
    printf("Creating directory %s\n...", directoryPath);
    if (mkdir(directoryPath, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0){
        printf("\tDirectory created!\n");
    } else if (errno == EEXIST){
        printf("\tDirectory already exists!\n");
    } else if (errno == ENOENT){
        printf("\tPart of directory does not exist!\n");
        exit(0);
    } else {
        printf("\tError creating directory: %s, errno = %i\n", strerror(errno), errno);
        exit(0);
    }

    // opening file  
    printf("Opening file at %s\n", fullPathCopy);
    if((fp=fopen(fullPathCopy, "wb"))==NULL){
        printf("Cannot open file.\n");
    }

    // writing to file
    printf("Writing %s to folder %s\n", fileName, directoryPath);
    if(fwrite(myArray, sizeof(float), arraySize, fp) != arraySize){
        printf("File write error.");
        exit(1);
    }

    // closing file 
    fclose(fp);
}

