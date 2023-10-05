
#include <stdio.h>
#include <libgen.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include "saveArray.h"

void saveArray(double myArray[][2], int arraySize, char fullPath[]){
    printf("Saving array of size %d to %s\n", arraySize, fullPath);
    FILE *fp;
    const char * fileName = basename(fullPath);
    const char * directoryPath = dirname(fullPath);

    // creating directory (read/write/search permissions for owner and group, and with read/search permissions for others)
    printf("Creating directory %s\n...", directoryPath);
    if (mkdir(directoryPath, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0){
        printf("Directory %s created!\n", directoryPath);
    } else if (errno == EEXIST){
        printf("Directory %s already exists!\n", directoryPath);
    }

    // opening file  
    printf("Opening file %s\n", fullPath);
    if((fp=fopen(fullPath, "w"))==NULL){
        printf("Cannot open file.\n");
    }

    // writing to file
    printf("Saving %s to folder %s\n", fileName, directoryPath);
    if(fwrite(myArray, sizeof(float), arraySize, fp) != arraySize){
        printf("File write error.");
    }

    // closing file 
    fclose(fp);
}

