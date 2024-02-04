CC = mpicc

FP_LIBS = -lm
GD_LIBS = -lm

FP_OBJS = MPI_Fempois.o
GD_OBJS = GridDist.o

all: MPI_Fempois GridDist

clean:
	rm -f *.o 

MPI_Fempois: $(FP_OBJS)
	mpicc -o $@.x $(FP_OBJS) $(FP_LIBS)

GridDist: $(GD_OBJS)
	gcc -o $@.x $(GD_OBJS) $(GD_LIBS)

MPI_Fempois.o: MPI_Fempois.c
	mpicc -c MPI_Fempois.c

GridDist.o: GridDist.c
	gcc -c GridDist.c