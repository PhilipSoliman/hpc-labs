%%cuda

// the subroutine for GPU code can be found in several separated text file from the Brightspace.
// You can add these subroutines to this main code.
////////////////////////////////////////////


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cuda.h"

#define PRINT_PROGRESS 0

const int BLOCK_SIZE = 100;  // number of threads per block

// Input Array Variables
float* h_MatA = NULL;
float* d_MatA = NULL;

// Output Array
float* h_VecV = NULL;
float* d_VecV = NULL;
float* h_VecW = NULL;
float* d_VecW = NULL;
float* h_NormW = NULL;
float* d_NormW = NULL;

// Variables to change
int N;
int GlobalSize = 5000;         // this is the dimension of the matrix, GlobalSize*GlobalSize
int BlockSize = BLOCK_SIZE;            // number of threads in each block
const float EPS = 0.000005;    // tolerence of the error
int max_iteration = 100;       // the maximum iteration steps
int timing_runs = 5;

// Functions
void Cleanup(void);
void InitOne(float*, int);
void UploadArray(float*, int);
float CPUReduce(float*, int);
void  Arguments(int, char**);
void checkCardVersion(void);
void ParseArguments(int, char**);

// Kernels
__global__ void Av_Product(float* g_MatA, float* g_VecV, float* g_VecW, int N);
__global__ void FindNormW(float* g_VecW, float * g_NormW, int N);
__global__ void NormalizeW(float *g_VecW, float *d_NormW, float *g_VecV, int N);
__global__ void ComputeLamda( float* g_VecV,float* g_VecW, float * g_Lamda,int N);

__global__ void Av_Product_Global(float* g_MatA, float* g_VecV, float* g_VecW, int N);
__global__ void FindNormW_Global(float* g_VecW, float * g_NormW, int N);
__global__ void NormalizeW_Global(float *g_VecW, float *d_NormW, float *g_VecV, int N);
__global__ void ComputeLamda_Global( float* g_VecV,float* g_VecW, float * g_Lamda,int N);

void CPU_AvProduct()
{
	//int N = GlobalSize;
	int matIndex =0;
    for(int i=0;i<N;i++)
	{
		h_VecW[i] = 0;
		for(int j=0;j<N;j++)
		{
			matIndex = i*N + j;
			h_VecW[i] += h_MatA[matIndex] * h_VecV[j];

		}
	}
}

void CPU_NormalizeW()
{
	//int N = GlobalSize;
	float normW=0;
	for(int i=0;i<N;i++)
		normW += h_VecW[i] * h_VecW[i];

	normW = sqrt(normW);
	for(int i=0;i<N;i++)
		h_VecV[i] = h_VecW[i]/normW;
}

float CPU_ComputeLamda()
{
	//int N = GlobalSize;
	float lamda =0;
	for(int i=0;i<N;i++)
		lamda += h_VecV[i] * h_VecW[i];

	return lamda;
}

void RunCPUPowerMethod()
{
	float oldLamda =0;
	float lamda=0;

	//AvProduct
	CPU_AvProduct();
  int i;
	//power loop
	for (i=0;i<max_iteration;i++)
	{
		CPU_NormalizeW();
		CPU_AvProduct();
		lamda= CPU_ComputeLamda();
    if (i % 10 && PRINT_PROGRESS == 1)
		  printf("CPU lamda at %d: %f \n", i, lamda);
		// If residual is lass than epsilon break
		if(abs(oldLamda - lamda) < EPS)
			break;
		oldLamda = lamda;
	}
  printf("CPU lamda at %d: %f \n", i, lamda);
}

void RunGPUPowerMethod(char* memtype, double *time_array, int blocksPerGrid, int threadsPerBlock, int sharedMemSize = NULL)
{
  float oldLamda = 0.0;
  float newLamda = 0.0;
  int k = 0;
  //int N = GlobalSize;
  size_t norm_size = sizeof(float);
  size_t vec_size = N * sizeof(float);
  struct timespec t_start, t_end;
  struct timespec t_mem_start, t_mem_end;
  double runtime;
  double memtranstime = 0.0;

  // Initialize input matrix
  InitOne(h_VecV, N);
  cudaMemcpy(d_VecV, h_VecV, vec_size, cudaMemcpyHostToDevice);

  clock_gettime(CLOCK_REALTIME,&t_start);
  if (strcmp(memtype, "shared") == 0)
  {
    if (sharedMemSize == NULL)
    {
        printf("No shared memory size specified.");
        exit(1);
    }
   	for (k = 0; k < max_iteration; k++)
    {

      Av_Product<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_MatA, d_VecV, d_VecW, N);
      cudaDeviceSynchronize();

      FindNormW<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_VecW, d_NormW, N);
      cudaDeviceSynchronize();

      // copy squared norm to CPU and take square root, then send it back to device (time)
      clock_gettime(CLOCK_REALTIME, &t_mem_start);
      cudaMemcpy(h_NormW, d_NormW, norm_size, cudaMemcpyDeviceToHost);
      h_NormW[0] = sqrt(h_NormW[0]);
      cudaMemcpy(d_NormW, h_NormW, norm_size, cudaMemcpyHostToDevice);
      clock_gettime(CLOCK_REALTIME,&t_mem_end);
      memtranstime += (t_mem_end.tv_sec - t_mem_start.tv_sec) + 1e-9*(t_mem_end.tv_nsec - t_mem_start.tv_nsec);

      NormalizeW<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_VecW, d_NormW, d_VecV, N);
      cudaDeviceSynchronize();

      cudaMemset(d_NormW, 0, norm_size);
      ComputeLamda<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_VecV, d_VecW, d_NormW, N);
      cudaDeviceSynchronize();

      // copy lamda from device to new lamda in host
      clock_gettime(CLOCK_REALTIME, &t_mem_start);
      cudaMemcpy(h_NormW, d_NormW, norm_size, cudaMemcpyDeviceToHost);
      clock_gettime(CLOCK_REALTIME,&t_mem_end);
      memtranstime += (t_mem_end.tv_sec - t_mem_start.tv_sec) + 1e-9*(t_mem_end.tv_nsec - t_mem_start.tv_nsec);

      newLamda = h_NormW[0];

      // print result
      if (k % 10 == 0 && PRINT_PROGRESS == 1)
        printf("GPU (shared) lamda at %d: %f \n", k, newLamda);

      if (abs(oldLamda - newLamda) < 2*EPS)
        break;

      oldLamda = newLamda;

      k++;
    }
  }
  else if (strcmp(memtype, "global") == 0)
  {
    for (k = 0; k < max_iteration; k++)
    {
      Av_Product_Global<<<blocksPerGrid, threadsPerBlock>>>(d_MatA, d_VecV, d_VecW, N);
      cudaDeviceSynchronize();

      FindNormW_Global<<<blocksPerGrid, threadsPerBlock>>>(d_VecW, d_NormW, N);
      cudaDeviceSynchronize();

      // copy squared norm to CPU and take square root, then send it back to device
      clock_gettime(CLOCK_REALTIME, &t_mem_start);
      cudaMemcpy(h_NormW, d_NormW, norm_size, cudaMemcpyDeviceToHost);
      h_NormW[0] = sqrt(h_NormW[0]);
      cudaMemcpy(d_NormW, h_NormW, norm_size, cudaMemcpyHostToDevice);
      clock_gettime(CLOCK_REALTIME,&t_mem_end);
      memtranstime += (t_mem_end.tv_sec - t_mem_start.tv_sec) + 1e-9*(t_mem_end.tv_nsec - t_mem_start.tv_nsec);

      NormalizeW_Global<<<blocksPerGrid, threadsPerBlock>>>(d_VecW, d_NormW, d_VecV, N);
      cudaDeviceSynchronize();

      cudaMemset(d_NormW, 0, norm_size);
      ComputeLamda_Global<<<blocksPerGrid, threadsPerBlock>>>(d_VecV, d_VecW, d_NormW, N);
      cudaDeviceSynchronize();

      // copy lamda from device to new lamda in host
      clock_gettime(CLOCK_REALTIME, &t_mem_start);
      cudaMemcpy(h_NormW, d_NormW, norm_size, cudaMemcpyDeviceToHost);
      clock_gettime(CLOCK_REALTIME,&t_mem_end);
      memtranstime += (t_mem_end.tv_sec - t_mem_start.tv_sec) + 1e-9*(t_mem_end.tv_nsec - t_mem_start.tv_nsec);

      newLamda = h_NormW[0];

      // print result
      if (k % 10 == 0 && PRINT_PROGRESS == 1)
        printf("GPU (global) lamda at %d: %f \n", k, newLamda);

      if (abs(oldLamda - newLamda) < 2*EPS)
        break;

      oldLamda = newLamda;

      k++;
    }
  }

  printf("GPU (%s) lamda at %d: %f \n", memtype, k, newLamda);
  clock_gettime(CLOCK_REALTIME,&t_end);
  runtime = (t_end.tv_sec - t_start.tv_sec) + 1e-9*(t_end.tv_nsec - t_start.tv_nsec);

  // reset vectors to zero for possible next run
  cudaMemset(d_VecV, 0, vec_size);
  cudaMemset(d_VecW, 0, vec_size);
  cudaMemset(d_NormW, 0, norm_size);

  // return runtime and memtranstime
  time_array[0] = runtime;
  time_array[1] = memtranstime;
}

void ParseArguments(int argc, char** argv) {
    // If you want to parse some arguments
}

// Host code
int main(int argc, char** argv)
{

    struct timespec t_start,t_end;
    double cpu_time;
    double cpu_averaged;
    double* gpu_global_time;
    double* gpu_shared_time;
    gpu_global_time = (double *)malloc(2 * sizeof(double));
    gpu_shared_time = (double *)malloc(2 * sizeof(double));
    double* gpu_global_averaged;
    double* gpu_shared_averaged;
    gpu_global_averaged = (double *)malloc(2 * sizeof(double));
    gpu_shared_averaged = (double *)malloc(2 * sizeof(double));
    ParseArguments(argc, argv);
    char* memtype = (char *)malloc(10*sizeof(char));
    size_t vec_size;
    size_t mat_size;
    size_t norm_size;
    int Ns[1] = {50};
    for (int idx = 0; idx < sizeof(Ns)/sizeof(int); idx++)
    {
      N = Ns[idx];
      printf("\nMatrix size run:  %d X %d \n", N, N);

      cpu_averaged = 0.0;
      for (int i = 0; i < 2; i++)
      {
          gpu_global_averaged[i] = 0.0;
          gpu_shared_averaged[i] = 0.0;
      }

      size_t vec_size = N * sizeof(float);
      size_t mat_size = N * N * sizeof(float);
      size_t norm_size = sizeof(float);

      // Allocate normalized value in host memory
      h_NormW = (float*)malloc(norm_size);
      // Allocate input matrix in host memory
      h_MatA = (float*)malloc(mat_size);
      // Allocate initial vector V in host memory
      h_VecV = (float*)malloc(vec_size);
      // Allocate W vector for computations
      h_VecW = (float*)malloc(vec_size);


      // Initialize input matrix
      UploadArray(h_MatA, N);

      printf("Power method in CPU starts\n");
      printf("*************************************\n");
      for (int i = 0; i < timing_runs; i++)
      {
        printf("Timing run: %i\n", i+1);
        InitOne(h_VecV, N);
        clock_gettime(CLOCK_REALTIME,&t_start);
        RunCPUPowerMethod();
        clock_gettime(CLOCK_REALTIME,&t_end);
        cpu_time = (t_end.tv_sec - t_start.tv_sec) + 1e-9*(t_end.tv_nsec - t_start.tv_nsec);
        cpu_averaged += cpu_time/timing_runs;
      }
      printf("*************************************\n");
      printf("Power method in CPU is finished\n");

      // This is the starting points of GPU
      printf("\nPower method in GPU starts\n");
      checkCardVersion();

      clock_gettime(CLOCK_REALTIME,&t_start);  // Here I start to count

      // Set the kernel arguments
      int threadsPerBlock = BlockSize;
      int sharedMemSize = threadsPerBlock * threadsPerBlock * sizeof(float); // in per block, the memory is shared
      int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

      // Allocate matrix and vectors in device memory
      cudaMalloc((void**)&d_MatA, mat_size);
      cudaMalloc((void**)&d_VecV, vec_size);
      cudaMalloc((void**)&d_VecW, vec_size); // This vector is only used by the device
      cudaMalloc((void**)&d_NormW, norm_size);

      //Copy from host memory to device memory
      cudaMemcpy(d_MatA, h_MatA, mat_size, cudaMemcpyHostToDevice);
      //cudaMemcpy(d_VecV, h_VecV, vec_size, cudaMemcpyHostToDevice);

      //cutilCheckError(cutStopTimer(timer_mem));

      //Power method loops
      printf("*************************************\n");
      for (int i = 0; i < timing_runs; i++)
      {
        printf("Timing run : %i\n", i + 1);
        memtype = "shared";
        RunGPUPowerMethod(memtype, gpu_shared_time, blocksPerGrid, threadsPerBlock, sharedMemSize = sharedMemSize);
        memtype = "global";
        RunGPUPowerMethod(memtype, gpu_global_time, blocksPerGrid, threadsPerBlock);

        for (int k = 0; k < 2; k++)
        {
          gpu_global_averaged[k] += gpu_global_time[k]/timing_runs;
          gpu_shared_averaged[k] += gpu_shared_time[k]/timing_runs;
        }
      }
      printf("*************************************\n");
      printf("GPU power method is finished\n");
      printf("\nSUMMARY\n");
      printf("array size: %i\n", N);
      printf("threads: %i\n", threadsPerBlock);
      printf("CPU                     : %6.2e secs\n", cpu_averaged);
      printf("GPU (global)            : %6.2e secs (mem. transfer = %6.2e secs)\n", gpu_global_averaged[0], gpu_global_averaged[1]);
      printf("GPU (shared)            : %6.2e secs (mem. transfer = %6.2e secs)\n", gpu_shared_averaged[0], gpu_shared_averaged[1]);
      printf("Allocated shared memory : %3.2e KB\n", sharedMemSize * 1e-3);

      printf("\nSaving data to file");
      char fn_tmp[75] = "powerMethodGPU_n=%i_nthreads=%i.txt";
      char fn[100];
      sprintf(fn, fn_tmp, N, threadsPerBlock);
      FILE *f = fopen(fn, "w");
      if (f == NULL)
      {
        printf("Failed to open file");
        exit(1);
      }
      fprintf(f, "CPU: %f\n", cpu_averaged);
      fprintf(f, "GPU (global): %f, %f\n", gpu_global_averaged[0], gpu_global_averaged[1]);
      fprintf(f, "GPU (shared): %f, %f\n", gpu_shared_averaged[0], gpu_shared_averaged[1]);
      fprintf(f, "Allocated shared memory (KB): %f\n", sharedMemSize * 1e-3);
      fclose(f);

      Cleanup();
    }
}

void Cleanup(void)
{
    // Free device memory
    if (d_MatA)
        cudaFree(d_MatA);
    if (d_VecV)
        cudaFree(d_VecV);
    if (d_VecW)
        cudaFree(d_VecW);
	  if (d_NormW)
		    cudaFree(d_NormW);

    // Free host memory
    if (h_MatA)
        free(h_MatA);
    if (h_VecV)
        free(h_VecV);
    if (h_VecW)
        free(h_VecW);
    if (h_NormW)
        free(h_NormW);

    exit(0);
}

// Allocates an array with zero value.
void InitOne(float* data, int n)
{
    for (int i = 0; i < n; i++)
        data[i] = 0;
	data[0]=1;
}

void UploadArray(float* data, int n)
{
   int total = n*n;
   int value=1;
    for (int i = 0; i < total; i++)
    {
    	data[i] = (int) (rand() % (int)(101));//1;//value;
	    value ++; if(value>n) value =1;
      // data[i] = 1;
    }
}

// Obtain program arguments
void Arguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i)
    {
        if (strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-size") == 0)
        {
            N = atoi(argv[i+1]);
		    i = i + 1;
        }
        if (strcmp(argv[i], "--max_iteration") == 0 || strcmp(argv[i], "-max_iteration") == 0)
        {
            max_iteration = atoi(argv[i+1]);
		    i = i + 1;
        }
    }
}


void checkCardVersion()
{
   cudaDeviceProp prop;

   cudaGetDeviceProperties(&prop, 0);

   printf("This GPU has major architecture %d, minor %d \n",prop.major,prop.minor);
   if(prop.major < 2)
   {
      fprintf(stderr,"Need compute capability 2 or higher.\n");
      exit(1);
   }
}

//   _    _ _______ _____ _      _____ _________     __     __  __ ______ _______ _    _  ____  _____   _____
//  | |  | |__   __|_   _| |    |_   _|__   __\ \   / /    |  \/  |  ____|__   __| |  | |/ __ \|  __ \ / ____|
//  | |  | |  | |    | | | |      | |    | |   \ \_/ /     | \  / | |__     | |  | |__| | |  | | |  | | (___
//  | |  | |  | |    | | | |      | |    | |    \   /      | |\/| |  __|    | |  |  __  | |  | | |  | |\___ \
//  | |__| |  | |   _| |_| |____ _| |_   | |     | |       | |  | | |____   | |  | |  | | |__| | |__| |____) |
//   \____/   |_|  |_____|______|_____|  |_|     |_|       |_|  |_|______|  |_|  |_|  |_|\____/|_____/|_____/


// This is a list of utility methods that should you use in your code

/*****************************************************************************
This function finds the product of Matrix A and vector V
*****************************************************************************/

// ****************************************************************************************************************************************************/
// parallelization method for the Matrix-vector multiplication as follows:

// each thread handle a multiplication of each row of Matrix A and vector V;

// The share memory is limited for a block, instead of reading an entire row of matrix A or vector V from global memory to share memory,
// a square submatrix of A is shared by a block, the size of square submatrix is BLOCK_SIZE*BLOCK_SIZE; Thus, a for-loop is used to
// handle a multiplication of each row of Matrix A and vector V step by step. In eacg step, two subvectors with size BLOCK_SIZE is multiplied.
//*****************************************************************************************************************************************************/


__global__ void Av_Product(float* g_MatA, float* g_VecV, float* g_VecW, int N)
{
    // Block index
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;

    int aBegin = N * BLOCK_SIZE * bx;

    int aEnd   = aBegin + N - 1;
    int step  = BLOCK_SIZE;

    int bBegin = 0;//BLOCK_SIZE * bx;
    int bIndex=0;
    int aIndex =0;
    float Csub = 0;

    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += step, b += step)
    {

        __shared__ float As[BLOCK_SIZE*BLOCK_SIZE];

        __shared__ float bs[BLOCK_SIZE];


        for (int aa = 0; aa < BLOCK_SIZE;aa+= 1)
        {
            aIndex = a + tx + aa*N;
            if( aIndex < N*N)
        	    As[tx+aa*BLOCK_SIZE] = g_MatA[aIndex];
		        else
        	    As[tx+aa*BLOCK_SIZE] = 0;
        }

        bIndex = b+tx;
   	    if(bIndex<N)
		      bs[tx] = g_VecV[bIndex];
	      else
		      bs[tx] = 0;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[k+tx*BLOCK_SIZE] * bs[k];
        }
        __syncthreads();
    }

    g_VecW[ BLOCK_SIZE * bx + tx] = Csub;
}

/****************************************************
Normalizes vector W : W/norm(W)
****************************************************/
__global__ void FindNormW(float* g_VecW, float * g_NormW, int N)
{
  // shared memory size declared at kernel launch
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;

  // For thread ids greater than data space
  if (globalid < N) {
     sdata[tid] =  g_VecW[globalid];
  }
  else {
     sdata[tid] = 0;  // Case of extra threads above N
  }

  // each thread loads one element from global to shared mem
  __syncthreads();

  sdata[tid] = sdata[tid] * sdata[tid];
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x / 2; s > 0; s = s >> 1) {
     if (tid < s) {
         sdata[tid] = sdata[tid] + sdata[tid+ s];
     }
     __syncthreads();
  }
   // atomic operations:
  if (tid == 0) atomicAdd(g_NormW,sdata[0]);
}

__global__ void NormalizeW(float* g_VecW, float* g_NormW, float* g_VecV, int N)
{
  // shared memory size declared at kernel launch
  extern __shared__ float sNormData[];
  unsigned int tid = threadIdx.x;
  unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;

  if(tid==0) sNormData[0] =  g_NormW[0];
  __syncthreads();

  // For thread ids greater than data space
  if (globalid < N) {
     g_VecV[globalid] = g_VecW[globalid]/sNormData[0];
  }

}

__global__ void ComputeLamda( float* g_VecV, float* g_VecW, float * g_Lamda,int N)
{
  // shared memory size declared at kernel launch
  extern __shared__ float sdataVW[];
  unsigned int tid = threadIdx.x;
  unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;

  // For thread ids greater than data space
  if (globalid < N) {
     sdataVW[tid] =  g_VecV[globalid] * g_VecW[globalid];
  }
  else {
     sdataVW[tid] = 0;  // Case of extra threads above N
  }

  // each thread loads one element from global to shared mem
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x / 2; s > 0; s = s >> 1) {
     if (tid < s) {
         sdataVW[tid] = sdataVW[tid] + sdataVW[tid+ s];
     }
     __syncthreads();
  }
   // atomic operations:
  if (tid == 0) atomicAdd(g_Lamda,sdataVW[0]);
}


/********************************************************************************

                  GLOBAL MEMORY VERSION OF GPU FUNCTIONS

********************************************************************************/

__global__ void Av_Product_Global(float* g_MatA, float* g_VecV, float* g_VecW, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N)
  {
    float sum = 0.0;
      for (int i = 0; i < N; i++)
      {
          sum += g_MatA[idx * N + i] * g_VecV[i];
      }
    g_VecW[idx] = sum;
  }
  __syncthreads();
}

__global__ void FindNormW_Global(float* g_VecW, float * g_NormW, int N)
{
  // shared memory size declared at kernel launch
  unsigned int globalid = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0;

  while (globalid < N)
  {
      sum += g_VecW[globalid] * g_VecW[globalid];
      globalid += gridDim.x * blockDim.x;
  }

  atomicAdd(g_NormW,sum);
  __syncthreads();
}

__global__ void NormalizeW_Global(float* g_VecW, float* g_NormW, float* g_VecV, int N)
{
  unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;

  if (globalid < N) {
     g_VecV[globalid] = g_VecW[globalid]/g_NormW[0];
  }

  __syncthreads();
}

__global__ void ComputeLamda_Global(float* g_VecV, float* g_VecW, float * g_Lamda,int N)
{
  unsigned int globalid = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0;

  while (globalid < N)
  {
      sum += g_VecW[globalid] * g_VecV[globalid];
      globalid += gridDim.x * blockDim.x;
  }

  atomicAdd(g_Lamda,sum);
  __syncthreads();
}
