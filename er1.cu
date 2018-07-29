#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda.h"
#include "cublas.h"

void StartKernelTiming(cudaEvent_t& tic, cudaEvent_t& toc, cudaStream_t iStream);
void StopKernelTiming(cudaEvent_t& tic, cudaEvent_t& toc, cudaStream_t iStream, float* ptimer);

#define TX 16

int main(int argc , char *argv[])
{
    	if(argc != 3)
    	{
        	printf("\n Usage: %s <HEIGHT> <WIDTH> \n",argv[0]);
        	return 1;
    	} 
	
	int M=atoi(argv[1]);
	int N=atoi(argv[2]);

    	int n;
    
    	/* fill host array*/
    	double *h_A = (double*) calloc(M*N, sizeof(double));
    	double *h_x = (double*) calloc(N, sizeof(double));
    	double *h_y = (double*) calloc(M, sizeof(double));
	    
	for(n=0;n<M*N;++n) {
      		h_A[n] = drand48();
      	}
    
	for(n=0;n<N;++n) {
		h_x[n] = drand48();
	}
		
    	/* now do cublas version */
    	double *cublas_A, *cublas_x, *cublas_y;
    	cublasInit();
    
    	cublasAlloc(M*N, sizeof(double), (void**) &cublas_A);
    	cublasAlloc(N, sizeof(double), (void**) &cublas_x);
    	cublasAlloc(M, sizeof(double), (void**) &cublas_y);
    
    	cublasSetMatrix(M, N, sizeof(double), h_A, M, cublas_A, M);
    	cublasSetVector(N, sizeof(double), h_x, 1, cublas_x, 1);
    
	
    	double alpha = 1.f, beta = 0.f;

    	cudaEvent_t tic, toc;
    	float elapsed_time = 0.f;
    	StartKernelTiming(tic, toc, 0);
    	
    	cublasDgemv ('T', N, M, 
		   alpha, cublas_A, N,
		   cublas_x, 1,
		   beta, cublas_y, 1);

	
    	cublasGetVector(M, sizeof(double), cublas_y, 1, h_y, 1); 
    
    	StopKernelTiming(tic,toc, 0, &elapsed_time);
	
	/*block kwdika pou tupwnei ta apotelesmata tou pollaplasiasmou (tou vector)*/
	/*int k=0;
	for(k=0; k<M; k++)
	{
		printf("gpu: %f\n", h_y[k]);
		
	}*/
	
    
    	/* convert from miliseconds to seconds */
    	elapsed_time /= 1000.0;
    
    	/* output elapsed time */
   	printf("elapsed time:%g sec \n", elapsed_time);

    	free(h_A);	
    	free(h_x);	
    	free(h_y);
	
    	cublasFree(cublas_A);
    	cublasFree(cublas_x);
    	cublasFree(cublas_y);

  	cublasShutdown();
}

void StartKernelTiming(cudaEvent_t& tic, cudaEvent_t& toc, cudaStream_t iStream)
{
  cudaEventCreate(&tic); 
  cudaEventCreate(&toc);
  cudaEventRecord(tic, iStream);
}

void StopKernelTiming(cudaEvent_t& tic, cudaEvent_t& toc, cudaStream_t iStream, float* ptimer)
{
  float kt;
  cudaEventRecord(toc, iStream);
  cudaEventSynchronize(toc);
  cudaEventElapsedTime(&kt, tic, toc);
  cudaEventDestroy(tic); cudaEventDestroy(toc);
  (*ptimer) += kt;
}
