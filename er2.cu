#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
void StartKernelTiming(cudaEvent_t& tic, cudaEvent_t& toc, cudaStream_t iStream);
void StopKernelTiming(cudaEvent_t& tic, cudaEvent_t& toc, cudaStream_t iStream, float* ptimer);
__global__ void vecMat1(double *_dst, double* _mat, double* _v, int _w, int _h );

int main(int argc , char *argv[])
{
    	if(argc != 3)
    	{
        	printf("\n Usage: %s <HEIGHT> <WIDTH> \n",argv[0]);
        	return 1;
    	} 
	
	
	int h=atoi(argv[1]);
	int w=atoi(argv[2]);
	int n;
	const unsigned int THREADS_PER_BLOCK = 512;
	double *hostMat = (double*) calloc(h*w, sizeof(double));
	double *hostVec = (double*) calloc(w, sizeof(double));
	double *hostResVec = (double*) calloc(w, sizeof(double));
	    
	for(n=0;n<h*w;++n){
      		hostMat[n]=rand() % 100;
      	}
    
	for(n=0;n<w;++n)
	{
	   	hostVec[n] = rand() % 100;
	}

	
	// allocate memory
	double *gpuMat, *gpuVec, *gpuResVec;
	cudaMalloc( (void**)&gpuMat, w*h* sizeof(double)  );
	cudaMalloc( (void**)&gpuVec, w * sizeof(double) );
	cudaMalloc( (void**)&gpuResVec, h * sizeof(double) );

	// upload M and x
	cudaMemcpy( gpuMat, (void*) hostMat, w*h * sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy( gpuVec, (void*) hostVec, w * sizeof(double),cudaMemcpyHostToDevice );


	// compute the block and grid dimensions
	dim3 threadBlock( THREADS_PER_BLOCK, 1 );
	const unsigned int numBlocks = (n - 1)/THREADS_PER_BLOCK + 1;
	dim3 blockGrid( numBlocks, 1, 1);

	//xronometrhsh
	cudaEvent_t tic, toc;
    	float elapsed_time = 0.f;
    	StartKernelTiming(tic, toc, 0);

	vecMat1<<< blockGrid, threadBlock >>>( gpuResVec, gpuMat, gpuVec,w,h);
	cudaThreadSynchronize() ;

	// download result y
	cudaMemcpy( hostResVec, gpuResVec, h * sizeof(double), cudaMemcpyDeviceToHost) ;

	StopKernelTiming(tic,toc, 0, &elapsed_time); /*telos xronometrhshs*/
    

   	/*int k=0;
	for(k=0; k<h; k++)
	{
		printf("gpu: %f\n", hostResVec[k]);
		
	/*}
	    
    	/* convert from miliseconds to seconds */
    	elapsed_time /= 1000.0;
    
    	/* output elapsed time */
   	printf("elapsed time:%g sec \n", elapsed_time);
	
	cudaFree( gpuMat );
	cudaFree( gpuVec );
	cudaFree( gpuResVec );

	free(hostMat);
	free(hostVec);
	free(hostResVec);

}

__global__ void vecMat1(double *_dst, double* _mat, double* _v, int _w, int _h ) 
{
	// row index the thread is operating on
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < _h) {
		float res = 0.;
		// dot product of one line
		for (int j = 0; j < _w; ++j) {
			res += _mat[i*_w + j] * _v[j];
		}
	// write result to global memory
	_dst[i] = res;
	}
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
