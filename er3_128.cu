#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"
void StartKernelTiming(cudaEvent_t& tic, cudaEvent_t& toc, cudaStream_t iStream);
void StopKernelTiming(cudaEvent_t& tic, cudaEvent_t& toc, cudaStream_t iStream, float* ptimer);
__global__ void vecMat1(double *_dst, double* _mat, double* _v, int _w, int _h);
__device__ double atomicAdd(double* address, double val);

#define BLOCK_HEIGHT 128
#define BLOCK_WIDTH 512



int main(int argc , char *argv[])
{
    	if(argc != 3)
    	{
        	printf("\n Usage: %s <HEIGHT> <WIDTH> \n",argv[0]);
        	return 1;
    	} 
	
	double block_height=128;
	double block_width=512;
	int h=atoi(argv[2]);
	int w=atoi(argv[1]);
	int n;
	const unsigned int THREADS_PER_BLOCK = 512;

	
	int r=ceil(h/block_height);
	int c=ceil(w/block_width);

	
	double *hostMat = (double*) calloc(h*w, sizeof(double));
	double *hostVec = (double*) calloc(h, sizeof(double));
	double *hostResVec = (double*) calloc(w, sizeof(double));
	bzero(hostResVec, w*sizeof(double));
	
    
	for(n=0;n<h*w;++n)
	{
      		hostMat[n] = drand48();
      	}
    
	for(n=0;n<h;++n) 
	   {
	  	hostVec[n] = drand48();
	   }

	   
	// allocate memory
	double *gpuMat;
	double *gpuVec;
	double *gpuResVec;

	cudaMalloc( (void**)&gpuVec, h * sizeof(double) );


	cudaMalloc( (void**)&gpuResVec, w * sizeof(double) );
	 
	cudaMalloc( (void**)&gpuMat, w*h* sizeof(double)  );

 

	// upload M and x
	cudaMemcpy( gpuMat, (void*) hostMat, w*h * sizeof(double),cudaMemcpyHostToDevice);

	cudaMemcpy( gpuVec, (void*) hostVec, h * sizeof(double),cudaMemcpyHostToDevice );

	cudaMemcpy( gpuResVec, (void*) hostResVec, w * sizeof(double),cudaMemcpyHostToDevice );

	
	// compute the block and grid dimensions
	dim3 threadBlock( THREADS_PER_BLOCK, 1 ); // 1 dimension block
	
	dim3 gridDim( c, r, 1);
	
	//xronometrhsh
	cudaEvent_t tic, toc;
    	float elapsed_time = 0;
    	StartKernelTiming(tic, toc, 0);

	vecMat1<<<  gridDim, threadBlock >>>( gpuResVec, gpuMat, gpuVec, w, h);

	// download result y
	cudaMemcpy( hostResVec, gpuResVec, w * sizeof(double), cudaMemcpyDeviceToHost) ;

	StopKernelTiming(tic,toc, 0, &elapsed_time); /*telos xronometrhshs*/
    
	/*int k=0;
	for(k=0; k<w; k++)
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
	
}  ///endOfMain

__global__ void vecMat1(double *_dst, double* _mat, double* _v, int _w, int _h) 
{
	__shared__ int blockx;
	__shared__ int blocky;
	__shared__ int blockheight; //height tou block
	__shared__ double xs[BLOCK_WIDTH]; //tile of x in shared memory


	blocky=blockIdx.y*BLOCK_HEIGHT; //index y of matrix block;briskw se poio block-grammh eimai (arxh) 
	blockx=blockIdx.x*BLOCK_WIDTH; //index x of matrix block;     >>         block-sthlh  >>

	if (threadIdx.x==0)
	{

		if ( (blockIdx.y+1)*BLOCK_HEIGHT <= _h )//elegxw an to telos tou block <= mhkos sthlhs
			blockheight=BLOCK_HEIGHT;
		else
			blockheight=_h- blocky;
	}

	syncthreads();

	//load in shared memory-one element per thread, BLOCK_WIDTH elements in total
	if (threadIdx.x < blockheight)
		xs[threadIdx.x]=_v[blocky + threadIdx.x]; 

	syncthreads();
	
	double res = 0;
	int i = blockx + threadIdx.x; //arithmos sthlhs mhtrwou
	
	if ( i< _w)
	{
		
		for (int j=0; j<blockheight; j++)
			res+=_mat[(blocky+j)*(_w)+i]*xs[j];
						
		double z=atomicAdd(_dst+i, res);
	}

} //endOfvecMat1

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

void StartKernelTiming(cudaEvent_t& tic, cudaEvent_t& toc, cudaStream_t iStream)
{
  	cudaEventCreate(&tic); 
  	cudaEventCreate(&toc);
  	cudaEventRecord(tic, iStream);
}

void StopKernelTiming(cudaEvent_t& tic, cudaEvent_t& toc, cudaStream_t iStream, float* ptimer)
//---------------------------------------------------------
{
  	float kt;
  	cudaEventRecord(toc, iStream);
  	cudaEventSynchronize(toc);
  	cudaEventElapsedTime(&kt, tic, toc);
  	cudaEventDestroy(tic); cudaEventDestroy(toc);
  	(*ptimer) += kt;
}
