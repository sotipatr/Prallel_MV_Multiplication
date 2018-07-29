all: er1.cu er2.cu er3_64.cu er3_128.cu
	nvcc  -arch=sm_13 -o er1 er1.cu -lcublas
	nvcc  -arch=sm_13 -g -pg -o er2 er2.cu
	nvcc  -arch=sm_13 -g -pg -o er3_64 er3_64.cu
	nvcc  -arch=sm_13 -g -pg -o er3_128 er3_128.cu
