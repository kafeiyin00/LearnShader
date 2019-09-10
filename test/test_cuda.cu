#include <iostream>
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/include/cuda_runtime.h"
#include "cuda_runtime.h"


__global__ void func(float* d_out, float* d_in) {
    int idx = threadIdx.x;
    float f = d_in[idx];
    d_out[idx] = f*f;
}


void getCudaState(){
	cudaDeviceProp prop;
	int count;
	cudaGetDeviceCount(&count);
	for (int i = 0; i < count; i++)
	{

		cudaGetDeviceProperties(&prop, i);
		std::cout << "显卡名称：" << prop.name << std::endl;
		std::cout << "显存大小：" << prop.totalGlobalMem / 1024 / 1024 << " MB" << std::endl;
		std::cout << "一个block的共享内存大小：" << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
		std::cout << "block最大线程数：" << prop.maxThreadsPerBlock << std::endl;
	}
}
int main()
{
	getCudaState();


	const int arraySize = 200;
    const int byteSize = arraySize*sizeof(float);

    //cpu
    float h_in[arraySize];
    for (int i = 0; i < arraySize; ++i) {
        h_in[i]=i;
    }
    float h_out[arraySize];

    //GPU
    float * d_in;
    float * d_out;

    //allocate GPU memory
    cudaMalloc((void**)&d_in,byteSize);
    cudaMalloc((void**)&d_out,byteSize);

    //from cpu to gpu
    cudaMemcpy(d_in,h_in,byteSize,cudaMemcpyHostToDevice);

    func<<<1,arraySize>>>(d_out,d_in);

    //from gpu to cpu
    cudaMemcpy(h_out,d_out,byteSize,cudaMemcpyDeviceToHost);


    for (int i = 0; i < arraySize; ++i) {
        std::cout<<h_out[i]<<"\n";
    }

    cudaFree(d_in);
    cudaFree(d_out);
	return 0;
}