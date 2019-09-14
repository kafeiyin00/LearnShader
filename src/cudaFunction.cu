#include <cstdio>
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/include/cuda_runtime.h"
#include "cudaFunction.cuh"

__global__ void gpu_normalEstimation(float* d_depth, int width, int height,
        float* d_nx,float* d_ny,float* d_nz) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //printf("%d", pointSize);
    if (idx >= width*height) {
        return;
    }
    int row = idx/width;
    int col = idx%width;
    float tempDepth = d_depth[idx];

    float phi, theta;
    phi = 1.0*row/height*3.1415926;
    theta = 1.0*col/width*2*3.1415926;

    float x = tempDepth*sin(phi)*cos(theta);
    float y = tempDepth*sin(phi)*sin(theta);
    float z = tempDepth*cos(phi);


    float leftDepth, topDepth;//p1,p2
    if(col == 0){
        return;
    } else{
        leftDepth = d_depth[row*width+col-1];
    }
    if(row == 0){
        return;
    } else{
        topDepth = d_depth[(row-1)*width+col];
    }

    if(leftDepth > 999 || topDepth>999 || tempDepth>999){
        return;
    }

    float leftphi, lefttheta;
    leftphi = 1.0*row/height*3.1415926;
    lefttheta = 1.0*(col-1)/width*2*3.1415926;
    float x1 = leftDepth*sin(leftphi)*cos(lefttheta);
    float y1 = leftDepth*sin(leftphi)*sin(lefttheta);
    float z1 = leftDepth*cos(leftphi);


    float topphi, toptheta;
    topphi = 1.0*(row-1)/height*3.1415926;
    toptheta = 1.0*col/width*2*3.1415926;
    float x2 = topDepth*sin(topphi)*cos(toptheta);
    float y2 = topDepth*sin(topphi)*sin(toptheta);
    float z2 = topDepth*cos(topphi);


	float n1x = x1 - x;
	float n1y = y1 - y;
	float n1z = z1 - z;

	float n2x = x2 - x;
	float n2y = y2 - y;
	float n2z = z2 - z;

	d_nx[idx] = n1y * n2z - n1z * n2y;
	d_ny[idx] = n1x * n2z - n1z * n2x;
	d_nz[idx] = n1x * n2y - n1y * n2x;

	//printf("%f\n",d_ny[idx]);

}

void calculateNormal(float* depth, int width, int height, float* nx,float* ny,float* nz){

    float *d_depth, *d_nx, *d_ny, *d_nz;
    //allocate GPU memory
    cudaMalloc((void**)&d_depth, width*height * sizeof(float));
    cudaMalloc((void**)&d_nx, width*height * sizeof(float));
    cudaMalloc((void**)&d_ny, width*height * sizeof(float));
    cudaMalloc((void**)&d_nz, width*height * sizeof(float));

    //from cpu to gpu
    cudaMemcpy(d_depth, depth, width*height * sizeof(float), cudaMemcpyHostToDevice);

    gpu_normalEstimation <<< width*height / 256 + 1, 256 >>> (d_depth, width, height,
            d_nx,d_ny,d_nz);

    //from gpu to cpu
    cudaMemcpy(nx, d_nx, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ny, d_ny, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(nz, d_nz, width * height * sizeof(float), cudaMemcpyDeviceToHost);
}