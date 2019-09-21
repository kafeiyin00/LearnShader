#include <cstdio>
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/include/cuda_runtime.h"
#include <cuda_runtime_api.h>

#include "cudaFunction.cuh"

#include <Eigen/Core>

#include <Eigen/Eigenvalues>

#include "svd3_cuda.h"
#include <iostream>

#define FRAME_WIDTH 1200 // 180 deg -90-90
#define FRAME_HEIGHT 300 // 40 deg 70-110
#define OFFSET_THETA 90.0
#define OFFSET_PHI 70.0
#define RANGE_THETA 180.0
#define RANGE_PHI 40.0

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

__global__ void gpu_planarityEstimation(float* d_depth, int width, int height,
        float* d_planarity){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //printf("%d", pointSize);
    if (idx >= width*height) {
        return;
    }
    int row = idx/width;
    int col = idx%width;

    if(col <=20 || col >= (width-20)){
        return;
    }
    if(d_depth[idx] < 1){
        return;
    }

    Eigen::Matrix<float, 3, 41> sampleMatrix;
    for (int i = -20; i <= 20; ++i) {
        int tmp_col = col+i;
        float tempDepth = d_depth[row*width+tmp_col];
        float phi, theta;
        phi = 1.0*row/FRAME_HEIGHT*RANGE_PHI+OFFSET_PHI / 180 * 3.1415926;
        theta = 1.0*col/FRAME_WIDTH*RANGE_THETA+OFFSET_THETA / 180 * 3.1415926;

        float x = tempDepth*sin(phi)*cos(theta);
        float y = tempDepth*sin(phi)*sin(theta);
        float z = tempDepth*cos(phi);
        sampleMatrix.col(i+20) = Eigen::Vector3f(x,y,z);
    }

    // estimate pca
    Eigen::Vector3f mean = Eigen::Vector3f(
            sampleMatrix.row(0).mean(),
            sampleMatrix.row(1).mean(),
            sampleMatrix.row(2).mean());

    if(mean.norm() < 0.001){
        return;
    }

    for (int i = 0; i <= 40; ++i){
        sampleMatrix.col(i) = sampleMatrix.col(i) - mean;
    }
    Eigen::Matrix3f cov_matrix = sampleMatrix*sampleMatrix.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig; // default constructor
    eig.computeDirect(cov_matrix); // works for 2x2 and 3x3 matrices, does not require loops
    Eigen::Vector3f D = eig.eigenvalues();
    //printf("%f %f %f\n",D(0),D(1),D(2));
//    Eigen::EigenSolver<Eigen::Matrix3f> es(cov_matrix);
//    Eigen::Vector3f D = es.pseudoEigenvalueMatrix().diagonal();
    float line_feature = (sqrt(abs(D(2)))-sqrt(abs(D(1))))/sqrt(abs(D(2)));
//
//    d_planarity[idx] = line_feature;
//    float u11,u12,u13,u21,u22,u23,u31,u32,u33;	// output U
//    float s11;
//    float s22;
//    float s33; // output S
//    float v11,v12,v13,v21,v22,v23,v31,v32,v33;
//    svd( cov_matrix(0,0), cov_matrix(0,1), cov_matrix(0,2),
//         cov_matrix(1,0), cov_matrix(1,1), cov_matrix(1,2),
//         cov_matrix(2,0), cov_matrix(2,1), cov_matrix(2,2),		// input A
//            u11, u12,u13, u21, u22, u23, u31,u32,u33,	// output U
//            s11,
//            s22,
//            s33,
//            v11, v12, v13, v21, v22, v23, v31, v32, v33);
//    float line_feature = (sqrt(s33)-sqrt(s22))/sqrt(s33);
    printf("%f\n",line_feature);
//    d_planarity[idx] = line_feature;
}

void getCudaState(){
    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount(&count);
    for (int i = 0; i < count; i++)
    {

        cudaGetDeviceProperties(&prop, i);
        printf("name: %s\n", prop.name);
        printf("totalGlobalMem: %llu MB\n", prop.totalGlobalMem / 1024 / 1024);
        printf("blockmaxThreadsPerBlock: %d \n", prop.maxThreadsPerBlock);

        //std::cout << "name：" << prop.name << std::endl;
        //std::cout << "totalGlobalMem：" << prop.totalGlobalMem / 1024 / 1024 << " MB" << std::endl;
        //std::cout << "block count：" << prop.multiProcessorCount<< std::endl;
        //std::cout << "totalGlobalMem：" << prop.totalGlobalMem / 1024 << " KB" << std::endl;
        //std::cout << "blockmaxThreadsPerBlock：" << prop.maxThreadsPerBlock << std::endl;
    }
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

void calculatePlanarity(float* depth, int width, int height, float* planarity)
{
    float *d_depth, *d_planarity, *d_ny, *d_nz;
    //allocate GPU memory
    cudaMalloc((void**)&d_depth, width*height * sizeof(float));
    cudaMalloc((void**)&d_planarity, width*height * sizeof(float));

    //from cpu to gpu
    cudaMemcpy(d_depth, depth, width*height * sizeof(float), cudaMemcpyHostToDevice);

    gpu_planarityEstimation <<< width*height / 256 + 1, 256 >>> (d_depth, width, height,
            d_planarity);

    //from gpu to cpu
    cudaMemcpy(planarity, d_planarity, width * height * sizeof(float), cudaMemcpyDeviceToHost);

}