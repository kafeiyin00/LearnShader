#include "projection.cuh"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/include/cuda_runtime.h"
__global__ void gpu_project(float* d_xs, float* d_ys, float* d_zs,
	int* prjectionId, float* d_ds, int pointSize) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//printf("%d", pointSize);
	if (idx >= pointSize) {
		return;
	}
	//printf("%d", pointSize);
	float x = d_xs[idx];
	float y = d_ys[idx];
	float z = d_zs[idx];

	//bearing vector to theta phi
	double theta = atan2(y, x) / 3.1415926 * 180.0 + 180.0;
	double phi = atan2(sqrt(x*x + y * y), z) / 3.1415926 * 180.0;
	float depth = sqrt(x * x + y * y + z * z);

	//theta phi to image coordinates
	int u = 2000 - (theta / 360.0) * 2000;
	int v = phi / 180.0 * 1000;

	prjectionId[idx] = v * 2000 + u;
	d_ds[idx] = depth;

}

void project(float *h_xs, float *h_ys, float *h_zs, float *h_ds, int *h_pojectionIds, int nPoints) {
	float *d_xs, *d_ys, *d_zs, *d_ds;
	int *d_pojectionIds;

	//allocate GPU memory
	cudaMalloc((void**)&d_xs, nPoints * sizeof(float));
	cudaMalloc((void**)&d_ys, nPoints * sizeof(float));
	cudaMalloc((void**)&d_zs, nPoints * sizeof(float));

	cudaMalloc((void**)&d_pojectionIds, nPoints * sizeof(int));
	cudaMalloc((void**)&d_ds, nPoints * sizeof(float));

	//from cpu to gpu
	cudaMemcpy(d_xs, h_xs, nPoints * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ys, h_ys, nPoints * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_zs, h_zs, nPoints * sizeof(float), cudaMemcpyHostToDevice);

	gpu_project <<< nPoints / 256 + 1, 256 >>> (d_xs, d_ys, d_zs, d_pojectionIds, d_ds, nPoints);

	//from gpu to cpu
	cudaMemcpy(h_pojectionIds, d_pojectionIds, nPoints * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_ds, d_ds, nPoints * sizeof(float), cudaMemcpyDeviceToHost);
}