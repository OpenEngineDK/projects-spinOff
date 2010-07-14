// hello world

#include "MRI.hcu"

__constant__ float  c_b0;
__constant__ float3 c_b1;
__global__ void MRI_step_kernel(float dt, float* spin_packs);


void MRI_test(cuFloatComplex* input) {
        
}

__host__ void MRI_step(float dt, float* spin_packs, unsigned int w, unsigned int h, float b0) {
    const float3 b1 = make_float3(0.0,0.0,0.0);
    cudaMemcpyToSymbol(c_b0, &b0, sizeof(float));
    cudaMemcpyToSymbol(c_b1, &b1, sizeof(float3));

	dim3 blockDim(512,1,1);
	dim3 gridDim(int(((double)(w*h))/(double)blockDim.x),1,1);
    MRI_step_kernel<<< gridDim, blockDim >>>(dt, spin_packs);
}



__global__ void MRI_step_kernel(float dt, float* spin_packs) {

}
