// hello world

#include "MRI.hcu"
#include "float_util.hcu"

__constant__ float3 b;

const float T1 = 1e-5; // spin lattice in seconds.
const float T2 = 1e-6; // spin spin in seconds.
const float GYROMAGNETIC_RATIO = 42.58e6; // hertz pr tesla
const float BOLTZMANN_CONSTANT = 1.3805e-23; // Joule / Kelvin
const float PLANCK_CONSTANT = 6.626e-34; // Joule * seconds



void MRI_test(cuFloatComplex* input) {
        
}

__global__ void MRI_step_kernel(float dt, float3* spin_packs, float* eq) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float3 m = spin_packs[idx];
    /* spin_packs[idx] += dt*(cross(GYROMAGNETIC_RATIO * m, b) - make_float3(m.x / T2, m.y / T2, 0.0)  - make_float3(0.0, 0.0, (m.z - eq[idx])/T1)); */
}


__host__ void MRI_step(float dt, float* spin_packs, float* eq, unsigned int w, unsigned int h, float b0) {
    const float3 _b = make_float3(0.0,0.0,b0);
    cudaMemcpyToSymbol(b, &_b, sizeof(float3));

	dim3 blockDim(512,1,1);
	dim3 gridDim(int(((double)(w*h))/(double)blockDim.x),1,1);
    MRI_step_kernel<<< gridDim, blockDim >>>(dt, (float3*)spin_packs, eq);
}



