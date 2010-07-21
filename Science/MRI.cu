// hello world

#include "MRI.hcu"
#include "float_util.hcu"
#include <stdio.h>

__constant__ float3 b;
__constant__ float flip;

const float T1 = 1e-5; // spin lattice in seconds.
const float T2 = 1e-6; // spin spin in seconds.
const float GYROMAGNETIC_RATIO = 42.58e6; // hertz pr tesla
const float BOLTZMANN_CONSTANT = 1.3805e-23; // Joule / Kelvin
const float PLANCK_CONSTANT = 6.626e-34; // Joule * seconds


struct mat3x3 {
    float3 r1;
    float3 r2;
    float3 r3;
};

// __host__ __device__ float3 getRow(mat3x3 m, unsigned int i) {
//    switch (i) {        
//    case 0: return r1;
//    case 1: return r2;
//    case 2: return r3;
//    }
//    return make_float3(0.0,0.0,0.0);
// }

__host__ __device__ float3 getCol1(mat3x3 m) {

    return make_float3(m.r1.x, m.r2.x, m.r3.x);
}

__host__ __device__ float3 getCol2(mat3x3 m) {

    return make_float3(m.r1.y, m.r2.y, m.r3.y);
}

__host__ __device__ float3 getCol3(mat3x3 m) {

    return make_float3(m.r1.z, m.r2.z, m.r3.z);
}


__host__ __device__ mat3x3 make_mat3x3(float3 r1, float3 r2, float3 r3) {
    mat3x3 m;
    m.r1 = r1;
    m.r2 = r2;
    m.r3 = r3;
    return m;
}

__host__ __device__ mat3x3 operator*(mat3x3 m1, mat3x3 m2) {
    return make_mat3x3(make_float3(dot(m1.r1, getCol1(m2)), dot(m1.r1, getCol2(m2)), dot(m1.r1, getCol3(m2))),
                       make_float3(dot(m1.r2, getCol1(m2)), dot(m1.r2, getCol2(m2)), dot(m1.r2, getCol3(m2))),
                       make_float3(dot(m1.r3, getCol1(m2)), dot(m1.r3, getCol2(m2)), dot(m1.r3, getCol3(m2)))
                       );
}

__host__ __device__ float3 operator*(mat3x3 m, float3 v) {
    return make_float3(dot(m.r1, v), 
                       dot(m.r2, v), 
                       dot(m.r3, v));
}

__host__ __device__ mat3x3 rotX(float angle) {
    return make_mat3x3(make_float3(1.0,        0.0, 0.0       ),
                       make_float3(0.0, cos(angle), sin(angle)),
                       make_float3(0.0,-sin(angle), cos(angle))
                       );
}

__host__ __device__ mat3x3 rotZ(float angle) {
    return make_mat3x3(make_float3( cos(angle), sin(angle), 0.0),
                       make_float3(-sin(angle), cos(angle), 0.0),
                       make_float3(        0.0,        0.0, 1.0));
}

__host__ __device__ mat3x3 relax(float dt, float t1, float t2) {
    return make_mat3x3(make_float3(1.0/exp(dt/t2),         0.0,            0.0),
                       make_float3(        0.0, 1.0/exp(dt/t2),            0.0),
                       make_float3(        0.0,         0.0, 1.0-(1.0/exp(dt/t1)))
                       );
}

__host__ __device__ mat3x3 rf(float phaseAngle, float flipAngle) {
    return rotZ(phaseAngle) * rotX(flipAngle) * rotZ(-phaseAngle);
}

void MRI_test(cuFloatComplex* input) {
        
}

__global__ void MRI_step_kernel(float dt, float3* lab_spins, float3* ref_spins, float* eq, unsigned int size, float thetime) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;


    float omega = GYROMAGNETIC_RATIO * b.z;
    float3 m = ref_spins[idx];

    m += dt*make_float3(-m.x/T2, -m.y/T2, (m.z - eq[idx])/T1);

    ref_spins[idx] = m;
    lab_spins[idx] = make_float3(m.x * cos(omega * thetime) - m.y * sin(omega*thetime), m.x * sin(omega * thetime) + m.y * cos(omega*thetime),  m.z);
    // lab_spins[idx] += dt * (cross(GYROMAGNETIC_RATIO * m, b) - make_float3(m.x / T2, m.y / T2, 0.0) - make_float3(0.0, 0.0, (eq[idx] - m.z) / T1));
    
    //lab_spins[idx] = (rotZ(GYROMAGNETIC_RATIO * dt) * relax(dt, T1, T2)) * m;
    // m = relax(dt,T1,T2) * m;
    //lab_spins[idx] = relax(dt,T1,T2) * m;
    //lab_spins[idx] = /*(rotZ(GYROMAGNETIC_RATIO * 0.0 * dt) * relax(dt, T1, T2) * rf(0.0, b.x))*/  rotX(b.x) * m;
    //lab_spins[idx] = /*rotZ(GYROMAGNETIC_RATIO * b.z * dt) **/ (relax(dt, T1, T2) * (rotX(b.x) * m));
}

__global__ void MRI_step_kernel_anal(float t, float3* spin_packs, float* eq, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;
        
    float3 m = spin_packs[idx];

    float larmorFrequency = GYROMAGNETIC_RATIO;

    float2 e1;
    e1.x = cos(larmorFrequency * t);
    e1.y = sin(larmorFrequency * t);
    float2 e2;
    e2.x = -sin(larmorFrequency * t);
    e2.y = cos(larmorFrequency * t);


    float T1exp = exp(-t/T1);
    float T2exp = exp(-t/T2);


    float3 localNetMagnetization;

    localNetMagnetization.x = T2exp * eq[0];
    localNetMagnetization.y = T2exp * eq[1];
    localNetMagnetization.z = eq[2] * T1exp + 40 * 1 * (1-T1exp);

    m.x = localNetMagnetization.x * e1.x + localNetMagnetization.y * e2.x;
    m.y = localNetMagnetization.x * e1.y + localNetMagnetization.y * e2.y;
    m.z = localNetMagnetization.z;


    spin_packs[idx] = m;
}



__host__ void printVec3(float3 f) {
    printf("[%f %f %f]\n",f.x, f.y, f.z);
}

__host__ void printMat(mat3x3 m) {
    printVec3(m.r1);
    printVec3(m.r2);
    printVec3(m.r3);
}

float thetime = 0.0;

__host__ void MRI_step(float dt, float3* lab_spins, float3* ref_spins,
                       float* eq, unsigned int w, unsigned int h, float3 _b) {
    cudaMemcpyToSymbol(b, &_b, sizeof(float3));

	dim3 blockDim(512,1,1);
	dim3 gridDim(int(((double)(w*h))/(double)blockDim.x),1,1);
    thetime += dt;
    // MRI_step_kernel_anal<<< gridDim, blockDim >>>(thetime, (float3*)spin_packs, eq, w*h);
    MRI_step_kernel<<< gridDim, blockDim >>>(dt, lab_spins, ref_spins, eq, w*h, thetime);

    float3 v = make_float3(0.8, 0.1, 0.1);
    printf("v = ");
    printVec3(v);

    printf("relax = ");

    printMat(relax(dt, T1, T2));

    float3 v2 = relax(dt, T1, T2) * v;

    printf(" v * relax = ");

    printVec3(v2);
}
