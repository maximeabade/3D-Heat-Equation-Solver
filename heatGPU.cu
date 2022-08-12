#include "heatGPU.h"

__global__ void heatGPU(const float * uIn, float * uOut, const dim3 n, const float dt)
{
    
    float hx = 1./n.x;
    float hy = 1./n.y;
    float hz = 1./n.z;

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i < n.x - 1 && j < n.y - 1 && k < n.z - 1)
    {
        uOut[i + j * n.x + k * n.x * n.y] = uIn[i + j * n.x + k * n.x * n.y] + dt * (
            (uIn[(i-1) + j * n.x + k * n.x * n.y] + 
             uIn[(i+1) + j * n.x + k * n.x * n.y] ) / (hx * hx) +
            (uIn[i + (j-1) * n.x + k * n.x * n.y] + 
             uIn[i + (j+1) * n.x + k * n.x * n.y] ) / (hy * hy) +
            (uIn[i + j * n.x + (k-1) * n.x * n.y] + 
             uIn[i + j * n.x + (k+1) * n.x * n.y] ) / (hz * hz) 
            -uIn[i + j * n.x + k * n.x * n.y] * 2.0 / (hx *hx)
            -uIn[i + j * n.x + k * n.x * n.y] * 2.0 / (hy *hy)
            -uIn[i + j * n.x + k * n.x * n.y] * 2.0 / (hz *hz));
    }
}
