#include <cuda.h>
#include <curand.h>

__constant__ int imageWidth;
__constant__ int imageHeight;
__constant__ int cacheWidth;
__constant__ int cacheHeight;
__constant__ int kernelRadius;
__constant__ int kernelHeight;
__constant__ int nPoints;
__constant__ int nInitial;
__constant__ int nStep;
__constant__ float bandwidthA;
__constant__ float bandwidthB;

extern "C" __global__ void empiricalNullFilter(float* cache, float* pixels,
    float* std, int* kernelPointers, float* nullMean, float* nullStd) {

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < imageWidth && y < imageHeight) {
    int imagePointer = y*imageWidth + x;
    pixels[imagePointer] = std[imagePointer];
  }
}
