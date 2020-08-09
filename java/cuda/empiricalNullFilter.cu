#include <cuda.h>
#include <curand.h>

__constant__ int imageWidth;
__constant__ int imageHeight;
__constant__ int roiX;
__constant__ int roiY;
__constant__ int roiWidth;
__constant__ int roiHeight;
__constant__ int cacheWidth;
__constant__ int cacheHeight;
__constant__ int kernelRadius;
__constant__ int kernelHeight;
__constant__ int nPoints;
__constant__ int nInitial;
__constant__ int nStep;

//FUNCTION: NORMAL PDF
//args: where to evaluate
//return: pdf (up to a constant)
__device__ float normalPdf(float x) {
  return expf(-x/2);
}

__device__ bool findMode(float* cache, float bandwidth, int* kernelPointers,
    float* nullMean, float* secondDiff, float* densityAtMode) {
  return true;
}

extern "C" __global__ void empiricalNullFilter(float* cache,
    float* bandwidthImage, int* kernelPointers, float* nullMeanImage,
    float* nullStdImage) {

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < roiWidth && y < roiHeight) {
    int imagePointer = (y+roiY)*imageWidth + x + roiX;
    float bandwidth = bandwidthImage[imagePointer];
    float maxDensityAtMode = -INFINITY;
    bool isSuccess;
    float densityAtMode;
    float nullMean;
    float secondDiff;

    for (int i=0; i<nInitial; i++) {
      nullMean = nullMeanImage[imagePointer];
      isSuccess = findMode(cache, bandwidth, kernelPointers, &nullMean,
          &secondDiff, &densityAtMode);
      if (isSuccess) {
        if (densityAtMode > maxDensityAtMode) {
          maxDensityAtMode = densityAtMode;
          nullMeanImage[imagePointer] = nullMean;
          nullStdImage[imagePointer] = powf(-secondDiff, -0.5f);
        }
      }
    }

  }
}
