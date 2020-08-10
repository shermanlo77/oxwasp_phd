#include <cuda.h>
#include <curand.h>

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
    float* bandwidthRoi, int* kernelPointers, float* nullMeanRoi,
    float* nullStdRoi) {

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < roiWidth && y < roiHeight) {
    int roiPointer = y*roiWidth + x;
    float bandwidth = bandwidthRoi[roiPointer];
    float maxDensityAtMode = -INFINITY;
    bool isSuccess;
    float densityAtMode;
    float nullMean;
    float secondDiff;

    for (int i=0; i<nInitial; i++) {
      nullMean = nullMeanRoi[roiPointer];
      isSuccess = findMode(cache, bandwidth, kernelPointers, &nullMean,
          &secondDiff, &densityAtMode);
      if (isSuccess) {
        if (densityAtMode > maxDensityAtMode) {
          maxDensityAtMode = densityAtMode;
          nullMeanRoi[roiPointer] = nullMean;
          nullStdRoi[roiPointer] = powf(-secondDiff, -0.5f);
        }
      }
    }

  }
}
