#include <cuda.h>
#include <curand_kernel.h>

//See empiricalNullFilter - this is the main entry
//Notes: row major
__constant__ int roiWidth;
__constant__ int roiHeight;
__constant__ int cacheWidth;
__constant__ int kernelRadius;
__constant__ int kernelHeight;
__constant__ int nInitial; //number of initial values for Newton-Raphson
__constant__ int nStep; //number of steps for Newton-Raphson
__constant__ int cacheSharedWidth; //the width of the shared memory cache

/**FUNCTION: Get derivative
 * Set dxLnF to contain derivatives of the density estimate (of values in the
 *     kernel) evaluated at a point
 * PARAMETERS:
 *   cache: see empiricalNullFilter
 *   bandwidth: see findMode
 *   kernelPointers: see empiricalNullFilter
 *   value: where to evaluate the density estimate and the derivatives
 *   dxLnF: MODIFIED 3-element array, to store results. The elements are:
 *     1. the density (ignore any constant multiplied to it) (NOT THE LOG)
 *     2. the first derivative of the log density
 *     3. the second derivative of the log density
 */
__device__ void getDLnDensity(float* cacheShared, float bandwidth,
    int* kernelPointers, float* value, float* dxLnF) {

  //coordinates of the centre of the kernel
  int x0 = threadIdx.x;
  int y0 = threadIdx.y;

  //variables when going through all pixels in the kernel
  float z; //value of a pixel when looping through kernel
  float sumKernel[3] = {0.0f}; //store sums of weights
  float phiZ; //weight, use Gaussian kernel

  //pointer for cacheShared
  //point to the top left of the kernel
  float* cachePointer = cacheShared + y0*cacheSharedWidth + x0 + kernelRadius;

  //for each row in the kernel
  for (int i=0; i<2*kernelHeight; i++) {
    //for each column for this row
    for (int dx=kernelPointers[i++]; dx<=kernelPointers[i]; dx++) {
      //append to sum
      z = (*(cachePointer+dx) - *value) / bandwidth;
      phiZ = expf(-z*z/2);
      sumKernel[0] += phiZ;
      sumKernel[1] += phiZ * z;
      sumKernel[2] += phiZ * z * z;
    }
    cachePointer += cacheSharedWidth;
  }

  //work out derivatives
  float normaliser = bandwidth*sumKernel[0];
  dxLnF[0] = sumKernel[0];
  dxLnF[1] = sumKernel[1] / normaliser;
  dxLnF[2] = (sumKernel[0]*(sumKernel[2] - sumKernel[0])
      - sumKernel[1]*sumKernel[1]) / (normaliser * normaliser);
}

/**FUNCTION: Find mode
 * Use Newton-Raphson to find the maximum value of the density estimate. Uses
 *     the passed nullMean as the initial value and modifies it at each step,
 *     ending up with a final answer.
 * The second derivative of the log density and the density (up to a constant)
 *     at the final answer is stored in secondDiffLn and densityAtMode.
 * PARAMETERS:
 *   cache: see empiricalNullFilter
 *   bandwidth: bandwidth for the density estimate
 *   kernelPointers: see empiricalNullFilter
 *   nullMean: MODIFIED initial value for the Newton-Raphson method, modified
 *       to contain the final answer
 *   secondDiffLn: MODIFIED second derivative of the log density
 * RETURNS: true if sucessful, false otherwise
 */
__device__ bool findMode(float* cacheShared, float bandwidth, int* kernelPointers,
    float* nullMean, float* secondDiffLn, float* densityAtMode) {
  float dxLnF[3];
  //nStep of Newton-Raphson
  for (int i=0; i<nStep; i++) {
    getDLnDensity(cacheShared, bandwidth, kernelPointers, nullMean, dxLnF);
    *nullMean -= dxLnF[1] / dxLnF[2];
  }
  getDLnDensity(cacheShared, bandwidth, kernelPointers, nullMean, dxLnF);
  //need to check if answer is valid
  if (isfinite(*nullMean) && isfinite(dxLnF[0]) && isfinite(dxLnF[1])
      && isfinite(dxLnF[2]) && (dxLnF[2] < 0)) {
    *densityAtMode = dxLnF[0];
    *secondDiffLn = dxLnF[2];
    return true;
  } else {
    return false;
  }
}

/**FUNCTION: COPY CACHE TO SHARED MEMORY
 * Parameters:
 *   cachedShared: pointer to shared memory
 *   cache: pointer to cache
 *   kernelPointers: see empiricalNullFilter
 */
__device__ void copyCacheToSharedMemory(float* cacheShared, float* cache,
    int* kernelPointers) {
  //copy cache to shared memory
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;
  //point to top left
  float* cachePointer = cache + y0*cacheWidth + x0 + kernelRadius;
  float* cacheSharedPointer = cacheShared + threadIdx.y*cacheSharedWidth
      + threadIdx.x + kernelRadius;
  //for each row in the kernel
  for (int i=0; i<2*kernelHeight; i++) {
    //for each column for this row
    for (int dx=kernelPointers[i++]; dx<=kernelPointers[i]; dx++) {
      *(cacheSharedPointer+dx) = *(cachePointer+dx);
    }
    cachePointer += cacheWidth;
    cacheSharedPointer += cacheSharedWidth;
  }
}

/**KERNEL: Empirical Null Filter
 * Does the empirical null filter on the pixels in cache, giving the empirical
 *     null mean (aka mode) and the empirical null std.
 * PARAMETERS:
 *   cache: array of pixels in the cache
 *   initialSigmaRoi: array of pixels (same size as the ROI) containing standard
 *       deviations, used for producing random initial values for Newton-Raphson
 *   bandwidthRoi: array of pixels (same size as the ROI) containing
 *   kernelPointers: array (even number of elements, size 2*kernelHeight)
 *       containing pairs of integers, indicates for each row the position from
 *       the centre of the kernel
 *   nullMeanRoi: MODIFIED array of pixels (same size as ROI), pass results of
 *       median filter here to be used as initial values. Modified to contain
 *       the empricial null mean afterwards.
 *   nullStdRoi: MODIFIED array of pixels (same size as ROI) to contain the
 *       empirical null std
 */
extern "C" __global__ void empiricalNullFilter(float* cache,
    float* initialSigmaRoi, float* bandwidthRoi, int* kernelPointers,
    float* nullMeanRoi, float* nullStdRoi) {

  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;

  if (x0 < roiWidth && y0 < roiHeight) {

    //get shared memory
    extern __shared__ float cacheShared[];
    float* nullMeanShared = cacheShared
      + (blockDim.x+2*kernelRadius) * (blockDim.y+2*kernelRadius);
    float* secondDiffShared = nullMeanShared + blockDim.x * blockDim.y;

    //copy cache to shared memory
    copyCacheToSharedMemory(cacheShared, cache, kernelPointers);

    int roiIndex = y0*roiWidth + x0;
    int nullSharedIndex = threadIdx.y*blockDim.x + threadIdx.x;
    float* nullMeanSharedPointer = nullMeanShared + nullSharedIndex;
    float* secondDiffSharedPointer = secondDiffShared + nullSharedIndex;

    //for rng
    curandState_t state;
    curand_init(0, roiIndex, 0, &state);
    //nullMean used to store mode for each initial value
    float nullMean = nullMeanRoi[roiIndex]; //use median as first initial
    float median = nullMean;
    //modes with highest densities are stored in shared memory
    *nullMeanSharedPointer = nullMean;
    float sigma = initialSigmaRoi[roiIndex]; //how much noise to add
    float bandwidth = bandwidthRoi[roiIndex]; //bandwidth for density estimate
    bool isSuccess; //indiciate if newton-raphson was sucessful
    float densityAtMode; //density for this particular mode
    //second derivative of the log density, to set empirical null std
    float secondDiffLn;

    //keep solution with the highest density
    float maxDensityAtMode = -INFINITY;

    //try different initial values, the first one is the median, then add normal
        //noise neighbouring shared memory nullMean for different initial values
    int min;
    int nNeighbour;
    float initial0;
    if (nullSharedIndex == 0) {
      min = 0;
    } else {
      min = -1;
    }
    if (nullSharedIndex == blockDim.x*blockDim.y - 1) {
      nNeighbour = 1 - min;
    } else {
      nNeighbour = 2 - min;
    }

    for (int i=0; i<nInitial; i++) {
      isSuccess = findMode(cacheShared, bandwidth, kernelPointers, &nullMean,
          &secondDiffLn, &densityAtMode);
      //keep nullMean and nullStd with the highest density
      if (isSuccess) {
        if (densityAtMode > maxDensityAtMode) {
          maxDensityAtMode = densityAtMode;
          *nullMeanSharedPointer = nullMean;
          *secondDiffSharedPointer = -secondDiffLn;
        }
      }

      //try different initial value
      initial0 = *(nullMeanSharedPointer + i%nNeighbour + min);
      nullMean = (initial0 + median)/2 + sigma * curand_normal(&state);
    }

    //store final results
    nullMeanRoi[roiIndex] = *nullMeanSharedPointer;
    nullStdRoi[roiIndex] = powf(-*secondDiffSharedPointer, -0.5f);
  }
}
