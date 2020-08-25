//MIT License
//Copyright (c) 2020 Sherman Lo

#include <cuda.h>
#include <curand_kernel.h>

//See empiricalNullFilter - this is the main entry
//Notes: row major
//Note: shared memory is used to store the empirical null mean and std. IF big
    //enough, also the cache. Size becomes a problem if the kernel radius
    //becomes too big, in this case, the cache lives in global memory and
    //hopefully may be picked up in L1 and L2
  //set cachePointerWidth = cacheWidth if isCopyCacheToShared is false
  //set cachePointerWidth = blockDim.x + 2*kernelRadius if isCopyCacheToShared
      //is true
__constant__ int roiWidth;
__constant__ int roiHeight;
__constant__ int cacheWidth;
__constant__ int kernelRadius;
__constant__ int kernelHeight;
__constant__ int nInitial; //number of initial values for Newton-Raphson
__constant__ int nStep; //number of steps for Newton-Raphson
__constant__ int cachePointerWidth; //the width of the shared memory cache
__constant__ int isCopyCacheToShared; //indicate to copy cache to shared mem

/**FUNCTION: Get derivative
 * Set dxLnF to contain derivatives of the density estimate (of values in the
 *     kernel) evaluated at a point
 * PARAMETERS:
 *   cachePointer: see empiricalNullFilter
 *   bandwidth: see findMode
 *   kernelPointers: see empiricalNullFilter
 *   value: where to evaluate the density estimate and the derivatives
 *   dxLnF: MODIFIED 3-element array, to store results. The elements are:
 *     1. the density (ignore any constant multiplied to it) (NOT THE LOG)
 *     2. the first derivative of the log density
 *     3. the second derivative of the log density
 */
__device__ void getDLnDensity(float* cachePointer, float bandwidth,
    int* kernelPointers, float* value, float* dxLnF) {

  //variables when going through all pixels in the kernel
  float z; //value of a pixel when looping through kernel
  float sumKernel[3] = {0.0f}; //store sums of weights
  float phiZ; //weight, use Gaussian kernel

  //pointer for cacheShared
  //point to the top left of the kernel
  cachePointer -= kernelRadius*cachePointerWidth;

  //for each row in the kernel
  for (int i=0; i<2*kernelHeight; i++) {
    //for each column for this row
    for (int dx=kernelPointers[i++]; dx<=kernelPointers[i]; dx++) {
      //append to sum if the value in cachePointer is finite
      z = *(cachePointer+dx);
      if (isfinite(z)) {
        z -= *value;
        z /= bandwidth;
        phiZ = expf(-z*z/2);
        sumKernel[0] += phiZ;
        sumKernel[1] += phiZ * z;
        sumKernel[2] += phiZ * z * z;
      }
    }
    cachePointer += cachePointerWidth;
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
 *   cachePointer: see empiricalNullFilter
 *   bandwidth: bandwidth for the density estimate
 *   kernelPointers: see empiricalNullFilter
 *   nullMean: MODIFIED initial value for the Newton-Raphson method, modified
 *       to contain the final answer
 *   secondDiffLn: MODIFIED second derivative of the log density
 * RETURNS: true if sucessful, false otherwise
 */
__device__ bool findMode(float* cachePointer, float bandwidth,
    int* kernelPointers,float* nullMean, float* secondDiffLn,
    float* densityAtMode) {
  float dxLnF[3];
  //nStep of Newton-Raphson
  for (int i=0; i<nStep; i++) {
    getDLnDensity(cachePointer, bandwidth, kernelPointers, nullMean, dxLnF);
    *nullMean -= dxLnF[1] / dxLnF[2];
  }
  getDLnDensity(cachePointer, bandwidth, kernelPointers, nullMean, dxLnF);
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
__device__ void copyCacheToSharedMemory(float* dest, float* source,
    int* kernelPointers) {
  //point to top left
  dest -= kernelRadius*cachePointerWidth;
  source -= kernelRadius*cacheWidth;
  //for each row in the kernel
  for (int i=0; i<2*kernelHeight; i++) {
    //for each column for this row
    for (int dx=kernelPointers[i++]; dx<=kernelPointers[i]; dx++) {
      *(dest+dx) = *(source+dx);
    }
    source += cacheWidth;
    dest += cachePointerWidth;
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
 *   progressRoi: MODIFIED array of pixels (same size as ROI) initally contains
 *       all zeros. A filtered pixel will change it to a one.
 */
extern "C" __global__ void empiricalNullFilter(float* cache,
    float* initialSigmaRoi, float* bandwidthRoi, int* kernelPointers,
    float* nullMeanRoi, float* nullStdRoi, int* progressRoi) {

  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;
  //adjust pointer to the corresponding x y coordinates
  cache += (y0+kernelRadius)*cacheWidth + x0 + kernelRadius;
  //check if in roi
  //&&isfinite(*cache) is not required as accessing the cache from this
    //pixel is within bounds
  bool isInRoi = x0 < roiWidth && y0 < roiHeight;

  //get shared memory
  extern __shared__ float sharedMemory[];
  float* nullMeanSharedPointer = sharedMemory;
  float* secondDiffSharedPointer = nullMeanSharedPointer
      + blockDim.x * blockDim.y;
  float* cachePointer;

  //offset by the x and y coordinates
  int roiIndex = y0*roiWidth + x0;
  int nullSharedIndex = threadIdx.y*blockDim.x + threadIdx.x;

  //if the shared memory is big enough, copy the cache
  //cachePointer points to shared memory if shared memory allows it, otherwise
      //points to global memory
  if (isCopyCacheToShared) {
    cachePointer = secondDiffSharedPointer + blockDim.x * blockDim.y;
    cachePointer += (threadIdx.y+kernelRadius)*cachePointerWidth
        + threadIdx.x + kernelRadius;
    //copy cache to shared memory
    if (isInRoi) {
      copyCacheToSharedMemory(cachePointer, cache, kernelPointers);
    }
  } else {
    cachePointer = cache;
  }
  __syncthreads();

  //adjust pointer to the corresponding x y coordinates
  nullMeanSharedPointer += nullSharedIndex;
  secondDiffSharedPointer += nullSharedIndex;

  //for rng
  curandState_t state;
  curand_init(0, roiIndex, 0, &state);
  //nullMean used to store mode for each initial value
  float nullMean;
  float median;
  float sigma; //how much noise to add
  float bandwidth; //bandwidth for density estimate

  if (isInRoi) {
    nullMean = nullMeanRoi[roiIndex]; //use median as first initial
    median = nullMean;
    //modes with highest densities are stored in shared memory
    *nullMeanSharedPointer = nullMean;
    sigma = initialSigmaRoi[roiIndex]; //how much noise to add
    bandwidth = bandwidthRoi[roiIndex]; //bandwidth for density estimate
  }

  bool isSuccess; //indiciate if newton-raphson was sucessful
  float densityAtMode; //density for this particular mode
  //second derivative of the log density, to set empirical null std
  float secondDiffLn;
  //keep solution with the highest density
  float maxDensityAtMode = -INFINITY;

  //try different initial values, the first one is the median, then add
      //normal noise neighbouring shared memory nullMean for different
      //initial values rotate from -1, itself and +1 from current pointer
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
    if (isInRoi) {
      isSuccess = findMode(cachePointer, bandwidth, kernelPointers, &nullMean,
          &secondDiffLn, &densityAtMode);
      //keep nullMean and nullStd with the highest density
      if (isSuccess) {
        if (densityAtMode > maxDensityAtMode) {
          maxDensityAtMode = densityAtMode;
          *nullMeanSharedPointer = nullMean;
          *secondDiffSharedPointer = secondDiffLn;
        }
      }
    }

    //try different initial value
    __syncthreads();

    if (isInRoi) {
      initial0 = *(nullMeanSharedPointer + i%nNeighbour + min);
      //ensure the initial value is finite, otherwise use previous solution
      if (!isfinite(initial0)) {
        initial0 = nullMean;
      }
      nullMean = (initial0 + median)/2 + sigma * curand_normal(&state);
    }
  }

  //store final results
  if (isInRoi) {
    nullMeanRoi[roiIndex] = *nullMeanSharedPointer;
    nullStdRoi[roiIndex] = powf(-*secondDiffSharedPointer, -0.5f);
    progressRoi[roiIndex] = 1;
  }

}
