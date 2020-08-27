//MIT License
//Copyright (c) 2020 Sherman Lo

/**GPU version of the Empirical Null Filter
 *
 * Runs the compiled CUDA code. The CUDA cude is compiled into a .ptx file which is then used by the
 *     JCuda package.
 * Tries to be as similar as the CPU version. Main difference is how the mode soutions are shared.
 *     They are shared within block neighbours.
 * Performance is affected by block dimensions, 16x16 and 32x32 were found to be best from very
 *     from quick experimenting and eyeballing.
 * Tolerance for Newton-Raphson is disabled as this introduces branching in GPU code.
 */

package uk.ac.warwick.sip.empiricalnullfilter;

import ij.ImagePlus;
import ij.Macro;
import ij.gui.GenericDialog;
import ij.plugin.filter.PlugInFilterRunner;
import java.awt.Rectangle;
import java.io.InputStream;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdevice_attribute;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUstream;
import jcuda.driver.CUstream_flags;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;

public class EmpiricalNullFilterGpu extends EmpiricalNullFilter {

  //gpu variables
  private static int lastBlockDimX = 16;
  private static int lastBlockDimY = 16;
  private int blockDimX = lastBlockDimX;
  private int blockDimY = lastBlockDimY;

  //default constructor
  public EmpiricalNullFilterGpu() {
  }

  /**OVERRIDE
   * Override to include GPU variables
   */
  @Override
  public int showDialog(ImagePlus imp, String command, PlugInFilterRunner pfr) {
    int flags = super.showDialog(imp, command, pfr);
    if (Macro.getOptions() == null) {
      lastBlockDimX = this.getBlockDimX();
      lastBlockDimY = this.getBlockDimY();
    }
    return flags;
  }

  /**OVERRIDE
   * Override to include GPU variables
   */
  @Override
  public void showOptionsInDialog(GenericDialog genericDialog) {
    genericDialog.addMessage("Advanced options");
    genericDialog.addNumericField("number of initial values", this.getNInitial(), 0, 6, null);
    genericDialog.addNumericField("number of steps", this.getNStep(), 0, 6, null);
    genericDialog.addMessage("GPU options");
    genericDialog.addNumericField("Block dim x", this.getBlockDimX(), 0, 6, null);
    genericDialog.addNumericField("Block dim y", this.getBlockDimY(), 0, 6, null);
  }

  /**OVERRIDE
   * Override to include GPU variables
   */
  @Override
  protected void changeValueFromDialog(GenericDialog genericDialog) throws InvalidValueException {
    try {
      this.setNInitial((int) genericDialog.getNextNumber());
      this.setNStep((int) genericDialog.getNextNumber());
      this.setBlockDimX((int) genericDialog.getNextNumber());
      this.setBlockDimY((int) genericDialog.getNextNumber());
    } catch (InvalidValueException exception) {
      throw exception;
    }
  }

  /**OVERRIDE
   * Do filtering on GPU
   */
  @Override
  protected void doFiltering(final Cache cache) {

    //initalise GPU code
    JCudaDriver.setExceptionsEnabled(true);
    JCudaDriver.cuInit(0);
    CUdevice device = new CUdevice();
    JCudaDriver.cuDeviceGet(device, 0);
    CUcontext context = new CUcontext();
    JCudaDriver.cuCtxCreate(context, 0, device);

    //keep track of all pointers which allocates on device
    ArrayList<CUdeviceptr> devicePointerArray = new ArrayList<CUdeviceptr>();

    //check block dim
    int[] maxBlockDimX = new int[1];
    JCudaDriver.cuDeviceGetAttribute(maxBlockDimX,
        CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device);
    int[] maxBlockDimY = new int[1];
    JCudaDriver.cuDeviceGetAttribute(maxBlockDimY,
        CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device);
    if (this.blockDimX > maxBlockDimX[0]) {
      throw new RuntimeException("Block dimension x " + this.blockDimX + " exceeds maximum of "
          + maxBlockDimX[0]);
    } else if (this.blockDimY > maxBlockDimY[0]) {
      throw new RuntimeException("Block dimension y " + this.blockDimY + " exceeds maximum of "
          + maxBlockDimY[0]);
    }
    //check number of threads
    int[] maxNumberThreads = new int[1];
    int numberOfThreadPerBlock = this.blockDimX*this.blockDimY;
    JCudaDriver.cuDeviceGetAttribute(maxNumberThreads,
        CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
    if (numberOfThreadPerBlock > maxNumberThreads[0]) {
      throw new RuntimeException("Number of threads per block " + numberOfThreadPerBlock
          + " exceeds maximum of " + maxBlockDimX[0]);
    }

    //use try statement so that device memory is freeded when an exception is caught
    try {

      //load ptx code
      String ptxPath = "empiricalNullFilter.ptx";
      InputStream inputStream = this.getClass().getClassLoader().getResourceAsStream(ptxPath);
      Scanner scanner = new Scanner(inputStream);
      Scanner scannerAll = scanner.useDelimiter("\\A");
      String ptx = scannerAll.next();
      scanner.close();

      //load CUDA kernel
      CUmodule module = new CUmodule();
      JCudaDriver.cuModuleLoadData(module, ptx);
      CUfunction kernel = new CUfunction();
      JCudaDriver.cuModuleGetFunction(kernel, module, "empiricalNullFilter");

      //use cpu to get std, median and quantile filtering
      RankFilters rankFilters = new RankFilters();
      rankFilters.imageProcessor = this.imageProcessor;
      rankFilters.roi = this.roi;
      rankFilters.setRadius(this.getRadius());
      rankFilters.filter();

      Rectangle roiRectangle = this.imageProcessor.getRoi();
      int imageWidth = this.imageProcessor.getWidth();
      int roiX = roiRectangle.x;
      int roiY = roiRectangle.y;

      //get variables, put in [] to enumlate pointers
      int[] roiWidth = {roiRectangle.width};
      int[] roiHeight = {roiRectangle.height};
      int[] cacheWidth = {cache.getCacheWidth()};
      int[] kernelRadius = {Kernel.getKRadius()};
      int[] kernelHeight = {Kernel.getKHeight()};
      int[] nInitial = {this.nInitial};
      int[] nStep = {this.nStep};
      int[] cachePointerWidth = new int[1];
      int[] isCopyCacheToShared = new int[1];

      //get image to be filtered
      float[] pixels = (float[]) this.imageProcessor.getPixels();
      //get area of the different images
      int nPixelsInImage = pixels.length;
      int nPixelsInCache = cache.getCache().length;
      int nPixelsInRoi = roiWidth[0] * roiHeight[0];

      //instantiate output images, copy pixels over or fill with nan
      float[] nullMean = new float[nPixelsInImage];
      float[] nullStd = new float[nPixelsInImage];
      float[] std = rankFilters.getOutputImage(STD);
      float[] median = rankFilters.getOutputImage(Q2);
      float[] q1 = rankFilters.getOutputImage(Q1);
      float[] q3 = rankFilters.getOutputImage(Q3);
      int[] nFinite = rankFilters.getNFinite();
      Arrays.fill(nullMean, Float.NaN);
      Arrays.fill(nullStd, Float.NaN);

      //roi versions, smaller than or the same as the image
      //initialSigmaRoi contains standard deviation information, used for generating random initial
          //values
      float[] initialSigmaRoi = new float[nPixelsInRoi];
      float[] bandwidthRoi = new float[nPixelsInRoi];
      float[] nullMeanRoi = new float[nPixelsInRoi];
      float[] nullStdRoi = new float[nPixelsInRoi];
      Pointer h_progressRoi = new Pointer();
      JCudaDriver.cuMemAllocHost(h_progressRoi, nPixelsInRoi*Sizeof.INT);
      IntBuffer progressRoi = h_progressRoi.getByteBuffer(0, nPixelsInRoi*Sizeof.INT).asIntBuffer();

      //get the bandwidth
      int imagePointer;
      int roiPointer;
      float iqr; //iqr / 1.34 for bandwidth
      for (int y=0; y<roiHeight[0]; y++) {
        for (int x=0; x<roiWidth[0]; x++) {

          //pointers
          roiPointer = y*roiWidth[0] + x;
          imagePointer = (y+roiY)*imageWidth + x + roiX;
          //put median in nullMeanRoi so that they used as initial values
          nullMeanRoi[roiPointer] = median[imagePointer];
          bandwidthRoi[roiPointer] = std[imagePointer];
          progressRoi.put(roiPointer, 0);

          //bandwidth and iqr
          iqr = (q3[imagePointer] - q1[imagePointer]) / 1.34f;
          //handle iqr or std = 0
          if (Float.compare(bandwidthRoi[roiPointer], 0.0f) == 0) {
            bandwidthRoi[roiPointer] = 0.289f;
          }
          if (Float.compare(iqr, 0.0f) == 0) {
            iqr = bandwidthRoi[roiPointer];
          }

          //use standard deviation for generating random initial values
          initialSigmaRoi[roiPointer] = bandwidthRoi[roiPointer];

          //min over iqr and std, used for bandwidth
          if (iqr < bandwidthRoi[roiPointer]) {
            bandwidthRoi[roiPointer] = iqr;
          }
          bandwidthRoi[roiPointer] *= this.bandwidthParameterB *
              ((float) Math.pow((double)nFinite[imagePointer], -0.2))
              + this.bandwidthParameterA;
        }
      }

      //work out how much shared memory is needed
      //shared memory to store (for each block):
        //nullMean and nullStd
        //cache with padding (padding is kernelRadius of length)
      int sharedMemorySize = (2*this.blockDimX*this.blockDimY
          + (this.blockDimX+2*kernelRadius[0])*(this.blockDimY+2*kernelRadius[0])) * Sizeof.FLOAT;
      int[] maxSharedSize = new int[1];
      JCudaDriver.cuDeviceGetAttribute(maxSharedSize,
          CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device);

      //check if shared memory is large enough, if not, don't allocate memory for cache
      if (sharedMemorySize > maxSharedSize[0]) {
        //only for nullMean and nullStd
        sharedMemorySize = 2*this.blockDimX*this.blockDimY*Sizeof.FLOAT;
        cachePointerWidth[0] = cacheWidth[0];
        isCopyCacheToShared[0] = 0;
      } else {
        cachePointerWidth[0] = this.blockDimX + 2*kernelRadius[0];
        isCopyCacheToShared[0] = 1;
      }

      //perpare to send variables on contant GPU memory
      //host pointers
      Pointer h_roiWidth = Pointer.to(roiWidth);
      Pointer h_roiHeight = Pointer.to(roiHeight);
      Pointer h_cacheWidth = Pointer.to(cacheWidth);
      Pointer h_kernelRadius = Pointer.to(kernelRadius);
      Pointer h_kernelHeight = Pointer.to(kernelHeight);
      Pointer h_nInitial = Pointer.to(nInitial);
      Pointer h_nStep = Pointer.to(nStep);
      Pointer h_cachePointerWidth = Pointer.to(cachePointerWidth);
      Pointer h_isCopyCacheToShared = Pointer.to(isCopyCacheToShared);

      //device pointers
      CUdeviceptr d_roiWidth = new CUdeviceptr();
      CUdeviceptr d_roiHeight = new CUdeviceptr();
      CUdeviceptr d_cacheWidth = new CUdeviceptr();
      CUdeviceptr d_kernelRadius = new CUdeviceptr();
      CUdeviceptr d_kernelHeight = new CUdeviceptr();
      CUdeviceptr d_nInitial = new CUdeviceptr();
      CUdeviceptr d_nStep = new CUdeviceptr();
      CUdeviceptr d_cachePointerWidth = new CUdeviceptr();
      CUdeviceptr d_isCopyCacheToShared = new CUdeviceptr();

      long[] size = new long[1];

      //get pointers to constant memory
      JCudaDriver.cuModuleGetGlobal(d_roiWidth, size, module, "roiWidth");
      JCudaDriver.cuModuleGetGlobal(d_roiHeight, size, module, "roiHeight");
      JCudaDriver.cuModuleGetGlobal(d_cacheWidth, size, module, "cacheWidth");
      JCudaDriver.cuModuleGetGlobal(d_kernelRadius, size, module, "kernelRadius");
      JCudaDriver.cuModuleGetGlobal(d_kernelHeight, size, module, "kernelHeight");
      JCudaDriver.cuModuleGetGlobal(d_nInitial, size, module, "nInitial");
      JCudaDriver.cuModuleGetGlobal(d_nStep, size, module, "nStep");
      JCudaDriver.cuModuleGetGlobal(d_cachePointerWidth, size, module, "cachePointerWidth");
      JCudaDriver.cuModuleGetGlobal(d_isCopyCacheToShared, size, module, "isCopyCacheToShared");

      //copy from host to device
      JCudaDriver.cuMemcpyHtoD(d_roiWidth, h_roiWidth, Sizeof.INT);
      JCudaDriver.cuMemcpyHtoD(d_roiHeight, h_roiHeight, Sizeof.INT);
      JCudaDriver.cuMemcpyHtoD(d_cacheWidth, h_cacheWidth, Sizeof.INT);
      JCudaDriver.cuMemcpyHtoD(d_kernelRadius, h_kernelRadius, Sizeof.INT);
      JCudaDriver.cuMemcpyHtoD(d_kernelHeight, h_kernelHeight, Sizeof.INT);
      JCudaDriver.cuMemcpyHtoD(d_nInitial, h_nInitial, Sizeof.INT);
      JCudaDriver.cuMemcpyHtoD(d_nStep, h_nStep, Sizeof.INT);
      JCudaDriver.cuMemcpyHtoD(d_cachePointerWidth, h_cachePointerWidth, Sizeof.INT);
      JCudaDriver.cuMemcpyHtoD(d_isCopyCacheToShared, h_isCopyCacheToShared, Sizeof.INT);

      //perpare to send variables through kernel
      //host pointers
      Pointer h_cache = Pointer.to(cache.getCache());
      Pointer h_initialSigmaRoi = Pointer.to(initialSigmaRoi);
      Pointer h_bandwidthRoi = Pointer.to(bandwidthRoi);
      Pointer h_kernelPointers = Pointer.to(Kernel.getKernelPointer());
      Pointer h_nullMeanRoi = Pointer.to(nullMeanRoi);
      Pointer h_nullStdRoi = Pointer.to(nullStdRoi);

      //device pointers
      CUdeviceptr d_cache = new CUdeviceptr();
      CUdeviceptr d_initialSigmaRoi = new CUdeviceptr();
      CUdeviceptr d_bandwidthRoi = new CUdeviceptr();
      CUdeviceptr d_kernelPointers = new CUdeviceptr();
      CUdeviceptr d_nullMeanRoi = new CUdeviceptr();
      CUdeviceptr d_nullStdRoi = new CUdeviceptr();
      CUdeviceptr d_progressRoi = new CUdeviceptr();

      //allocate memory on device
      JCudaDriver.cuMemAlloc(d_cache, Sizeof.FLOAT*nPixelsInCache);
      devicePointerArray.add(d_cache);
      JCudaDriver.cuMemAlloc(d_initialSigmaRoi, Sizeof.FLOAT*nPixelsInRoi);
      devicePointerArray.add(d_initialSigmaRoi);
      JCudaDriver.cuMemAlloc(d_bandwidthRoi, Sizeof.FLOAT*nPixelsInRoi);
      devicePointerArray.add(d_bandwidthRoi);
      JCudaDriver.cuMemAlloc(d_kernelPointers, Sizeof.INT*2*Kernel.getKHeight());
      devicePointerArray.add(d_kernelPointers);
      JCudaDriver.cuMemAlloc(d_nullMeanRoi, Sizeof.FLOAT*nPixelsInRoi);
      devicePointerArray.add(d_nullMeanRoi);
      JCudaDriver.cuMemAlloc(d_nullStdRoi, Sizeof.FLOAT*nPixelsInRoi);
      devicePointerArray.add(d_nullStdRoi);
      JCudaDriver.cuMemAlloc(d_progressRoi, Sizeof.INT*nPixelsInRoi);
      devicePointerArray.add(d_progressRoi);

      //copy from host to device for the kernel parameters
      JCudaDriver.cuMemcpyHtoD(d_cache, h_cache, Sizeof.FLOAT*nPixelsInCache);
      JCudaDriver.cuMemcpyHtoD(d_initialSigmaRoi, h_initialSigmaRoi, Sizeof.FLOAT*nPixelsInRoi);
      JCudaDriver.cuMemcpyHtoD(d_bandwidthRoi, h_bandwidthRoi, Sizeof.FLOAT*nPixelsInRoi);
      JCudaDriver.cuMemcpyHtoD(d_kernelPointers, h_kernelPointers,
          Sizeof.INT*2*Kernel.getKHeight());
      JCudaDriver.cuMemcpyHtoD(d_nullMeanRoi, h_nullMeanRoi, Sizeof.FLOAT*nPixelsInRoi);
      JCudaDriver.cuMemcpyHtoD(d_nullStdRoi, h_nullStdRoi, Sizeof.FLOAT*nPixelsInRoi);
      JCudaDriver.cuMemcpyHtoD(d_progressRoi, h_progressRoi, Sizeof.INT*nPixelsInRoi);

      //put pointers in pointers, to pass to kernel
      Pointer kernelParameters = Pointer.to(
          Pointer.to(d_cache),
          Pointer.to(d_initialSigmaRoi),
          Pointer.to(d_bandwidthRoi),
          Pointer.to(d_kernelPointers),
          Pointer.to(d_nullMeanRoi),
          Pointer.to(d_nullStdRoi),
          Pointer.to(d_progressRoi)
      );

      //call kernel
      int nBlockX = (roiWidth[0] + this.blockDimX - 1) / this.blockDimX;
      int nBlockY = (roiHeight[0] + this.blockDimY - 1) / this.blockDimY;
      JCudaDriver.cuLaunchKernel(kernel, nBlockX, nBlockY, 1, this.blockDimX, this.blockDimY, 1,
          sharedMemorySize, null, kernelParameters, null);

      if (this.isShowProgressBar) {
        //while the kernel is running, keep track of progress
        CUstream cuStream = new CUstream();
        //use try except to free cuStream when ending
        try {
          JCudaDriver.cuStreamCreate(cuStream, CUstream_flags.CU_STREAM_NON_BLOCKING);
          int nPixelsDone = 0;
          while (nPixelsDone != nPixelsInRoi) {
            JCudaDriver.cuMemcpyDtoHAsync(h_progressRoi, d_progressRoi,
                Sizeof.INT*nPixelsInRoi, cuStream);
            JCudaDriver.cuStreamSynchronize(cuStream);
            nPixelsDone = 0;
            for (int i=0; i<nPixelsInRoi; i++) {
              nPixelsDone += progressRoi.get(i);
            }
            this.showProgress((double)nPixelsDone / (double)nPixelsInRoi);
          }
        }
        catch (Exception exception) {
          //do nothing, just no progress bar
        } finally {
          JCudaDriver.cuStreamDestroy(cuStream);
        }
      }

      //copy results over, device to host
      JCudaDriver.cuMemcpyDtoH(h_nullMeanRoi, d_nullMeanRoi, Sizeof.FLOAT*nPixelsInRoi);
      JCudaDriver.cuMemcpyDtoH(h_nullStdRoi, d_nullStdRoi, Sizeof.FLOAT*nPixelsInRoi);

      //copy roi image to the actual image
      //do the filtering given the null mean and null std
      float[] nullMeanStd = new float[2];
      for (int y=0; y<roiHeight[0]; y++) {
        for (int x=0; x<roiWidth[0]; x++) {
          if (this.roi.contains(this.roi.getBounds().x+x, this.roi.getBounds().y+y)) {
            roiPointer = y*roiWidth[0] + x;
            imagePointer = (y+roiY)*imageWidth + x + roiX;
            nullMean[imagePointer] = nullMeanRoi[roiPointer];
            nullStd[imagePointer] = nullStdRoi[roiPointer];

            nullMeanStd[0] = nullMean[imagePointer];
            nullMeanStd[1] = nullStd[imagePointer];
            this.updatePixelInImage(pixels, imagePointer, nullMeanStd);
          }
        }
      }

      //copy pixels over to outputImageArrau
      for (int i=0; i<this.nImageOutput; i++) {
        if ((this.outputImagePointer >> i) % 2 == 1) {
          switch (i) {
            case 0:
              this.outputImageArray[i].setPixels(nullMean);
              break;
            case 1:
              this.outputImageArray[i].setPixels(nullStd);
              break;
            case 2:
              this.outputImageArray[i].setPixels(std);
              break;
            case 3:
              this.outputImageArray[i].setPixels(q1);
              break;
            case 4:
              this.outputImageArray[i].setPixels(median);
              break;
            case 5:
              this.outputImageArray[i].setPixels(q3);
              break;
          }
        }
      }
      this.showProgress(1.0);
      this.pass++;
    } catch (Exception exception) {
      throw exception;
    } finally {
      //free memory and close device
      for (int i=0; i<devicePointerArray.size(); i++) {
        JCudaDriver.cuMemFree(devicePointerArray.get(i));
      }
      JCudaDriver.cuCtxDestroy(context);
      JCuda.cudaDeviceReset();
    }
  }

  protected int getBlockDimX() {
    return this.blockDimX;
  }

  protected int getBlockDimY() {
    return this.blockDimY;
  }

  public void setBlockDimX(int blockDimX) {
    this.blockDimX = blockDimX;
  }

  public void setBlockDimY(int blockDimY) {
    this.blockDimY = blockDimY;
  }

}
