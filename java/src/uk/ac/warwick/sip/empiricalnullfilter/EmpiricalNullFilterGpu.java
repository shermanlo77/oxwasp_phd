package uk.ac.warwick.sip.empiricalnullfilter;

import ij.ImagePlus;
import ij.Macro;
import ij.gui.GenericDialog;
import ij.plugin.filter.ExtendedPlugInFilter;
import ij.plugin.filter.PlugInFilterRunner;
import java.awt.Rectangle;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.RandomGenerator;

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
    genericDialog.addNumericField("bandwidth A", this.getBandwidthA(), 2, 6, null);
    genericDialog.addNumericField("bandwidth B", this.getBandwidthB(), 2, 6, null);
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
      this.setBandwidthA((float) genericDialog.getNextNumber());
      this.setBandwidthB((float) genericDialog.getNextNumber());
      this.setBlockDimX((int) genericDialog.getNextNumber());
      this.setBlockDimY((int) genericDialog.getNextNumber());
    } catch (InvalidValueException exception) {
      throw exception;
    }
  }


  /**OVERRIDE
   * Reflective padding (to avoid branching on GPU)
   */
  @Override
  protected Cache instantiateCache() {
    return new CacheReflect(this.imageProcessor, this.roi);
  }

  /**OVERRIDE
   * Do filtering on GPU
   */
  @Override
  protected void doFiltering(final Cache cache) {

    //use cpu to get std, median and quantile filtering
    RankFilters rankFilters = new RankFilters();
    rankFilters.imageProcessor = this.imageProcessor;
    rankFilters.roi = this.roi;
    rankFilters.filter();

    Rectangle roiRectangle = this.imageProcessor.getRoi();

    //get variables, put in [] to enumlate pointers
    int[] imageWidth = {this.imageProcessor.getWidth()};
    int[] imageHeight = {this.imageProcessor.getHeight()};
    int[] roiX = {roiRectangle.x};
    int[] roiY = {roiRectangle.y};
    int[] roiWidth = {roiRectangle.width};
    int[] roiHeight = {roiRectangle.height};
    int[] cacheWidth = {cache.getCacheWidth()};
    int[] cacheHeight = {cache.getCacheHeight()};
    int[] kernelRadius = {Kernel.getKRadius()};
    int[] kernelHeight = {Kernel.getKHeight()};
    int[] nPoints = {Kernel.getKNPoints()};
    int[] nInitial = {this.nInitial};
    int[] nStep = {this.nStep};

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
    Arrays.fill(nullMean, Float.NaN);
    Arrays.fill(nullStd, Float.NaN);

    //roi versions, smaller than or the same as the image
    //initialSigmaRoi contains standard deviation information, used for generating random initial
        //values
    float[] initialSigmaRoi = new float[nPixelsInRoi];
    float[] bandwidthRoi = new float[nPixelsInRoi];
    float[] nullMeanRoi = new float[nPixelsInRoi];
    float[] nullStdRoi = new float[nPixelsInRoi];

    //get the bandwidth
    int imagePointer;
    int roiPointer;
    float iqr; //iqr / 1.34 for bandwidth
    for (int y=0; y<roiHeight[0]; y++) {
      for (int x=0; x<roiWidth[0]; x++) {

        //pointers
        roiPointer = y*roiWidth[0] + x;
        imagePointer = (y+roiY[0])*imageWidth[0] + x + roiX[0];
        //put median in nullMeanRoi so that they used as initial values
        nullMeanRoi[roiPointer] = median[imagePointer];
        bandwidthRoi[roiPointer] = std[imagePointer];

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
            ((float) Math.pow((double)Kernel.getKNPoints(), -0.2))
            + this.bandwidthParameterA;
      }
    }

    //gpu code
    JCudaDriver.setExceptionsEnabled(true);
    //initalise CUDA
    JCudaDriver.cuInit(0);
    CUdevice device = new CUdevice();
    JCudaDriver.cuDeviceGet(device, 0);
    CUcontext context = new CUcontext();
    JCudaDriver.cuCtxCreate(context, 0, device);

    //load ptx code
    String ptxPath = "empiricalNullFilter.ptx";
    InputStream inputStream = this.getClass().getClassLoader().getResourceAsStream(ptxPath);
    Scanner scanner = new Scanner(inputStream);
    Scanner scannerAll = scanner.useDelimiter("\\A");
    String ptx = scannerAll.next();

    //load CUDA kernel
    CUmodule module = new CUmodule();
    JCudaDriver.cuModuleLoadData(module, ptx);
    CUfunction kernel = new CUfunction();
    JCudaDriver.cuModuleGetFunction(kernel, module, "empiricalNullFilter");

    //perpare to send variables on contant GPU memory
    //host pointers
    Pointer h_roiWidth = Pointer.to(roiWidth);
    Pointer h_roiHeight = Pointer.to(roiHeight);
    Pointer h_cacheWidth = Pointer.to(cacheWidth);
    Pointer h_cacheHeight = Pointer.to(cacheHeight);
    Pointer h_kernelRadius = Pointer.to(kernelRadius);
    Pointer h_kernelHeight = Pointer.to(kernelHeight);
    Pointer h_nPoints = Pointer.to(nPoints);
    Pointer h_nInitial = Pointer.to(nInitial);
    Pointer h_nStep = Pointer.to(nStep);

    //device pointers
    CUdeviceptr d_roiWidth = new CUdeviceptr();
    CUdeviceptr d_roiHeight = new CUdeviceptr();
    CUdeviceptr d_cacheWidth = new CUdeviceptr();
    CUdeviceptr d_cacheHeight = new CUdeviceptr();
    CUdeviceptr d_kernelRadius = new CUdeviceptr();
    CUdeviceptr d_kernelHeight = new CUdeviceptr();
    CUdeviceptr d_nPoints = new CUdeviceptr();
    CUdeviceptr d_nInitial = new CUdeviceptr();
    CUdeviceptr d_nStep = new CUdeviceptr();

    long[] size = new long[1];

    //get pointers to constant memory
    JCudaDriver.cuModuleGetGlobal(d_roiWidth, size, module, "roiWidth");
    JCudaDriver.cuModuleGetGlobal(d_roiHeight, size, module, "roiHeight");
    JCudaDriver.cuModuleGetGlobal(d_cacheWidth, size, module, "cacheWidth");
    JCudaDriver.cuModuleGetGlobal(d_cacheHeight, size, module, "cacheHeight");
    JCudaDriver.cuModuleGetGlobal(d_kernelRadius, size, module, "kernelRadius");
    JCudaDriver.cuModuleGetGlobal(d_kernelHeight, size, module, "kernelHeight");
    JCudaDriver.cuModuleGetGlobal(d_nPoints, size, module, "nPoints");
    JCudaDriver.cuModuleGetGlobal(d_nInitial, size, module, "nInitial");
    JCudaDriver.cuModuleGetGlobal(d_nStep, size, module, "nStep");

    //copy from host to device
    JCudaDriver.cuMemcpyHtoD(d_roiWidth, h_roiWidth, Sizeof.INT);
    JCudaDriver.cuMemcpyHtoD(d_roiHeight, h_roiHeight, Sizeof.INT);
    JCudaDriver.cuMemcpyHtoD(d_cacheWidth, h_cacheWidth, Sizeof.INT);
    JCudaDriver.cuMemcpyHtoD(d_cacheHeight, h_cacheHeight, Sizeof.INT);
    JCudaDriver.cuMemcpyHtoD(d_kernelRadius, h_kernelRadius, Sizeof.INT);
    JCudaDriver.cuMemcpyHtoD(d_kernelHeight, h_kernelHeight, Sizeof.INT);
    JCudaDriver.cuMemcpyHtoD(d_nPoints, h_nPoints, Sizeof.INT);
    JCudaDriver.cuMemcpyHtoD(d_nInitial, h_nInitial, Sizeof.INT);
    JCudaDriver.cuMemcpyHtoD(d_nStep, h_nStep, Sizeof.INT);

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

    //keep track of all pointers which allocates on device
    ArrayList<CUdeviceptr> devicePointerArray = new ArrayList<CUdeviceptr>();
    //use try statement so that device memory is freeded when an exception is caught
    try {
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

      //copy from host to device for the kernel parameters
      JCudaDriver.cuMemcpyHtoD(d_cache, h_cache, Sizeof.FLOAT*nPixelsInCache);
      JCudaDriver.cuMemcpyHtoD(d_initialSigmaRoi, h_bandwidthRoi, Sizeof.FLOAT*nPixelsInRoi);
      JCudaDriver.cuMemcpyHtoD(d_bandwidthRoi, h_bandwidthRoi, Sizeof.FLOAT*nPixelsInRoi);
      JCudaDriver.cuMemcpyHtoD(d_kernelPointers, h_kernelPointers, Sizeof.INT*2*Kernel.getKHeight());
      JCudaDriver.cuMemcpyHtoD(d_nullMeanRoi, h_nullMeanRoi, Sizeof.FLOAT*nPixelsInRoi);
      JCudaDriver.cuMemcpyHtoD(d_nullStdRoi, h_nullStdRoi, Sizeof.FLOAT*nPixelsInRoi);

      //put pointers in pointers, to pass to kernel
      Pointer kernelParameters = Pointer.to(
          Pointer.to(d_cache),
          Pointer.to(d_initialSigmaRoi),
          Pointer.to(d_bandwidthRoi),
          Pointer.to(d_kernelPointers),
          Pointer.to(d_nullMeanRoi),
          Pointer.to(d_nullStdRoi)
      );

      //call kernel
      int nBlockX = (roiWidth[0] + this.blockDimX - 1) / this.blockDimX;
      int nBlockY = (roiHeight[0] + this.blockDimY - 1) / this.blockDimY;

      JCudaDriver.cuLaunchKernel(kernel, nBlockX, nBlockY, 1, this.blockDimX, this.blockDimY, 1,
          0, null, kernelParameters, null);

      //copy results over, device to host
      JCudaDriver.cuMemcpyDtoH(h_nullMeanRoi, d_nullMeanRoi, Sizeof.FLOAT*nPixelsInRoi);
      JCudaDriver.cuMemcpyDtoH(h_nullStdRoi, d_nullStdRoi, Sizeof.FLOAT*nPixelsInRoi);
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

    //copy roi image to the actual image
    //do the filtering given the null mean and null std
    float[] nullMeanStd = new float[2];
    for (int y=0; y<roiHeight[0]; y++) {
      for (int x=0; x<roiWidth[0]; x++) {
        roiPointer = y*roiWidth[0] + x;
        imagePointer = (y+roiY[0])*imageWidth[0] + x + roiX[0];
        nullMean[imagePointer] = nullMeanRoi[roiPointer];
        nullStd[imagePointer] = nullStdRoi[roiPointer];

        nullMeanStd[0] = nullMean[imagePointer];
        nullMeanStd[1] = nullStd[imagePointer];
        this.updatePixelInImage(pixels, imagePointer, nullMeanStd);
      }
    }

    //copy pixels over to outputImageArrau
    for (int i=0; i<this.n_image_output; i++) {
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

  }

  protected int getBlockDimX() {
    return this.blockDimX;
  }

  protected int getBlockDimY() {
    return this.blockDimY;
  }

  protected void setBlockDimX(int blockDimX) {
    this.blockDimX = blockDimX;
  }

  protected void setBlockDimY(int blockDimY) {
    this.blockDimY = blockDimY;
  }

}
