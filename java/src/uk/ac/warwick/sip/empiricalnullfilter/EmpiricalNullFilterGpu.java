package uk.ac.warwick.sip.empiricalnullfilter;

import ij.ImagePlus;
import ij.Macro;
import ij.gui.GenericDialog;
import ij.plugin.filter.ExtendedPlugInFilter;
import ij.plugin.filter.PlugInFilterRunner;
import java.awt.Rectangle;
import java.io.InputStream;
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

  private static int lastBlockDimX = 32;
  private static int lastBlockDimY = 32;

  private int blockDimX = lastBlockDimX;
  private int blockDimY = lastBlockDimY;

  public EmpiricalNullFilterGpu() {
  }

  @Override
  public int showDialog(ImagePlus imp, String command, PlugInFilterRunner pfr) {
    int flags = super.showDialog(imp, command, pfr);
    if (Macro.getOptions() == null) {
      lastBlockDimX = this.getBlockDimX();
      lastBlockDimY = this.getBlockDimY();
    }
    return flags;
  }

  @Override
  public void showOptionsInDialog(GenericDialog genericDialog) {
    //add fields for the empirical null tuning parameters
    //integers do not show decimal points
    genericDialog.addMessage("Advanced options");
    genericDialog.addNumericField("number of initial values", this.getNInitial(),
        0, 6, null);
    genericDialog.addNumericField("number of steps", this.getNStep(),
        0, 6, null);
    genericDialog.addNumericField("bandwidth A", this.getBandwidthA(),
        2, 6, null);
    genericDialog.addNumericField("bandwidth B", this.getBandwidthB(),
        2, 6, null);
    genericDialog.addMessage("GPU options");
    genericDialog.addNumericField("Block dim x", this.getBlockDimX(),
        0, 6, null);
    genericDialog.addNumericField("Block dim y", this.getBlockDimY(),
        0, 6, null);
  }

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

  @Override
  protected Cache instantiateCache() {
    return new CacheReflect(this.imageProcessor, this.roi);
  }

  @Override
  protected void doFiltering(final Cache cache) {
    RankFilters rankFilters = new RankFilters();
    rankFilters.imageProcessor = this.imageProcessor;
    rankFilters.roi = this.roi;
    rankFilters.filter();

    float[] pixels = (float[]) this.imageProcessor.getPixels();
    int nPixelsInImage = pixels.length;
    int nPixelsInCache = cache.getCache().length;

    float[] nullMean = new float[nPixelsInImage];
    float[] nullStd = new float[nPixelsInImage];
    float[] std = rankFilters.getOutputImage(STD);
    float[] median = rankFilters.getOutputImage(Q2);
    float[] q1 = rankFilters.getOutputImage(Q1);
    float[] q3 = rankFilters.getOutputImage(Q3);

    Arrays.fill(nullMean, Float.NaN);
    Arrays.fill(nullStd, Float.NaN);

    float[] bandwidth = new float[nPixelsInImage];
    for (int i=0; i<nPixelsInImage; i++) {
      nullMean[i] = median[i];
      bandwidth[i] = std[i];
    }

    float iqr;
    for (int i=0; i<std.length; i++) {
      iqr = (q3[i] - q1[i]) / 1.34f;
      if (Float.compare(bandwidth[i], 0.0f) == 0) {
        bandwidth[i] = 0.289f;
      }
      if (Float.compare(iqr, 0.0f) == 0) {
        iqr = bandwidth[i];
      }
      if (iqr < bandwidth[i]) {
        bandwidth[i] = iqr;
      }
      bandwidth[i] *= this.bandwidthParameterB *
          ((float) Math.pow((double)Kernel.getKNPoints(), -0.2))
          + this.bandwidthParameterA;
    }

    //roi = region of interest
    Rectangle roiRectangle = this.imageProcessor.getRoi();

    //gpu code
    JCudaDriver.setExceptionsEnabled(true);
    //initalise CUDA
    JCudaDriver.cuInit(0);
    CUdevice device = new CUdevice();
    JCudaDriver.cuDeviceGet(device, 0);
    CUcontext context = new CUcontext();
    JCudaDriver.cuCtxCreate(context, 0, device);

    //load CUDA kernel
    String ptxPath = "empiricalNullFilter.ptx";
    InputStream inputStream = this.getClass().getClassLoader().getResourceAsStream(ptxPath);
    Scanner scanner = new Scanner(inputStream);
    Scanner scannerAll = scanner.useDelimiter("\\A");
    String ptx = scannerAll.next();

    CUmodule module = new CUmodule();
    JCudaDriver.cuModuleLoadData(module, ptx);
    CUfunction kernel = new CUfunction();
    JCudaDriver.cuModuleGetFunction(kernel, module, "empiricalNullFilter");

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

    Pointer h_imageWidth = Pointer.to(imageWidth);
    Pointer h_imageHeight = Pointer.to(imageHeight);
    Pointer h_roiX = Pointer.to(roiX);
    Pointer h_roiY = Pointer.to(roiY);
    Pointer h_roiWidth = Pointer.to(roiWidth);
    Pointer h_roiHeight = Pointer.to(roiHeight);
    Pointer h_cacheWidth = Pointer.to(cacheWidth);
    Pointer h_cacheHeight = Pointer.to(cacheHeight);
    Pointer h_kernelRadius = Pointer.to(kernelRadius);
    Pointer h_kernelHeight = Pointer.to(kernelHeight);
    Pointer h_nPoints = Pointer.to(nPoints);
    Pointer h_nInitial = Pointer.to(nInitial);
    Pointer h_nStep = Pointer.to(nStep);

    CUdeviceptr d_imageWidth = new CUdeviceptr();
    CUdeviceptr d_imageHeight = new CUdeviceptr();
    CUdeviceptr d_roiX = new CUdeviceptr();
    CUdeviceptr d_roiY = new CUdeviceptr();
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

    JCudaDriver.cuModuleGetGlobal(d_imageWidth, size, module, "imageWidth");
    JCudaDriver.cuModuleGetGlobal(d_imageHeight, size, module, "imageHeight");
    JCudaDriver.cuModuleGetGlobal(d_roiX, size, module, "roiX");
    JCudaDriver.cuModuleGetGlobal(d_roiY, size, module, "roiY");
    JCudaDriver.cuModuleGetGlobal(d_roiWidth, size, module, "roiWidth");
    JCudaDriver.cuModuleGetGlobal(d_roiHeight, size, module, "roiHeight");
    JCudaDriver.cuModuleGetGlobal(d_cacheWidth, size, module, "cacheWidth");
    JCudaDriver.cuModuleGetGlobal(d_cacheHeight, size, module, "cacheHeight");
    JCudaDriver.cuModuleGetGlobal(d_kernelRadius, size, module, "kernelRadius");
    JCudaDriver.cuModuleGetGlobal(d_kernelHeight, size, module, "kernelHeight");
    JCudaDriver.cuModuleGetGlobal(d_nPoints, size, module, "nPoints");
    JCudaDriver.cuModuleGetGlobal(d_nInitial, size, module, "nInitial");
    JCudaDriver.cuModuleGetGlobal(d_nStep, size, module, "nStep");

    JCudaDriver.cuMemcpyHtoD(d_imageWidth, h_imageWidth, Sizeof.INT);
    JCudaDriver.cuMemcpyHtoD(d_imageHeight, h_imageHeight, Sizeof.INT);
    JCudaDriver.cuMemcpyHtoD(d_roiX, h_roiX, Sizeof.INT);
    JCudaDriver.cuMemcpyHtoD(d_roiY, h_roiY, Sizeof.INT);
    JCudaDriver.cuMemcpyHtoD(d_roiWidth, h_roiWidth, Sizeof.INT);
    JCudaDriver.cuMemcpyHtoD(d_roiHeight, h_roiHeight, Sizeof.INT);
    JCudaDriver.cuMemcpyHtoD(d_cacheWidth, h_cacheWidth, Sizeof.INT);
    JCudaDriver.cuMemcpyHtoD(d_cacheHeight, h_cacheHeight, Sizeof.INT);
    JCudaDriver.cuMemcpyHtoD(d_kernelRadius, h_kernelRadius, Sizeof.INT);
    JCudaDriver.cuMemcpyHtoD(d_kernelHeight, h_kernelHeight, Sizeof.INT);
    JCudaDriver.cuMemcpyHtoD(d_nPoints, h_nPoints, Sizeof.INT);
    JCudaDriver.cuMemcpyHtoD(d_nInitial, h_nInitial, Sizeof.INT);
    JCudaDriver.cuMemcpyHtoD(d_nStep, h_nStep, Sizeof.INT);

    Pointer h_cache = Pointer.to(cache.getCache());
    Pointer h_bandwidth = Pointer.to(bandwidth);
    Pointer h_kernelPointers = Pointer.to(Kernel.getKernelPointer());
    Pointer h_nullMean = Pointer.to(nullMean);
    Pointer h_nullStd = Pointer.to(nullStd);

    CUdeviceptr d_cache = new CUdeviceptr();
    CUdeviceptr d_bandwidth = new CUdeviceptr();
    CUdeviceptr d_kernelPointers = new CUdeviceptr();
    CUdeviceptr d_nullMean = new CUdeviceptr();
    CUdeviceptr d_nullStd = new CUdeviceptr();

    JCudaDriver.cuMemAlloc(d_cache, Sizeof.FLOAT*nPixelsInCache);
    JCudaDriver.cuMemAlloc(d_bandwidth, Sizeof.FLOAT*nPixelsInImage);
    JCudaDriver.cuMemAlloc(d_kernelPointers, Sizeof.INT*2*Kernel.getKHeight());
    JCudaDriver.cuMemAlloc(d_nullMean, Sizeof.FLOAT*nPixelsInImage);
    JCudaDriver.cuMemAlloc(d_nullStd, Sizeof.FLOAT*nPixelsInImage);

    JCudaDriver.cuMemcpyHtoD(d_cache, h_cache, Sizeof.FLOAT*nPixelsInCache);
    JCudaDriver.cuMemcpyHtoD(d_bandwidth, h_bandwidth, Sizeof.FLOAT*nPixelsInImage);
    JCudaDriver.cuMemcpyHtoD(d_kernelPointers, h_kernelPointers, Sizeof.INT*2*Kernel.getKHeight());
    JCudaDriver.cuMemcpyHtoD(d_nullMean, h_nullMean, Sizeof.FLOAT*nPixelsInImage);
    JCudaDriver.cuMemcpyHtoD(d_nullStd, h_nullStd, Sizeof.FLOAT*nPixelsInImage);

    Pointer kernelParameters = Pointer.to(
        Pointer.to(d_cache),
        Pointer.to(d_bandwidth),
        Pointer.to(d_kernelPointers),
        Pointer.to(d_nullMean),
        Pointer.to(d_nullStd)
    );

    int blockDimX = 5;
    int blockDimY = 5;
    int nBlockX = (roiWidth[0] + blockDimX - 1) / blockDimX;
    int nBlockY = (roiHeight[0] + blockDimY - 1) / blockDimY;
    JCudaDriver.cuLaunchKernel(kernel, nBlockX, nBlockY, 1, blockDimX, blockDimY, 1, 0, null,
        kernelParameters, null);

    JCudaDriver.cuMemcpyDtoH(h_nullMean, d_nullMean, Sizeof.FLOAT*nPixelsInImage);
    JCudaDriver.cuMemcpyDtoH(h_nullStd, d_nullStd, Sizeof.FLOAT*nPixelsInImage);

    JCudaDriver.cuMemFree(d_cache);
    JCudaDriver.cuMemFree(d_bandwidth);
    JCudaDriver.cuMemFree(d_kernelPointers);
    JCudaDriver.cuMemFree(d_nullMean);
    JCudaDriver.cuMemFree(d_nullStd);
    JCuda.cudaDeviceReset();

    int imagePointer;
    for (int y=0; y<roiHeight[0]; y++) {
      for (int x=0; x<roiWidth[0]; x++) {
        imagePointer = (roiY[0]+y)*imageWidth[0] + roiX[0] + x;
        pixels[imagePointer] -= nullMean[imagePointer];
        pixels[imagePointer] /= nullStd[imagePointer];
      }
    }

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
