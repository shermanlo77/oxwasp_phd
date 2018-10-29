
import ij.*;
import ij.gui.GenericDialog;
import ij.gui.DialogListener;
import ij.gui.Roi;
import ij.process.*;
import ij.plugin.ContrastEnhancer;
import ij.plugin.filter.ExtendedPlugInFilter;
import ij.plugin.filter.PlugInFilterRunner;
import ij.plugin.filter.RankFilters;

import java.awt.*;
import java.awt.event.*;
import java.util.Arrays;
import java.util.Iterator;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;

/** This plugin implements the Mean, Minimum, Maximum, Variance, Median, Open Maxima, Close Maxima,
 *  Remove Outliers, Remove NaNs and Despeckle commands.
 */
 // Version 2012-07-15 M. Schmid: Fixes a bug that could cause preview not to work correctly
 // Version 2012-12-23 M. Schmid: Test for inverted LUT only once (not in each slice)
 // Version 2014-10-10 M. Schmid:   Fixes a bug that caused Threshold=0 when calling from API

public class EmpiricalNullFilter implements ExtendedPlugInFilter, DialogListener {
  
  //STATIC VARIABLES
  public static final int NULL_MEAN = 1, NULL_STD = 2, STD = 4, Q1 = 8, Q2 = 16, Q3 = 32;
  public static final int N_IMAGE_OUTPUT = 6;
  public static final String[] OUTPUT_NAME = {"null mean", "null std", "standard deviation",
      "quantile 1", "median", "quantile 3"};
  
  //array of float processors which contains images (or statistics) which are obtained from the
  //filter itself
  //entry 0: empiricial null mean
  //entry 1: empirical null std
  protected int outputImagePointer = NULL_MEAN + NULL_STD;
  protected FloatProcessor [] outputImageArray =
      new FloatProcessor[EmpiricalNullFilter.N_IMAGE_OUTPUT];
  
  // Filter parameters
  private double radius = 20;
  // Remember filter parameters for the next time
  private static double lastRadius; //separate for each filter type
  //
  // F u r t h e r   c l a s s   v a r i a b l e s
  private int flags = DOES_32;
  private ImageProcessor imageProcessor;
  private int nPasses = 1;      // The number of passes (color channels * stack slices)
  private PlugInFilterRunner pfr;
  private int pass;
  
  // M u l t i t h r e a d i n g - r e l a t e d
  private int numThreads = Prefs.getThreads();
  // Current state of processing is in class variables. Thus, stack parallelization must be done
  // ONLY with one thread for the image (not using these class variables):
  private int highestYinCache;    // the highest line read into the cache so far
  private boolean threadWaiting;    // a thread waits until it may read data
  private boolean copyingToCache;   // whether a thread is currently copying data to the cache
  
  /** Setup of the PlugInFilter. Returns the flags specifying the capabilities and needs
   * of the filter.
   *
   * @param arg Defines type of filter operation
   * @param imp The ImagePlus to be processed
   * @return    Flags specifying further action of the PlugInFilterRunner
   */
  public int setup(String arg, ImagePlus ip) {
    return flags;
  }

  public int showDialog(ImagePlus imp, String command, PlugInFilterRunner pfr) {
    
    GenericDialog genericDialog = new GenericDialog(command+"...");
    genericDialog.addNumericField("Radius", this.radius, 1, 6, "pixels");
    
    for (int i=0; i<N_IMAGE_OUTPUT; i++) {
      boolean defaultBoolean = (this.outputImagePointer >> i) % 2 == 1;
      genericDialog.addCheckbox("Show "+OUTPUT_NAME[i], defaultBoolean);
    }
    genericDialog.addDialogListener(this);
    genericDialog.showDialog();
    if (genericDialog.wasCanceled()) {
      return DONE;
    }
    IJ.register(this.getClass());
    
    this.pfr = pfr;
    
    return flags;
  }

  public boolean dialogItemChanged(GenericDialog gd, AWTEvent e) {
    this.setRadius(gd.getNextNumber());
    if (gd.invalidNumber() || this.radius < 0) {
      return false;
    }
    this.outputImagePointer = 0;
    for (int i=0; i<N_IMAGE_OUTPUT; i++) {
      boolean value = gd.getNextBoolean();
      if (value) {
        int pointer = 1;
        pointer  <<= i;
        this.outputImagePointer += pointer;
      }
    }    
    return true;
  }
  
  public void setOutputImage(int pointer) {
    this.outputImagePointer = pointer;
  }

  public void run(ImageProcessor ip) {
    this.imageProcessor = ip;
    this.rank();
    if (IJ.escapePressed()) {                 // interrupted by user?
      ip.reset();
    }
    for (int i=0; i<N_IMAGE_OUTPUT; i++) {
      if ( (outputImagePointer >> i) % 2 == 1) {
        ImagePlus output;
        output = new ImagePlus(OUTPUT_NAME[i], this.outputImageArray[i]);
        output.show();
      }
    }
  }
  
  public double getRadius(double radius) {
    return this.radius;
  }
  
  public void setRadius(double radius) {
    this.radius = radius;
  }
  
  /**METHOD: GET OUTPUT IMAGE
   * Returns an array of pixels from one of the requested output images
   * @param outputImagePointer e.g. NULL_MEAN, NULL_STD
   * @return float array containing the value of each pixel of a requested output image
   */
  public float [] getOutputImage(int outputImagePointer) {
    //for each output image
    for (int i=0; i<EmpiricalNullFilter.N_IMAGE_OUTPUT; i++) {
      //if the user requested this output image, return it
      if ( (outputImagePointer >> i) % 2 == 1) {
        return (float [] ) this.outputImageArray[i].getPixels();
      }
    }
    return null;
  }
  
  /** Filters an image by any method except 'despecle' or 'remove outliers'.
   *  @param ip    The ImageProcessor that should be filtered (all 4 types supported)
   *  @param radius  Determines the kernel size, see Process>Filters>Show Circular Masks.
   *           Must not be negative. No checking is done for large values that would
   *           lead to excessive computing times.
   *  @param filterType May be MEAN, MIN, MAX, VARIANCE, or MEDIAN.
   */
  /** Filters an image by any method except 'despecle' (for 'despeckle', use 'median' and radius=1)
   * @param ip The image subject to filtering
   * @param radius The kernel radius
   * @param filterType as defined above; DESPECKLE is not a valid type here; use median and
   *      a radius of 1.0 instead
   * @param whichOutliers BRIGHT_OUTLIERS or DARK_OUTLIERS for 'outliers' filter
   * @param threshold Threshold for 'outliers' filter
   */
  public void rank() {
    Rectangle roi = this.imageProcessor.getRoi();
    int[] lineRadii = this.makeLineRadii(this.radius);
    
    //instantiate new images for each outout
    for (int i=0; i<EmpiricalNullFilter.N_IMAGE_OUTPUT; i++) {
      if ( (this.outputImagePointer >> i) % 2 == 1) {
        this.outputImageArray[i] =
            new FloatProcessor(this.imageProcessor.getWidth(), this.imageProcessor.getHeight());
      }
    }
    
    boolean[] aborted = new boolean[1];           // returns whether interrupted during preview or ESC pressed
    
    // Filter a grayscale image or one channel of an RGB image with several threads
    // Implementation: each thread uses the same input buffer (cache), always works on the next unfiltered line
    // Usually, one thread reads reads several lines into the cache, while the others are processing the data.
    // 'aborted[0]' is set if the main thread has been interrupted (during preview) or ESC pressed.
    // 'aborted' must not be a class variable because it signals the other threads to stop; and this may be caused
    // by an interrupted preview thread after the main calculation has been started.
    
    int numThreads = Math.min(roi.height, this.numThreads);
    if (numThreads==0)
      return;
    
    int kHeight = kHeight(lineRadii);
    int kRadius  = kRadius(lineRadii);
    final int cacheWidth = roi.width+2*kRadius;
    final int cacheHeight = kHeight + (numThreads>1 ? 2*numThreads : 0);
    // 'cache' is the input buffer. Each line y in the image is mapped onto cache line y%cacheHeight
    final float[] cache = new float[cacheWidth*cacheHeight];
    this.highestYinCache = Math.max(roi.y-kHeight/2, 0) - 1; //this line+1 will be read into the cache first
    
    ImageProcessor imageProcessor = this.imageProcessor;
    
    final int[] yForThread = new int[numThreads];   //threads announce here which line they currently process
    Arrays.fill(yForThread, -1);
    yForThread[numThreads-1] = roi.y-1;         //first thread started should begin at roi.y
    //IJ.log("going to filter lines "+roi.y+"-"+(roi.y+roi.height-1)+"; cacheHeight="+cacheHeight);
    final Thread[] threads = new Thread[numThreads-1];  //thread number 0 is this one, not in the array
    for (int t=numThreads-1; t>0; t--) {
      final int ti=t;
      final Thread thread = new Thread(
          new Runnable() {
            final public void run() {
              doFiltering(imageProcessor, lineRadii, cache, cacheWidth, cacheHeight, yForThread, ti, aborted);
            }
          },
      "RankFilters-"+t);
      thread.setPriority(Thread.currentThread().getPriority());
      thread.start();
      threads[ti-1] = thread;
    }

    doFiltering(imageProcessor, lineRadii, cache, cacheWidth, cacheHeight, yForThread, 0, aborted);
    for (final Thread thread : threads)
      try {
          if (thread != null) thread.join();
      } catch (InterruptedException e) {
        aborted[0] = true;
        Thread.currentThread().interrupt();   //keep interrupted status (PlugInFilterRunner needs it)
      }
    this.showProgress(1.0);
    pass++;
  }

  // Filter a grayscale image or one channel of an RGB image using one thread
  //
  // Synchronization: unless a thread is waiting, we avoid the overhead of 'synchronized'
  // statements. That's because a thread waiting for another one should be rare.
  //
  // Data handling: The area needed for processing a line is written into the array 'cache'.
  // This is a stripe of sufficient width for all threads to have each thread processing one
  // line, and some extra space if one thread is finished to start the next line.
  // This array is padded at the edges of the image so that a surrounding with radius kRadius
  // for each pixel processed is within 'cache'. Out-of-image
  // pixels are set to the value of the nearest edge pixel. When adding a new line, the lines in
  // 'cache' are not shifted but rather the smaller array with the start and end pointers of the
  // kernel area is modified to point at the addresses for the next line.
  //
  // Algorithm: For mean and variance, except for very small radius, usually do not calculate the
  // sum over all pixels. This sum is calculated for the first pixel of every line only. For the
  // following pixels, add the new values and subtract those that are not in the sum any more.
  // For min/max, also first look at the new values, use their maximum if larger than the old
  // one. The look at the values not in the area any more; if it does not contain the old
  // maximum, leave the maximum unchanged. Otherwise, determine the maximum inside the area.
  // For outliers, calculate the median only if the pixel deviates by more than the threshold
  // from any pixel in the area. Therfore min or max is calculated; this is a much faster
  // operation than the median.
  private void doFiltering(ImageProcessor ip, int[] lineRadii, float[] cache, int cacheWidth, int cacheHeight,
      int [] yForThread, int threadNumber, boolean[] aborted) {
    if (aborted[0] || Thread.currentThread().isInterrupted()) return;
    int width = ip.getWidth();
    int height = ip.getHeight();
    Rectangle roi = ip.getRoi();

    int kHeight = kHeight(lineRadii);
    int kRadius  = kRadius(lineRadii);
    int kNPoints = kNPoints(lineRadii);

    int xmin = roi.x - kRadius;
    int xmax = roi.x + roi.width + kRadius;
    int[]cachePointers = makeCachePointers(lineRadii, cacheWidth);

    int padLeft = xmin<0 ? -xmin : 0;
    int padRight = xmax>width? xmax-width : 0;
    int xminInside = xmin>0 ? xmin : 0;
    int xmaxInside = xmax<width ? xmax : width;
    int widthInside = xmaxInside - xminInside;
    
    double[] sums = new double[2];
    float[] medianBuf = new float[kNPoints];
    float[] medianBuf1 = new float[kNPoints];
    float [] quantiles = new float[3];
    
    boolean smallKernel = kRadius < 2;
    
    //values is a 2D array
    //each dimension contain the pixel values for each image
    //dim 1:
      //0. filtered image
      //1. 2. 3. ... are the output images
    float[][] values = new float[EmpiricalNullFilter.N_IMAGE_OUTPUT+1][];
    Object pixels = ip.getPixels();
    //get the pixels of the filtered image
    values[0] = (float[]) pixels;
    //get the pixels of each of the output images (only if requested)
    for (int i=0; i<EmpiricalNullFilter.N_IMAGE_OUTPUT; i++) {
      if ( (this.outputImagePointer >> i) % 2 == 1) {
        values[i+1] = (float[]) this.outputImageArray[i].getPixels();
      }
    }
    
    int numThreads = yForThread.length;
    long lastTime = System.currentTimeMillis();
    int previousY = kHeight/2-cacheHeight;
    
    while (!aborted[0]) {
      int y = arrayMax(yForThread) + 1;   // y of the next line that needs processing
      yForThread[threadNumber] = y;
      //IJ.log("thread "+threadNumber+" @y="+y+" needs"+(y-kHeight/2)+"-"+(y+kHeight/2)+" highestYinC="+highestYinCache);
      boolean threadFinished = y >= roi.y+roi.height;
      if (numThreads>1 && (threadWaiting || threadFinished))    // 'if' is not synchronized to avoid overhead
        synchronized(this) {
          notifyAll();          // we may have blocked another thread
          //IJ.log("thread "+threadNumber+" @y="+y+" notifying");
        }
      if (threadFinished)
        return;               // all done, break the loop

      if (threadNumber==0) {          // main thread checks for abort and ProgressBar
        long time = System.currentTimeMillis();
        if (time-lastTime>100) {
          lastTime = time;
          this.showProgress((y-roi.y)/(double)(roi.height));
          if (Thread.currentThread().isInterrupted() || (this.imageProcessor!= null && IJ.escapePressed())) {
            aborted[0] = true;
            synchronized(this) {notifyAll();}
            return;
          }
        }
      }

      for (int i=0; i<cachePointers.length; i++)  //shift kernel pointers to new line
        cachePointers[i] = (cachePointers[i] + cacheWidth*(y-previousY))%cache.length;
      previousY = y;

      if (numThreads>1) {             // thread synchronization
        int slowestThreadY = arrayMinNonNegative(yForThread); // non-synchronized check to avoid overhead
        if (y - slowestThreadY + kHeight > cacheHeight) { // we would overwrite data needed by another thread
          synchronized(this) {
            slowestThreadY = arrayMinNonNegative(yForThread); //recheck whether we have to wait
            if (y - slowestThreadY + kHeight > cacheHeight) {
              do {
                notifyAll();      // avoid deadlock: wake up others waiting
                threadWaiting = true;
                //IJ.log("Thread "+threadNumber+" waiting @y="+y+" slowest@y="+slowestThreadY);
                try {
                  wait();
                  if (aborted[0]) return;
                } catch (InterruptedException e) {
                  aborted[0] = true;
                  notifyAll();
                  Thread.currentThread().interrupt(); //keep interrupted status (PlugInFilterRunner needs it)
                  return;
                }
                slowestThreadY = arrayMinNonNegative(yForThread);
              } while (y - slowestThreadY + kHeight > cacheHeight);
            } //if
            threadWaiting = false;
          }
        }
      }

      if (numThreads==1) {                              // R E A D
        int yStartReading = y==roi.y ? Math.max(roi.y-kHeight/2, 0) : y+kHeight/2;
        for (int yNew = yStartReading; yNew<=y+kHeight/2; yNew++) { //only 1 line except at start
          readLineToCacheOrPad(pixels, width, height, roi.y, xminInside, widthInside,
              cache, cacheWidth, cacheHeight, padLeft, padRight, kHeight, yNew);
        }
      } else {
        if (!copyingToCache || highestYinCache < y+kHeight/2) synchronized(cache) {
          copyingToCache = true;        // copy new line(s) into cache
          while (highestYinCache < arrayMinNonNegative(yForThread) - kHeight/2 + cacheHeight - 1) {
            int yNew = highestYinCache + 1;
            readLineToCacheOrPad(pixels, width, height, roi.y, xminInside, widthInside,
              cache, cacheWidth, cacheHeight, padLeft, padRight, kHeight, yNew);
            highestYinCache = yNew;
          }
          copyingToCache = false;
        }
      }
      
      int cacheLineP = cacheWidth * (y % cacheHeight) + kRadius;  //points to pixel (roi.x, y)
      this.filterLine(values, width, cache, cachePointers, kNPoints, cacheLineP, roi, y, // F I L T E R
          sums, medianBuf, medianBuf1, quantiles, smallKernel);
      //System.out.println("thread "+threadNumber+" @y="+y+" line done");
    } // while (!aborted[0]); loop over y (lines)
  }

  private int arrayMax(int[] array) {
    int max = Integer.MIN_VALUE;
    for (int i=0; i<array.length; i++)
      if (array[i] > max) max = array[i];
    return max;
  }

  //returns the minimum of the array, but not less than 0
  private int arrayMinNonNegative(int[] array) {
    int min = Integer.MAX_VALUE;
    for (int i=0; i<array.length; i++)
      if (array[i]<min) min = array[i];
    return min<0 ? 0 : min;
  }

  private void filterLine(float[][] values, int width, float[] cache, int[] cachePointers, int kNPoints, int cacheLineP, Rectangle roi, int y,
      double[] sums, float[] medianBuf, float[] medianBuf1, float[] quantiles, boolean smallKernel) {
      int valuesP = roi.x+y*width;
      boolean fullCalculation = true;
      float std; //standard deviation
      int nData = 0; //number of non-nan data
      Percentile percentile = new Percentile();
      float initialValue = 0;
      
      //set required variables
      NormalDistribution normal = new NormalDistribution();
      MersenneTwister rng = new MersenneTwister(System.currentTimeMillis());
      //then for each pixel in this line
      for (int x=0; x<roi.width; x++, valuesP++) { // x is with respect to roi.x
        
        //set the first value to be the median
        getQuantiles(cache, x, cachePointers, medianBuf, medianBuf1, kNPoints, cache[cacheLineP], quantiles, percentile);
        if (x==0) {
          initialValue = quantiles[1];
        }
        
        if (fullCalculation) {
          //for small kernel, always use the full area, not incremental algorithm
          fullCalculation = smallKernel;
          nData = getAreaSums(cache, x, cachePointers, sums);
        } else {
          nData = addSideSums(cache, x, cachePointers, sums, nData);
          //avoid perpetuating NaNs into remaining line
          if (Double.isNaN(sums[0])) {
            fullCalculation = true;
          }
        }
        
        if (nData != 0) {
          std = (float) Math.sqrt(((sums[1] - sums[0]*sums[0]/nData)/(nData-1)));
        } else {
          throw new RuntimeException("no non-NaN data at line "+y+" column "+x);
        }
        
        EmpiricalNull empiricalNull = new EmpiricalNull(cache, x, cachePointers , initialValue, quantiles,
            std, nData, normal, rng);
        empiricalNull.estimateNull();
        values[0][valuesP] = (cache[cacheLineP+x] - empiricalNull.nullMean) / empiricalNull.nullStd;
        initialValue = empiricalNull.nullMean;
        for (int i=0; i<EmpiricalNullFilter.N_IMAGE_OUTPUT; i++) {
          if ( (this.outputImagePointer >> i) % 2 == 1) {
            switch (i) {
              case 0:
                values[1][valuesP] = empiricalNull.nullMean;
                break;
              case 1:
                values[2][valuesP] = empiricalNull.nullStd;
                break;
              case 2:
                values[3][valuesP] = std;
                break;
              case 3:
                values[4][valuesP] = quantiles[0];
                break;
              case 4:
                values[5][valuesP] = quantiles[1];
                break;
              case 5:
                values[6][valuesP] = quantiles[2];
                break;
            }
          }
        }
      }
    }

  /** Read a line into the cache (including padding in x).
   *  If y>=height, instead of reading new data, it duplicates the line y=height-1.
   *  If y==0, it also creates the data for y<0, as far as necessary, thus filling the cache with
   *  more than one line (padding by duplicating the y=0 row).
   */
  //EDIT: Padding contains NaN
  private static void readLineToCacheOrPad(Object pixels, int width, int height, int roiY, int xminInside, int widthInside,
      float[]cache, int cacheWidth, int cacheHeight, int padLeft, int padRight, int kHeight, int y) {
    int lineInCache = y%cacheHeight;
    if (y < height) {
      readLineToCache(pixels, y*width, xminInside, widthInside, cache, lineInCache*cacheWidth, padLeft, padRight);
      if (y==0) for (int prevY = roiY-kHeight/2; prevY<0; prevY++) {  //for y<0, pad with y=0 border pixels
        int prevLineInCache = cacheHeight+prevY;
        //EDIT: Padding contains NaN
        Arrays.fill(cache, prevLineInCache*cacheWidth, prevLineInCache*cacheWidth + cacheWidth,
            Float.NaN);
      }
    } else
      //EDIT: Padding contains NaN
      Arrays.fill(cache, lineInCache*cacheWidth, lineInCache*cacheWidth + cacheWidth, Float.NaN);
  }

  /** Read a line into the cache (includes conversion to flaot). Pad with edge pixels in x if necessary */
  //EDIT: Padding contains NaN
  private static void readLineToCache(Object pixels, int pixelLineP, int xminInside, int widthInside,
                float[] cache, int cacheLineP, int padLeft, int padRight) {
    System.arraycopy(pixels, pixelLineP+xminInside, cache, cacheLineP+padLeft, widthInside);
    //EDIT: Padding contains NaN
    for (int cp=cacheLineP; cp<cacheLineP+padLeft; cp++)
      cache[cp] = Float.NaN;
    for (int cp=cacheLineP+padLeft+widthInside; cp<cacheLineP+padLeft+widthInside+padRight; cp++)
      cache[cp] = Float.NaN;
  }
  
  /** Get sum of values and values squared within the kernel area.
   *  x between 0 and cacheWidth-1
   *  Output is written to array sums[0] = sum; sums[1] = sum of squares */
  private static int getAreaSums(float[] cache, int xCache0, int[] kernel, double[] sums) {
    double sum=0, sum2=0;
    int nData = 0;
    for (int kk=0; kk<kernel.length; kk++) {  // y within the cache stripe (we have 2 kernel pointers per cache line)
      for (int p=kernel[kk++]+xCache0; p<=kernel[kk]+xCache0; p++) {
        double v = cache[p];
        if (!Double.isNaN(v)) {
          sum += v;
          sum2 += v*v;
          nData++;
        }
      }
    }
    sums[0] = sum;
    sums[1] = sum2;
    
    return nData;
  }

  /** Add all values and values squared at the right border inside minus at the left border outside the kernal area.
   *  Output is added or subtracted to/from array sums[0] += sum; sums[1] += sum of squares  when at
   *  the right border, minus when at the left border */
  private static int addSideSums(float[] cache, int xCache0, int[] kernel, double[] sums, int nData) {
    double sum=0, sum2=0;
    for (int kk=0; kk<kernel.length; /*k++;k++ below*/) {
      
      double v = cache[kernel[kk++]+(xCache0-1)]; //this value is not in the kernel area any more
      if (!Double.isNaN(v)) {
        sum -= v;
        sum2 -= v*v;
        nData--;
      }
      
      v = cache[kernel[kk++]+xCache0];            //this value comes into the kernel area
      if (!Double.isNaN(v)) {
        sum += v;
        sum2 += v*v;
        nData++;
      }
      
    }
    sums[0] += sum;
    sums[1] += sum2;
    return nData;
  }
  
  /** Get median of values within kernel-sized neighborhood.
   *  NaN data values are ignored; the output is NaN only if there are only NaN values in the
   *  kernel-sized neighborhood */
  private static void getQuantiles(float[] cache, int xCache0, int[] kernel,
      float[] belowBuf, float[] aboveBuf, int kNPoints, float guess, float[] quantiles, Percentile percentile) {
    int nAbove = 0, nBelow = 0;
    for (int kk=0; kk<kernel.length; kk++) {
      for (int p=kernel[kk++]+xCache0; p<=kernel[kk]+xCache0; p++) {
        float v = cache[p];
        if (Float.isNaN(v)) {
          kNPoints--;
        } else if (v > guess) {
          aboveBuf[nAbove] = v;
          nAbove++;
        }
        else if (v < guess) {
          belowBuf[nBelow] = v;
          nBelow++;
        }
      }
    }
    
    for (int i=0; i<3; i++) {
      int index = (int) Math.floor( ((double)((i+1) * kNPoints)) / 4.0 );
      if (kNPoints == 0) {
        quantiles[i] = Float.NaN;  //only NaN data in the neighborhood?
      } else if (index<nBelow) {
        quantiles[i] =  findNthLowestNumber(belowBuf, nBelow, index);
      } else if ((kNPoints-index)<nAbove) {
        quantiles[i] =  findNthLowestNumber(aboveBuf, nAbove, index - nBelow  - 1);
      } else {
        quantiles[i] =  guess;
      }
    }
  }

  /** Find the n-th lowest number in part of an array
   *  @param buf The input array. Only values 0 ... bufLength are read. <code>buf</code> will be modified.
   *  @param bufLength Number of values in <code>buf</code> that should be read
   *  @param n which value should be found; n=0 for the lowest, n=bufLength-1 for the highest
   *  @return the value */
  public final static float findNthLowestNumber(float[] buf, int bufLength, int n) {
    // Hoare's find, algorithm, based on http://www.geocities.com/zabrodskyvlada/3alg.html
    // Contributed by Heinz Klar
    int i,j;
    int l=0;
    int m=bufLength-1;
    float med=buf[n];
    float dum ;

    while (l<m) {
      i=l ;
      j=m ;
      do {
        while (buf[i]<med) i++ ;
        while (med<buf[j]) j-- ;
        dum=buf[j];
        buf[j]=buf[i];
        buf[i]=dum;
        i++ ; j-- ;
      } while ((j>=n) && (i<=n)) ;
      if (j<n) l=i ;
      if (n<i) m=j ;
      med=buf[n] ;
    }
  return med ;
  }
  
  /** Create a circular kernel (structuring element) of a given radius.
   *  @param radius
   *  Radius = 0.5 includes the 4 neighbors of the pixel in the center,
   *  radius = 1 corresponds to a 3x3 kernel size.
   *  @return the circular kernel
   *  The output is an array that gives the length of each line of the structuring element
   *  (kernel) to the left (negative) and to the right (positive):
   *  [0] left in line 0, [1] right in line 0,
   *  [2] left in line 2, ...
   *  The maximum (absolute) value should be kernelRadius.
   *  Array elements at the end:
   *  length-2: nPoints, number of pixels in the kernel area
   *  length-1: kernelRadius in x direction (kernel width is 2*kernelRadius+1)
   *  Kernel height can be calculated as (array length - 1)/2 (odd number);
   *  Kernel radius in y direction is kernel height/2 (truncating integer division).
   *  Note that kernel width and height are the same for the circular kernels used here,
   *  but treated separately for the case of future extensions with non-circular kernels.
   */
  protected int[] makeLineRadii(double radius) {
    if (radius>=1.5 && radius<1.75) //this code creates the same sizes as the previous RankFilters
      radius = 1.75;
    else if (radius>=2.5 && radius<2.85)
      radius = 2.85;
    int r2 = (int) (radius*radius) + 1;
    int kRadius = (int)(Math.sqrt(r2+1e-10));
    int kHeight = 2*kRadius + 1;
    int[] kernel = new int[2*kHeight + 2];
    kernel[2*kRadius] = -kRadius;
    kernel[2*kRadius+1] =  kRadius;
    int nPoints = 2*kRadius+1;
    for (int y=1; y<=kRadius; y++) {    //lines above and below center together
      int dx = (int)(Math.sqrt(r2-y*y+1e-10));
      kernel[2*(kRadius-y)] = -dx;
      kernel[2*(kRadius-y)+1] =  dx;
      kernel[2*(kRadius+y)] = -dx;
      kernel[2*(kRadius+y)+1] =  dx;
      nPoints += 4*dx+2;  //2*dx+1 for each line, above&below
    }
    kernel[kernel.length-2] = nPoints;
    kernel[kernel.length-1] = kRadius;
    //for (int i=0; i<kHeight;i++)IJ.log(i+": "+kernel[2*i]+"-"+kernel[2*i+1]);
    return kernel;
  }

  //kernel height
  private int kHeight(int[] lineRadii) {
    return (lineRadii.length-2)/2;
  }

  //kernel radius in x direction. width is 2+kRadius+1
  private int kRadius(int[] lineRadii) {
    return lineRadii[lineRadii.length-1];
  }

  //number of points in kernal area
  private int kNPoints(int[] lineRadii) {
    return lineRadii[lineRadii.length-2];
  }

  //cache pointers for a given kernel
  protected int[] makeCachePointers(int[] lineRadii, int cacheWidth) {
    int kRadius = kRadius(lineRadii);
    int kHeight = kHeight(lineRadii);
    int[] cachePointers = new int[2*kHeight];
    for (int i=0; i<kHeight; i++) {
      cachePointers[2*i]   = i*cacheWidth+kRadius + lineRadii[2*i];
      cachePointers[2*i+1] = i*cacheWidth+kRadius + lineRadii[2*i+1];
    }
    return cachePointers;
  }

  void showMasks() {
    int w=150, h=150;
    ImageStack stack = new ImageStack(w, h);
    //for (double r=0.1; r<3; r+=0.01) {
    for (double r=0.5; r<50; r+=0.5) {
      ImageProcessor ip = new FloatProcessor(w,h,new int[w*h]);
      float[] pixels = (float[])ip.getPixels();
      int[] lineRadii = makeLineRadii(r);
      int kHeight = kHeight(lineRadii);
      int kRadius = kRadius(lineRadii);
      int y0 = h/2-kHeight/2;
      for (int i = 0, y = y0; i<kHeight; i++, y++)
        for (int x = w/2+lineRadii[2*i], p = x+y*w; x <= w/2+lineRadii[2*i+1]; x++, p++)
          pixels[p] = 1f;
      stack.addSlice("radius="+r+", size="+(2*kRadius+1), ip);
    }
    new ImagePlus("Masks", stack).show();
  }

  /** This method is called by ImageJ to set the number of calls to run(ip)
   *  corresponding to 100% of the progress bar */
  public void setNPasses (int nPasses) {
    this.nPasses = nPasses;
    pass = 0;
  }

  private void showProgress(double percent) {
    int nPasses2 = nPasses;
    percent = (double)pass/nPasses2 + percent/nPasses2;
    IJ.showProgress(percent);
  }
}
