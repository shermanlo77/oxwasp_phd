/**EDITED RankFilters.java
 * See https://github.com/imagej/ImageJA/blob/master/src/main/java/ij/plugin/filter/RankFilters.java
 */
/** This plugin implements the Mean, Minimum, Maximum, Variance, Median, Open Maxima, Close Maxima,
 *  Remove Outliers, Remove NaNs and Despeckle commands.
 */
 // Version 2012-07-15 M. Schmid: Fixes a bug that could cause preview not to work correctly
 // Version 2012-12-23 M. Schmid: Test for inverted LUT only once (not in each slice)
 // Version 2014-10-10 M. Schmid:   Fixes a bug that caused Threshold=0 when calling from API

package uk.ac.warwick.sip.empiricalnullfilter;

import ij.gui.DialogListener;
import ij.gui.GenericDialog;
import ij.gui.Roi;
import ij.io.Opener;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.Macro;
import ij.plugin.filter.ExtendedPlugInFilter;
import ij.plugin.filter.PlugInFilterRunner;
import ij.Prefs;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import java.awt.AWTEvent;
import java.awt.Rectangle;
import java.io.IOException;
import java.util.Arrays;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.omg.DynamicAny.DynAnyPackage.InvalidValue;

/**CLASS: EMPIRICAL NULL FILTER
 * Implementation of the empirical null filter
 * @author Sherman Ip
 */
public class EmpiricalNullFilter implements ExtendedPlugInFilter, DialogListener {
  
  //STATIC FINAL VARIABLES
  
  //the filter can output the std and quantiles filter
  //the static int are options for what output images to show
  //to combine the options, either add them together or use the OR operator
  public static final int NULL_MEAN = 1, NULL_STD = 2, STD = 4, Q1 = 8, Q2 = 16, Q3 = 32;
  public static final int N_IMAGE_OUTPUT = 6; //number of output images which can be shown
  //name of each output image
  public static final String[] OUTPUT_NAME = {"null mean", "null std", "standard deviation",
      "quantile 1", "median", "quantile 3"};
  //this filter only works on 32-bit images, this is indicated in FLAGS
  private static final int FLAGS = DOES_32;
  
  //STATIC VARIABLES
  private static double lastRadius = 0;
  private static int lastOutputImagePointer = -1;
  
  //MEMBER VARIABLES
  
  //which output images to show
  private int outputImagePointer = NULL_MEAN + NULL_STD;
  //array of float processors which contains images (or statistics) which are obtained from the
  //filter itself, eg null mean, null std, std, q1, q2, q3
  private FloatProcessor [] outputImageArray = new FloatProcessor[N_IMAGE_OUTPUT];
  private double radius = 20; //radius of the kernel
  
  private ImageProcessor imageProcessor; //the image to be filtered
  private Roi roi; //region of interest
  //used by showDialog, unused but needed in case deleted by automatic garbage collection
  private PlugInFilterRunner pfr;
  private boolean isShowProgressBar = false;
  protected int nPasses = 1; // The number of passes (color channels * stack slices)
  protected int pass;
  
  //EMPIRICAL NULL RELATED
  private int nInitial = EmpiricalNull.N_INITIAL;
  private int nStep = EmpiricalNull.N_STEP;
  private float log10Tolerance = EmpiricalNull.LOG_10_TOLERANCE;
  private float bandwidthParameterA = EmpiricalNull.BANDWIDTH_PARAMETER_A;
  private float bandwidthParameterB = EmpiricalNull.BANDWIDTH_PARAMETER_B;
  
  //MULTITHREADING RELATED
  private int numThreads = Prefs.getThreads();
  // Current state of processing is in class variables. Thus, stack parallelization must be done
  // ONLY with one thread for the image (not using these class variables):
  private int highestYinCache; // the highest line read into the cache so far
  private boolean threadWaiting; // a thread waits until it may read data
  private boolean copyingToCache; // whether a thread is currently copying data to the cache
  
  /**CONSTRUCTOR
   * Empty constructor, used by ImageJ
   */
  public EmpiricalNullFilter() {
  }
  
  /**IMPLEMENTED: SETUP
   * Setup of the PlugInFilter. Returns the flags specifying the capabilities and needs
   * of the filter.
   * @param arg not used
   * @param imp not used
   * @return Flags specifying further action of the PlugInFilterRunner
   */
  @Override
  public int setup(String arg, ImagePlus ip) {
    //save the image processor
    this.imageProcessor = ip.getProcessor();
    //save the roi
    this.roi = ip.getRoi();
    //if there is no roi, then this.roi is null
    //instantiate a new rectangle roi where the roi is the whole image
    if (this.roi == null) {
      this.roi = new Roi(ip.getProcessor().getRoi());
    }
    return FLAGS;
  }
  
  /**IMPLEMENTED: RUN
   * For the use of ExtendedPlugInFilter
   * Do the filtering
   * @param ip image to be filtered
   */
  @Override
  public void run(ImageProcessor ip) {
    //save the image
    this.imageProcessor = ip;
    //do the filtering
    this.filter();
    //interrupted by user?
    if (IJ.escapePressed()) {
      ip.reset();
    }
    //show each requested output image
    for (int i=0; i<N_IMAGE_OUTPUT; i++) {
      if ( (outputImagePointer >> i) % 2 == 1) {
        ImagePlus output;
        output = new ImagePlus(OUTPUT_NAME[i], this.outputImageArray[i]);
        output.show();
      }
    }
  }
  
  /**IMPLEMENTED: SHOW DIALOG
   * Dialog box for setting the radius and which output images to show
   * @param imp image to apply the filter on
   * @param command name of this command
   * @param pfr
   */
  @Override
  public int showDialog(ImagePlus imp, String command, PlugInFilterRunner pfr) {
    
    GenericDialog genericDialog = new GenericDialog(command+"...");
    
    //set up radius field, use the last radius if it exists
    genericDialog.addMessage("Settings");
    if (lastRadius > 0) {
      this.radius = lastRadius;
    }
    genericDialog.addNumericField("Radius", this.radius, 1, 6, "pixels");
    
    //set up checkbox for each output image, use the last pointer if it exists
    genericDialog.addMessage("Output images");
    if (lastOutputImagePointer >= 0) {
      this.outputImagePointer = lastOutputImagePointer;
    }
    for (int i=0; i<N_IMAGE_OUTPUT; i++) {
      boolean defaultBoolean = (this.outputImagePointer >> i) % 2 == 1;
      genericDialog.addCheckbox("Show "+OUTPUT_NAME[i], defaultBoolean);
    }
    
    //add fields for the empirical null tuning parameters
    //integers do not show decimal points
    genericDialog.addMessage("Advanced options");
    genericDialog.addNumericField("number of initial values", this.getNInitial(),
        0, 6, null);
    genericDialog.addNumericField("number of steps", this.getNStep(),
        0, 6, null);
    genericDialog.addNumericField("log tolerance", this.getLog10Tolerance(),
        1, 6, null);
    genericDialog.addNumericField("bandwidth A", this.getBandwidthA(),
        1, 6, null);
    genericDialog.addNumericField("bandwidth B", this.getBandwidthB(),
        1, 6, null);
    
    //the DialogItemChanged method will be called on user input
    genericDialog.addDialogListener(this);
    genericDialog.showDialog(); //display the dialog
    if (genericDialog.wasCanceled()) {
      return DONE;
    }
    //protected static class variables (filter parameters) from garbage collection
    IJ.register(this.getClass());
    
    if (Macro.getOptions() == null) { //interactive only: remember radius entered
      lastRadius = this.radius;
      lastOutputImagePointer = this.outputImagePointer;
    }
    
    //save a copy of pfr
    this.pfr = pfr;
    
    return FLAGS;
  }
  
  /**IMPLEMENTED: DIALOG ITEM CHANGED
   * Called on user input
   * Update the radius and outputImagePointer
   */
  @Override
  public boolean dialogItemChanged(GenericDialog gd, AWTEvent e) {
    //get the radius and set it
    this.setRadius(gd.getNextNumber());
    if (gd.invalidNumber() || this.radius < 0) {
      return false;
    }
    //get the output image options and save it
    this.outputImagePointer = 0;
    for (int i=0; i<N_IMAGE_OUTPUT; i++) {
      boolean value = gd.getNextBoolean();
      if (value) {
        int pointer = 1;
        pointer  <<= i;
        this.outputImagePointer += pointer;
      }
    }
    //get the empirical null parameters and set it
    //if an exception is caught, return false to indicate an invalid item change
    try {
      this.setNInitial((int)gd.getNextNumber());
      this.setNStep((int)gd.getNextNumber());
      this.setLog10Tolerance((float)gd.getNextNumber());
      this.setBandwidthA((float)gd.getNextNumber());
      this.setBandwidthB((float)gd.getNextNumber());
    } catch (InvalidValue exception) {
      return false;
    }
    
    return true;
  }
  
  /**METHOD: GET RADIUS
   * @return the radius of the kernel
   */
  public double getRadius() {
    return this.radius;
  }
  
  /**METHOD: SET RADIUS
   * @param radius The radius of the kernel
   */
  public void setRadius(double radius) {
    this.radius = radius;
  }
  
  /**METHOD: GET FILTERED IMAGE
   * @return array of pixels of the filtered image
   */
  public float [] getFilteredImage() {
    return (float []) this.imageProcessor.getPixels();
  }
  
  /**METHOD: GET OUTPUT IMAGE
   * Returns an array of pixels from one of the requested output images
   * @param outputImagePointer e.g. NULL_MEAN, NULL_STD
   * @return float array containing the value of each pixel of a requested output image
   */
  public float [] getOutputImage(int outputImagePointer) {
    //for each output image
    for (int i=0; i<N_IMAGE_OUTPUT; i++) {
      //if the user requested this output image, return it
      if ( (outputImagePointer >> i) % 2 == 1) {
        return (float [] ) this.outputImageArray[i].getPixels();
      }
    }
    return null;
  }
  
  /**METHOD: SET OUTPUT IMAGE
   * Set outputImagePointer, this indicate which output images to show
   * @param pointer which output images to show, use the static int variable, e.g. NULL_MEAN
   */
  public void setOutputImage(int pointer) {
    this.outputImagePointer = pointer;
  }
  
  /**METHOD: FILTER
   * Call the method filter using the image passed in the parameter
   * @param image image to be filtered
   */
  public void filter(float [][] image) {
    this.imageProcessor = new FloatProcessor(image);
    this.roi = new Roi(this.imageProcessor.getRoi());
    this.filter();
  }
  
  /**METHOD: FILTER
   * Call the method filter using the image and ROI passed in the parameter
   * @param image image to be filtered
   * @param roiPath path to the roi file
   * @throws IOException throws exception if opener.openRoi fails and returns a null
   */
  public void filter(float [][] image, String roiPath) throws IOException {
    Opener opener = new Opener();
    Roi roi = opener.openRoi(roiPath);
    if (roi == null) {
      throw new IOException("Failed to load roi in "+roiPath);
    }
    this.imageProcessor = new FloatProcessor(image);
    this.imageProcessor.setRoi(roi);
    this.roi = roi;
    this.filter();
  }
  
  /**METHOD: FILTER
   * Do the empirical null filter using several threads
   * Implementation: each thread uses the same input buffer (cache), always works on the next
   * unfiltered line
   * Usually, one thread reads reads several lines into the cache,
   * while the others are processing the data.
   * 'aborted[0]' is set if the main thread has been interrupted (during preview) or ESC pressed.
   * 'aborted' must not be a class variable because it signals the other threads to stop;
   * and this may be caused by an interrupted preview thread after the main calculation has been
   * started.
   */
  public void filter() {
    
    //roi = region of interest
    Rectangle roiRectangle = this.imageProcessor.getRoi();
    //pointers which indicate the shape of the kernel
    final int[] lineRadii = this.makeLineRadii(this.radius);
    
    //instantiate new images for each outout
    for (int i=0; i<N_IMAGE_OUTPUT; i++) {
      if ( (this.outputImagePointer >> i) % 2 == 1) {
        FloatProcessor outputProcessor = (FloatProcessor) this.imageProcessor.duplicate();
        float [] pixels = (float []) outputProcessor.getPixels();
        Arrays.fill(pixels, Float.NaN);
        this.outputImageArray[i] = outputProcessor;
      }
    }
    
    //returns whether interrupted during preview or ESC pressed
    final boolean[] aborted = new boolean[1];
    
    //get the number of threads
    int numThreads = Math.min(roiRectangle.height, this.numThreads);
    if (numThreads==0) {
      return;
    }
    
    //get properties of the kernel and the cache
    int kHeight = kHeight(lineRadii);
    int kRadius  = kRadius(lineRadii);
    final int cacheWidth = roiRectangle.width+2*kRadius;
    final int cacheHeight = kHeight + (numThreads>1 ? 2*numThreads : 0);
    //'cache' is the input buffer. Each line y in the image is mapped onto cache line y%cacheHeight
    final float[] cache = new float[cacheWidth*cacheHeight];
    //this line+1 will be read into the cache first
    this.highestYinCache = Math.max(roiRectangle.y-kHeight/2, 0) - 1;
    
    //copy the pointer of the image processor
    final ImageProcessor imageProcessor = this.imageProcessor;
    
    //threads announce here which line they currently process
    final int[] yForThread = new int[numThreads];
    Arrays.fill(yForThread, -1);
    yForThread[numThreads-1] = roiRectangle.y-1; //first thread started should begin at roi.y
    //thread number 0 is this one, not in the array
    final Thread[] threads = new Thread[numThreads-1];
    //this rng is for producing random seeds for each thread
    RandomGenerator rng = new MersenneTwister(System.currentTimeMillis());
    //instantiate threads and start them
    for (int t=numThreads-1; t>0; t--) {
      final int ti = t; //thread number
      final long seed = rng.nextLong();
      //SEE ANONYMOUS CLASS
      //thread runs method doFiltering
      final Thread thread = new Thread(
          new Runnable() {
            final public void run() {
              threadFilter(imageProcessor, lineRadii, cache, cacheWidth, cacheHeight, yForThread,
                  ti, seed, aborted);
            }
          },
      "RankFilters-"+t);
      thread.setPriority(Thread.currentThread().getPriority());
      thread.start();
      threads[ti-1] = thread;
    }
    
    //main thread start filtering
    this.threadFilter(imageProcessor, lineRadii, cache, cacheWidth, cacheHeight, yForThread, 0,
        rng.nextLong(), aborted);
    
    //join each thread
    for (final Thread thread : threads) {
      try {
          if (thread != null) thread.join();
      } catch (InterruptedException e) {
        aborted[0] = true;
        Thread.currentThread().interrupt(); //keep interrupted status (PlugInFilterRunner needs it)
      }
    }
    
    this.showProgress(1.0);
    pass++;
  }
  
  /**METHOD: THREAD FILTER
   * Empirical null filter a grayscale image for a given thread
   * Synchronization: unless a thread is waiting, we avoid the overhead of 'synchronized'
   * statements. That's because a thread waiting for another one should be rare.
   *
   * Data handling: The area needed for processing a line is written into the array 'cache'.
   * This is a stripe of sufficient width for all threads to have each thread processing one
   * line, and some extra space if one thread is finished to start the next line.
   * This array is padded at the edges of the image so that a surrounding with radius kRadius
   * for each pixel processed is within 'cache'. Out-of-image
   * pixels are set to the value of the nearest edge pixel. When adding a new line, the lines in
   * 'cache' are not shifted but rather the smaller array with the start and end pointers of the
   * kernel area is modified to point at the addresses for the next line.
   *
   * Notes: For mean and variance, except for very small radius, usually do not calculate the
   * sum over all pixels. This sum is calculated for the first pixel of every line only. For the
   * following pixels, add the new values and subtract those that are not in the sum any more.
   * 
   * @param ip the image to be filtered
   * @param lineRadii pointer used by the kernel, see method makeLineRadii
   * @param cache pointer to the cache
   * @param cacheWidth
   * @param cacheHeight
   * @param yForThread array indicating which y a thread is filtering
   * @param threadNumber
   * @param seed seed for rng
   * @param aborted
   */
  private void threadFilter(ImageProcessor ip, int[] lineRadii, float[] cache, int cacheWidth,
      int cacheHeight, int [] yForThread, int threadNumber, long seed, boolean[] aborted) {
    
    if (aborted[0] || Thread.currentThread().isInterrupted()) {
      return;
    }
    
    //get properties of this image
    int width = ip.getWidth();
    int height = ip.getHeight();
    Rectangle roiRectangle = ip.getRoi();
    
    //get properties of this kernel
    int kHeight = kHeight(lineRadii);
    int kRadius  = kRadius(lineRadii);
    int kNPoints = kNPoints(lineRadii);
    
    //get the boundary
    int xmin = roiRectangle.x - kRadius;
    int xmax = roiRectangle.x + roiRectangle.width + kRadius;
    
    //get the pointer of the kernel given the width of the cache
    int[]cachePointers = makeCachePointers(lineRadii, cacheWidth);
    
    //pad out the image, eg when the kernel is on the boundary of the image
    int padLeft = xmin<0 ? -xmin : 0;
    int padRight = xmax>width? xmax-width : 0;
    int xminInside = xmin>0 ? xmin : 0;
    int xmaxInside = xmax<width ? xmax : width;
    int widthInside = xmaxInside - xminInside;
    
    //arrays to store calculations of pixels contained in a kernel
    double[] sums = new double[2]; //[0] sum of greyvalues, [1] sum of greyvalues squared
    double[] quartileBuf = new double[kNPoints]; //stores greyvalues of pixels in a kernel
    //[0,1,2] 1st 2nd and 3rd quartiles greyvalues of pixels in a kernel
    float [] quartiles = new float[3];
    
    //for calculation the normal pdf
    NormalDistribution normal = new NormalDistribution();
    //rng for trying out different initial values
    RandomGenerator rng = new MersenneTwister(seed);
    
    boolean smallKernel = kRadius < 2; //indicate if this kernel is small
    
    //values is a 2D array
    //each dimension contain the pixel values for each image
    //dim 1:
      //0. filtered image
      //1. 2. 3. ... are the output images
    float[][] values = new float[N_IMAGE_OUTPUT+1][];
    float [] pixels = (float[]) ip.getPixels();
    //get the pixels of the filtered image
    values[0] = pixels;
    //get the pixels of each of the output images (only if requested)
    for (int i=0; i<N_IMAGE_OUTPUT; i++) {
      if ( (this.outputImagePointer >> i) % 2 == 1) {
        values[i+1] = (float[]) this.outputImageArray[i].getPixels();
      }
    }
    
    int numThreads = yForThread.length;
    long lastTime = System.currentTimeMillis();
    int previousY = kHeight/2-cacheHeight;
    
    //while loop, loop over each y
    while (!aborted[0]) {
      
      //=====THREAD CONTROL===== (untouched from original source code)
      
      int y = arrayMax(yForThread) + 1; // y of the next line that needs processing
      yForThread[threadNumber] = y; //indicate that this thread is working on y
      boolean threadFinished = y >= roiRectangle.y+roiRectangle.height;
      //'if' is not synchronized to avoid overhead
      if (numThreads>1 && (threadWaiting || threadFinished))
        synchronized(this) {
          notifyAll();          // we may have blocked another thread
        }
      if (threadFinished) {
        return; // all done, break the loop
      }
      if (threadNumber==0) { // main thread checks for abort and ProgressBar
        long time = System.currentTimeMillis();
        if (time-lastTime>100) {
          lastTime = time;
          this.showProgress((y-roiRectangle.y)/(double)(roiRectangle.height));
          if (Thread.currentThread().isInterrupted()
              || (this.imageProcessor!= null && IJ.escapePressed())) {
            aborted[0] = true;
            synchronized(this) {notifyAll();}
            return;
          }
        }
      }
      
      for (int i=0; i<cachePointers.length; i++) {  //shift kernel pointers to new line
        cachePointers[i] = (cachePointers[i] + cacheWidth*(y-previousY))%cache.length;
      }
      previousY = y;
      
      if (numThreads>1) { // thread synchronization
        //non-synchronized check to avoid overhead
        int slowestThreadY = arrayMinNonNegative(yForThread);
       //we would overwrite data needed by another thread
        if (y - slowestThreadY + kHeight > cacheHeight) {
          synchronized(this) {
            slowestThreadY = arrayMinNonNegative(yForThread); //recheck whether we have to wait
            if (y - slowestThreadY + kHeight > cacheHeight) {
              do {
                notifyAll(); //avoid deadlock: wake up others waiting
                threadWaiting = true;
                try {
                  wait();
                  if (aborted[0]) {
                    return;
                  }
                } catch (InterruptedException e) {
                  aborted[0] = true;
                  notifyAll();
                  //keep interrupted status (PlugInFilterRunner needs it)
                  Thread.currentThread().interrupt();
                  return;
                }
                slowestThreadY = arrayMinNonNegative(yForThread);
              } while (y - slowestThreadY + kHeight > cacheHeight);
            } //end if
            threadWaiting = false;
          }
        }
      }
      
      //=====READ INTO CACHE===== (untouched from original source code)
      
      if (numThreads==1) {
        int yStartReading = y==roiRectangle.y ? Math.max(roiRectangle.y-kHeight/2, 0) : y+kHeight/2;
        for (int yNew = yStartReading; yNew<=y+kHeight/2; yNew++) { //only 1 line except at start
          this.readLineToCacheOrPad(pixels, width, height, roiRectangle.y, xminInside, widthInside,
              cache, cacheWidth, cacheHeight, padLeft, padRight, kHeight, yNew);
        }
      } else {
        if (!copyingToCache || highestYinCache < y+kHeight/2) synchronized(cache) {
          copyingToCache = true; // copy new line(s) into cache
          while (highestYinCache < arrayMinNonNegative(yForThread) - kHeight/2 + cacheHeight - 1) {
            int yNew = highestYinCache + 1;
            this.readLineToCacheOrPad(pixels, width, height, roiRectangle.y, xminInside, widthInside,
              cache, cacheWidth, cacheHeight, padLeft, padRight, kHeight, yNew);
            highestYinCache = yNew;
          }
          copyingToCache = false;
        }
      }
      
      //=====FILTER A LINE=====
      
      int cacheLineP = cacheWidth * (y % cacheHeight) + kRadius;  //points to pixel (roiRectangle.x, y)
      this.filterLine(values, width, cache, cachePointers, kNPoints, cacheLineP, roiRectangle, y,
          sums, quartileBuf, quartiles, normal, rng, smallKernel);
    }// end while (!aborted[0]); loops over y (lines)
  }
  
  /**METHOD: ARRAY MAX
   * Used by thread control in threadFilter
   * @param array
   * @return maximum in array
   */
  private int arrayMax(int[] array) {
    int max = Integer.MIN_VALUE;
    for (int i=0; i<array.length; i++) {
      if (array[i] > max) {
        max = array[i];
      }
    }
    return max;
  }
  
  /**METHOD: ARRAY MIN NON NEGATIVE
   * Used by thread control in threadFilter
   * @param array
   * @return the minimum of the array, but not less than 0
   */
  private int arrayMinNonNegative(int[] array) {
    int min = Integer.MAX_VALUE;
    for (int i=0; i<array.length; i++) {
      if (array[i]<min) {
        min = array[i];
      }
    }
    return min<0 ? 0 : min;
  }
  
  /**METHOD: FILTER LINE
   * Empirical null filter a line
   * @param values array of float [] for output values to be stored
   * @param width width of the image
   * @param cache contains pixels of the pre-filter image
   * @param cachePointers pointers used by the kernel
   * @param kNPoints number of points a kernel contains
   * @param cacheLineP pointer for the current y line in the cache
   * @param roiRectangle
   * @param y current row
   * @param sums stores sum calculations in a kernel (size 2)
   * @param quartileBuf stores greyvalues in a kernel (size kNPoints)
   * @param quartiles stores quartiles in a kernel (size 2)
   * @param normal normal distribution to evaluate the normal pdf
   * @param rng random number generator, used for trying out different initial values
   * @param smallKernel indicate if this kernel is small or not
   */
  private void filterLine(float[][] values, int width, float[] cache, int[] cachePointers,
      int kNPoints, int cacheLineP, Rectangle roiRectangle, int y, double[] sums, double[] quartileBuf,
      float[] quartiles, NormalDistribution normal, RandomGenerator rng,
      boolean smallKernel) {
    
    //declare the pointer for a pixel in values
    int valuesP = roiRectangle.x+y*width;
    //indicate if a full calculation is to be done
    //that is to do a calculation without using results from the previous x
    boolean fullCalculation = true;
    float std; //standard deviation
    int nData = 0; //number of non-nan data
    float initialValue = 0; //initial value to be used for the newton-raphson method
    
    //then for each pixel in this line
    for (int x=0; x<roiRectangle.width; x++, valuesP++) { // x is with respect to roiRectangle.x
      
      //if this pixel is not in the roi, for the next pixel do a full calculation as the summation
          //cannot be propagate
      //else this pixel is in the roi and filter this pixel
      if (!this.roi.contains(roiRectangle.x+x, y)) {
        fullCalculation = true;
      } else {
        getQuartiles(cache, x, cachePointers, quartileBuf, kNPoints, quartiles);
        
        if (fullCalculation) {
          //set the initial value to be the median
          initialValue = quartiles[1];
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
        
        if (nData < 2) {
          fullCalculation = true;
          values[0][valuesP] = Float.NaN;
        } else {
          
          //calculate the standard deviation
          std = (float) Math.sqrt(((sums[1] - sums[0]*sums[0]/nData)/(nData-1)));
          
          //get the empirical null
          EmpiricalNull empiricalNull = new EmpiricalNull(this.nInitial, this.nStep,
              this.log10Tolerance, this.bandwidthParameterA, this.bandwidthParameterB, cache, x,
              cachePointers , initialValue, quartiles, std, nData, normal, rng);
          empiricalNull.estimateNull();
          //normalise this pixel
          values[0][valuesP] = (cache[cacheLineP+x] - empiricalNull.getNullMean())
              / empiricalNull.getNullStd();
          //for the next x, the initial value is this nullMean
          initialValue = empiricalNull.getNullMean();
          //for each requested output image, save that statistic
          for (int i=0; i<N_IMAGE_OUTPUT; i++) {
            if ( (this.outputImagePointer >> i) % 2 == 1) {
              switch (i) {
                case 0:
                  values[1][valuesP] = empiricalNull.getNullMean();
                  break;
                case 1:
                  values[2][valuesP] = empiricalNull.getNullStd();
                  break;
                case 2:
                  values[3][valuesP] = std;
                  break;
                case 3:
                  values[4][valuesP] = quartiles[0];
                  break;
                case 4:
                  values[5][valuesP] = quartiles[1];
                  break;
                case 5:
                  values[6][valuesP] = quartiles[2];
                  break;
              }
            }
          }
        }
      }
    }
  }
  
  /**METHOD: READ LINE TO CACHE OR PAD
   * Read a line into the cache (including padding in x), anything outside the boundary is nan
   * @param pixels
   * @param width
   * @param height
   * @param roiY
   * @param xminInside
   * @param widthInside
   * @param cache modified
   * @param cacheWidth
   * @param cacheHeight
   * @param padLeft
   * @param padRight
   * @param kHeight
   * @param y
   */
  private void readLineToCacheOrPad(float [] pixels, int width, int height, int roiY,
      int xminInside, int widthInside, float[]cache, int cacheWidth, int cacheHeight, int padLeft,
      int padRight, int kHeight, int y) {
    int lineInCache = y%cacheHeight;
    if (y < height) {
      readLineToCache(pixels, y*width, y, xminInside, widthInside, cache, lineInCache*cacheWidth,
          padLeft, padRight);
      if (y==0) {
        for (int prevY = roiY-kHeight/2; prevY<0; prevY++) {  //for y<0, pad with nan
          int prevLineInCache = cacheHeight+prevY;
          Arrays.fill(cache, prevLineInCache*cacheWidth, prevLineInCache*cacheWidth + cacheWidth,
              Float.NaN);
        }
      }
    } else {
      Arrays.fill(cache, lineInCache*cacheWidth, lineInCache*cacheWidth + cacheWidth, Float.NaN);
    }
  }
  
  /**METHOD: READ LINE TO CACHE
   * Read a line into the cache (includes conversion to flaot)
   * Pad with nan if necessary
   * @param pixels
   * @param pixelLineP
   * @param y
   * @param xminInside
   * @param widthInside
   * @param cache modified
   * @param cacheLineP
   * @param padLeft
   * @param padRight
   */
  private void readLineToCache(float [] pixels, int pixelLineP, int y, int xminInside,
      int widthInside, float[] cache, int cacheLineP, int padLeft, int padRight) {
    
    //for each pixel in the line
    for (int x=0; x<widthInside; x++) {
      //if this pixel is in the roi, copy it to the cache, else put nan in the cache
      float toCopytoCache;
      if (!this.roi.contains(xminInside + x, y)) {
        toCopytoCache = Float.NaN;
      } else {
        toCopytoCache = pixels[pixelLineP+xminInside + x];
      }
      cache[cacheLineP+padLeft+x] = toCopytoCache;
    }
    //Padding contains NaN
    for (int cp=cacheLineP; cp<cacheLineP+padLeft; cp++) {
      cache[cp] = Float.NaN;
    }
    for (int cp=cacheLineP+padLeft+widthInside; cp<cacheLineP+padLeft+widthInside+padRight; cp++) {
      cache[cp] = Float.NaN;
    }
  }
  
  /**METHOD: GET AREA SUMS
   * Get sum of values and values squared within the kernel area.
   * x between 0 and cacheWidth-1
   * Output is written to array sums[0] = sum; sums[1] = sum of squares
   * Ignores nan
   * Returns the number of non-nan numbers
   * @param cache
   * @param xCache0
   * @param kernel
   * @param sums modified
   * @return  number of non-nan numbers
   */
  private static int getAreaSums(float[] cache, int xCache0, int[] kernel, double[] sums) {
    double sum=0, sum2=0;
    int nData = 0;
    //y within the cache stripe (we have 2 kernel pointers per cache line)
    for (int kk=0; kk<kernel.length; kk++) {
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
  
  /**METHOD: ADD SIDE SUMS
   * Add all values and values squared at the right border inside minus at the left border outside
   * the kernal area.
   * Output is added or subtracted to/from array sums[0] += sum; sums[1] += sum of squares  when at
   * the right border, minus when at the left border
   * @param cache
   * @param xCache0
   * @param kernel
   * @param sums modified
   * @param nData
   * @return number of non-nan numbers
   */
  private static int addSideSums(float[] cache, int xCache0, int[] kernel, double[] sums,
      int nData) {
    double sum=0, sum2=0;
    //for each row
    for (int kk=0; kk<kernel.length; /*k++;k++ below*/) {
      double v = cache[kernel[kk++]+(xCache0-1)]; //this value is not in the kernel area any more
      if (!Double.isNaN(v)) {
        sum -= v;
        sum2 -= v*v;
        nData--;
      }
      v = cache[kernel[kk++]+xCache0]; //this value comes into the kernel area
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
  
  /**METHOD: GET QUARTILES
   * Get the quartiles of values within kernel-sized neighborhood.
   * nan values are ignored
   * @param cache
   * @param xCache0
   * @param kernel
   * @param quartileBuf
   * @param kNPoints
   * @param quartiles modified
   */
  private static void getQuartiles(float[] cache, int xCache0, int[] kernel,
      double[] quartileBuf, int kNPoints, float[] quartiles) {
    //copy the greyvalues in a pixel into quartileBuf
    int nFinite=0;
    for (int kk=0; kk<kernel.length; kk++) {
      for (int p=kernel[kk++]+xCache0; p<=kernel[kk]+xCache0; p++) {
        float v = cache[p];
        if (!Float.isNaN(v)) {
          quartileBuf[nFinite] = (double) v;
          nFinite++;
        }
        
      }
    }
    //percentile only works if there are 2 or more values
    if (nFinite >= 2) {
      Percentile percentile = new Percentile();
      percentile.setData(quartileBuf, 0, nFinite);
      for (int i=0; i<3; i++) {
        quartiles[i] = (float) percentile.evaluate((i+1) * 25.0);
      }
    }
  }
  
  /**METHOD: MAKE LINE RADII
   * Create a circular kernel (structuring element) of a given radius.
   * @param radius
   * Radius = 0.5 includes the 4 neighbors of the pixel in the center,
   *  radius = 1 corresponds to a 3x3 kernel size.
   * @return the circular kernel
   * The output is an array that gives the length of each line of the structuring element
   * (kernel) to the left (negative) and to the right (positive):
   * [0] left in line 0, [1] right in line 0,
   * [2] left in line 2, ...
   * The maximum (absolute) value should be kernelRadius.
   * Array elements at the end:
   * length-2: nPoints, number of pixels in the kernel area
   * length-1: kernelRadius in x direction (kernel width is 2*kernelRadius+1)
   * Kernel height can be calculated as (array length - 1)/2 (odd number);
   * Kernel radius in y direction is kernel height/2 (truncating integer division).
   * Note that kernel width and height are the same for the circular kernels used here,
   * but treated separately for the case of future extensions with non-circular kernels.
   * e.g. r=0.5 will return [0,0,-1,1,0,0,nPoints, kernelRadius]
   * e.g. r=3 will return [-1,1,-2,2,-3,3,-3,3,-3,3,-2,2,-1,1,nPoints, kernelRadius]
   */
  private int[] makeLineRadii(double radius) {
    if (radius>=1.5 && radius<1.75) {//this code creates the same sizes as the previous RankFilters
      radius = 1.75;
    } else if (radius>=2.5 && radius<2.85) {
      radius = 2.85;
    }
    int r2 = (int) (radius*radius) + 1;
    int kRadius = (int)(Math.sqrt(r2+1e-10));
    int kHeight = 2*kRadius + 1;
    int[] kernel = new int[2*kHeight + 2];
    kernel[2*kRadius] = -kRadius;
    kernel[2*kRadius+1] =  kRadius;
    int nPoints = 2*kRadius+1;
    for (int y=1; y<=kRadius; y++) { //lines above and below center together
      int dx = (int)(Math.sqrt(r2-y*y+1e-10));
      kernel[2*(kRadius-y)] = -dx;
      kernel[2*(kRadius-y)+1] =  dx;
      kernel[2*(kRadius+y)] = -dx;
      kernel[2*(kRadius+y)+1] =  dx;
      nPoints += 4*dx+2; //2*dx+1 for each line, above&below
    }
    kernel[kernel.length-2] = nPoints;
    kernel[kernel.length-1] = kRadius;
    return kernel;
  }
  
  /**METHOD: KERNEL HEIGHT
   * @param lineRadii 
   * @return kernel height
   */
  private int kHeight(int[] lineRadii) {
    return (lineRadii.length-2)/2;
  }
  
  //
  /**METHOD: KERNEL RADIUS
   * @param lineRadii see makeLineRadii
   * @return kernel radius in x direction. width is 2+kRadius+1
   */
  private int kRadius(int[] lineRadii) {
    return lineRadii[lineRadii.length-1];
  }
  
  /**METHOD: KERNEL N POINTS
   * @param lineRadii see makeLineRadii
   * @return number of points in kernal area
   */
  private int kNPoints(int[] lineRadii) {
    return lineRadii[lineRadii.length-2];
  }
  
  /**METHOD: MAKE CACHE POINTERS
   * cache pointers for a given kernel and cache width
   * @param lineRadii see makeLineRadii
   * @param cacheWidth width of the cache
   * @return cache pointers for a given kernel
   * e.g. radius = 0.5, cache width = 10 returns [1,1,10,12,21,21];
   */
  private int[] makeCachePointers(int[] lineRadii, int cacheWidth) {
    int kRadius = kRadius(lineRadii);
    int kHeight = kHeight(lineRadii);
    int[] cachePointers = new int[2*kHeight];
    for (int i=0; i<kHeight; i++) {
      cachePointers[2*i]   = i*cacheWidth+kRadius + lineRadii[2*i];
      cachePointers[2*i+1] = i*cacheWidth+kRadius + lineRadii[2*i+1];
    }
    return cachePointers;
  }
  
  //=====STATIC FUNCTIONS AND PROCEDURES=====
  
  /**PROCEDURE: SET NUMBER OF INITIAL POINTS
   * @param nInitial must be 1 or bigger
   * @throws InvalidValue
   */
  public void setNInitial(int nInitial) throws InvalidValue {
    if (nInitial>0) {
      this.nInitial = nInitial;
    } else {
      throw new InvalidValue("number of initial points must be positive");
    }
  }
  
  /**FUNCTION: GET N INITIAL
   * @return number of initial points to try out in newton-raphson
   */
  public int getNInitial() {
    return this.nInitial;
  }
  
  /**PROCEDURE: SET NUMBER OF STEPS
   * @param nStep
   * @throws InvalidValue
   */
  public void setNStep(int nStep) throws InvalidValue {
    if (nStep>0) {
      this.nStep = nStep;
    } else {
      throw new InvalidValue("number of steps must be positive");
    }
  }
  
  /**FUNCTION: GET N STEP
   * @return number of steps to do in newton-raphson
   */
  public int getNStep() {
    return this.nStep;
  }
  
  /**PROCEDURE: SET LOG 10  TOLERANCE
   * Stops the newton-raphson algorithm when (Math.abs(dxLnF[1])<tolerance)
   * where dxLnF is the first diff of the log density
   * @param log10Tolerance
   */
  public void setLog10Tolerance(float log10Tolerance){
    this.log10Tolerance = log10Tolerance;
  }
  
  /**METHOD: GET LOG 10 TOLERANCE
   * @return log10 tolerance when doing newton-raphson
   */
  public float getLog10Tolerance() {
    return this.log10Tolerance;
  }
  
  /**PROCEDURE: SET BANDWIDTH A
   * The bandwidth for the density estimate is
   * bandwidthParameterB * Math.min(dataStd, iqr/1.34f)
        * ((float) Math.pow((double) this.n, -0.2))
        + bandwidthParameterA;
   * @param bandwidthParameterA
   */
  public void setBandwidthA(float bandwidthParameterA) {
    this.bandwidthParameterA = bandwidthParameterA;
  }
  
  /**FUNCTION: GET BANDWIDTH A
   * The bandwidth for the density estimate is
   * bandwidthParameterB * Math.min(dataStd, iqr/1.34f)
        * ((float) Math.pow((double) this.n, -0.2))
        + bandwidthParameterA;
   * @return bandwidthParameterA
   */
  public float getBandwidthA() {
    return this.bandwidthParameterA;
  }
  
  /**METHOD: SET BANDWIDTH B
   * The bandwidth for the density estimate is
   * bandwidthParameterB * Math.min(dataStd, iqr/1.34f)
        * ((float) Math.pow((double) this.n, -0.2))
        + bandwidthParameterA;
   * @param bandwidthParameterB
   */
  public void setBandwidthB(float bandwidthParameterB) {
    this.bandwidthParameterB = bandwidthParameterB;
  }
  
  /**FUNCTION: GET BANDWIDTH B
   * The bandwidth for the density estimate is
   * bandwidthParameterB * Math.min(dataStd, iqr/1.34f)
        * ((float) Math.pow((double) this.n, -0.2))
        + bandwidthParameterA;
   * @return bandwidthParameterB
   */
  public float getBandwidthB() {
    return this.bandwidthParameterB;
  }
  
  /**METHOD: SHOW MASKS
   * Show the kernel
   */
  void showMasks() {
    int w=150, h=150;
    ImageStack stack = new ImageStack(w, h);
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
  
  /**METHOD: SET N PASSES
   * This method is called by ImageJ to set the number of calls to run(ip)
   * corresponding to 100% of the progress bar
   */
  @Override
  public void setNPasses (int nPasses) {
    this.nPasses = nPasses;
    pass = 0;
  }
  
  /**METHOD: SHOW PROGRESS
   * @param percent
   */
  protected void showProgress(double percent) {
    if (this.isShowProgressBar) {
      int nPasses2 = nPasses;
      percent = (double)pass/nPasses2 + percent/nPasses2;
      //print progress bar
      int length = 20;
      int nArrow = (int) Math.round(percent * 20);
      System.out.print("[");
      for (int i=0; i<length; i++) {
        if (i<=nArrow) {
          System.out.print(">");
        } else {
          System.out.print(".");
        }
      }
      System.out.println("]");
    }
  }
  
  /**METHOD: SET PROGRESS BAR
   * Turn the progress bar on or off
   * @param isShowProgressBar true to show the progress bar
   */
  public void setProgress(boolean isShowProgressBar) {
    this.isShowProgressBar = isShowProgressBar;
  }
  
}
