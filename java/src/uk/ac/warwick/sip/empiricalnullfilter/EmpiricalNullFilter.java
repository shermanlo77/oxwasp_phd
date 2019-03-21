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
  protected int nInitial = EmpiricalNull.N_INITIAL;
  protected int nStep = EmpiricalNull.N_STEP;
  protected float log10Tolerance = EmpiricalNull.LOG_10_TOLERANCE;
  protected float bandwidthParameterA = EmpiricalNull.BANDWIDTH_PARAMETER_A;
  protected float bandwidthParameterB = EmpiricalNull.BANDWIDTH_PARAMETER_B;
  
  //indicate if pixels in the kernel need to be copied to a float[]
  protected boolean isKernelCopy = true;
  //indicate if the kernel mean and var is required
  protected boolean isKernelMeanVar = true;
  //indicate if the kernel quartiles is required
  protected boolean isKernelQuartile = true;
  
  //seed for the rng
  private int seed = 1742863098;
  
  //MULTITHREADING RELATED
  private int numThreads = Prefs.getThreads();
  // Current state of processing is in class variables. Thus, stack parallelization must be done
  // ONLY with one thread for the image (not using these class variables):
  private boolean threadWaiting; // a thread waits until it may read data
  
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
        2, 6, null);
    genericDialog.addNumericField("bandwidth A", this.getBandwidthA(),
        2, 6, null);
    genericDialog.addNumericField("bandwidth B", this.getBandwidthB(),
        2, 6, null);
    
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
    //setup the kernel
    Kernel.setKernel(this.radius);
    
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
    
    final Cache cache = new Cache(numThreads, this.imageProcessor, this.roi);
    
    //threads announce here which line they currently process
    final int[] yForThread = new int[numThreads];
    Arrays.fill(yForThread, -1);
    yForThread[numThreads-1] = roiRectangle.y-1; //first thread started should begin at roi.y
    //thread number 0 is this one, not in the array
    final Thread[] threads = new Thread[numThreads-1];
    //this rng is for producing random seeds for each thread
    RandomGenerator rng = new MersenneTwister(this.seed);
    
    //produce a random seed for each row
    final int seeds[] = new int[roiRectangle.height];
    for (int i=0; i<this.imageProcessor.getHeight(); i++) {
      seeds[i] = rng.nextInt();
    }
    
    //instantiate threads and start them
    for (int t=numThreads-1; t>0; t--) {
      final int ti = t; //thread number
      //SEE ANONYMOUS CLASS
      //thread runs method doFiltering
      final Thread thread = new Thread(
          new Runnable() {
            final public void run() {
              threadFilter(cache, yForThread, ti, seeds, aborted);
            }
          },
      "RankFilters-"+t);
      thread.setPriority(Thread.currentThread().getPriority());
      thread.start();
      threads[ti-1] = thread;
    }
    
    //main thread start filtering
    this.threadFilter(cache, yForThread, 0, seeds, aborted);
    
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
  private void threadFilter(Cache cache, int [] yForThread, int threadNumber, int[] seeds,
      boolean[] aborted) {
    
    //get properties of this image
    Rectangle roiRectangle = this.imageProcessor.getRoi();
    //rng for trying out different initial values
    RandomGenerator rng = new MersenneTwister();
    
    //get the pointer of the kernel given the width of the cache
    Kernel kernel = new Kernel(cache, roi, this.isKernelCopy, this.isKernelMeanVar,
        this.isKernelQuartile);
    if (aborted[0] || Thread.currentThread().isInterrupted()) {
      return;
    }
    
    //for calculation the normal pdf
    NormalDistribution normal = new NormalDistribution();
    
    //values is a 2D array
    //each dimension contain the pixel values for each image
    //dim 1:
      //0. filtered image
      //1. 2. 3. ... are the output images
    float[][] values = new float[N_IMAGE_OUTPUT+1][];
    float [] pixels = (float[]) this.imageProcessor.getPixels();
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
      
      if (numThreads>1) { // thread synchronization
        //non-synchronized check to avoid overhead
        int slowestThreadY = arrayMinNonNegative(yForThread);
       //we would overwrite data needed by another thread
        if (y - slowestThreadY + Kernel.getKHeight() > cache.getCacheHeight()) {
          synchronized(this) {
            slowestThreadY = arrayMinNonNegative(yForThread); //recheck whether we have to wait
            if (y - slowestThreadY + Kernel.getKHeight() > cache.getCacheHeight()) {
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
              } while (y - slowestThreadY + Kernel.getKHeight() > cache.getCacheHeight());
            } //end if
            threadWaiting = false;
          }
        }
      }
      
      //=====READ INTO CACHE===== (untouched from original source code)
      
      cache.readIntoCache(yForThread, kernel);
      
      //=====FILTER A LINE=====
      
      //points to pixel (roiRectangle.x, y)
      rng.setSeed(seeds[y]);
      this.filterLine(values, cache, kernel, y, normal, rng);
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
   * @param quartiles stores quartiles in a kernel (size 3)
   * @param normal normal distribution to evaluate the normal pdf
   * @param rng random number generator, used for trying out different initial values
   * @param smallKernel indicate if this kernel is small or not
   */
  private void filterLine(float[][] values, Cache cache, Kernel kernel, int y,
      NormalDistribution normal, RandomGenerator rng) {
    
    int cacheLineP = cache.getCacheWidth() * (y % cache.getCacheHeight()) + Kernel.getKRadius();
    //declare the pointer for a pixel in values
    int valuesP = this.imageProcessor.getRoi().x+y*this.imageProcessor.getWidth();
    float initialValue = 0; //initial value to be used for the newton-raphson method
    kernel.moveToNewLine(y);
    boolean isPreviousFinite = false; //boolean to indicate if the previous pixel is finite
    do {
      if (kernel.isFinite()) {
        
        //if the previous pixel is not finite, then use the median as the initial value
        if (!isPreviousFinite) {
          initialValue = kernel.getQuartiles()[1];
        }
        isPreviousFinite = true;
        
        //get the null mean and null std
        try {
          float [] nullMeanStd = this.getNullMeanStd(initialValue, cache, kernel, normal, rng);
          //normalise this pixel
          values[0][valuesP] =
              (cache.getCache()[cacheLineP+kernel.getX()] - nullMeanStd[0]) / nullMeanStd[1];
          //for the next x, the initial value is this nullMean
          initialValue = nullMeanStd[0];
          //for each requested output image, save that statistic
          for (int i=0; i<N_IMAGE_OUTPUT; i++) {
            if ( (this.outputImagePointer >> i) % 2 == 1) {
              switch (i) {
                case 0:
                  values[1][valuesP] = nullMeanStd[0];
                  break;
                case 1:
                  values[2][valuesP] = nullMeanStd[1];
                  break;
                case 2:
                  values[3][valuesP] = kernel.getStd();
                  break;
                case 3:
                  values[4][valuesP] = kernel.getQuartiles()[0];
                  break;
                case 4:
                  values[5][valuesP] = kernel.getQuartiles()[1];
                  break;
                case 5:
                  values[6][valuesP] = kernel.getQuartiles()[2];
                  break;
              }
            }
          }
        } catch (Exception exception) {
          isPreviousFinite = false;
        }
      } else { //else this pixel is not finite
        isPreviousFinite = false;
      }
      valuesP++;
    } while(kernel.moveRight());
  }
  
  protected float[] getNullMeanStd(float initialValue, Cache cache, Kernel kernel,
      NormalDistribution normal, RandomGenerator rng) throws Exception{
    //declare 2 vector to store the null mean and null std
    float[] nullMeanStd = new float[2];
    //get the empirical null
    EmpiricalNull empiricalNull;
    
    //estimate the null and get it
    //try using the empirical null,if an exception is caught, then use the median as the initial 
        //value and try again, if another exception is caught, then throw exception
    try {
      empiricalNull = new EmpiricalNull(this.nInitial, this.nStep, this.log10Tolerance,
          this.bandwidthParameterA, this.bandwidthParameterB, initialValue, kernel, normal,
          rng);
      empiricalNull.estimateNull();
    } catch (Exception exception1) { //exception is caught, use median as initial value this time
      try {
        empiricalNull = new EmpiricalNull(this.nInitial, this.nStep, this.log10Tolerance,
            this.bandwidthParameterA, this.bandwidthParameterB, kernel.getMedian(), kernel, normal,
            rng);
        empiricalNull.estimateNull();
      } catch (Exception exceptionAfterMedian) { //median as initial value didn't work
        throw exceptionAfterMedian;
      }
    }
    //assign null mean and null std
    nullMeanStd[0] = empiricalNull.getNullMean();
    nullMeanStd[1] = empiricalNull.getNullStd();
    return nullMeanStd;
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
  
  /**FUNCTION: SET SEED
   * Set the seed for the rng used to give seeds for each row
   * @param seed
   */
  public void setSeed(int seed) {
    this.seed = seed;
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
      int y0 = h/2-Kernel.getKHeight()/2;
      for (int i = 0, y = y0; i<Kernel.getKHeight(); i++, y++)
        for (int x = w/2+Kernel.getKernelPointer()[2*i], p = x+y*w;
            x <= w/2+Kernel.getKernelPointer()[2*i+1]; x++, p++)
          pixels[p] = 1f;
      stack.addSlice("radius="+r+", size="+(2*Kernel.getKRadius()+1), ip);
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
