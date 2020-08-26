//MIT License
//Copyright (c) 2019 Sherman Lo

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

//CLASS: EMPIRICAL NULL FILTER
/**Locally normalise the grey values using the empirical null  mean (mode) and the empirical null
 *     std.
 *
 * <p>The method filter is overloaded but all are directed to filter() with no parameters.
 *     filter() with no parameter is used directly by ImageJ because the image is passed through the
 *     methods setup and run. The methods filter(float [][] image) and
 *     filter(float [][] image, String roiPath) are required to pass the image when used by MATLAB.
 *
 * <p>How to use:
 *   <ul><li>Compile to a .jar file and use ImageJ or Fiji</li></ul>
 *
 * <p>For use outside ImageJ such as MATLAB:
 *   <ul>
 *     <li>Use the empty constructor</li>
 *     <li>Set the radius using the method setRadius(double)</li>
 *     <li>Call the method setNPasses(1)</li>
 *     <li>Call the method filter(float [][] image) or filter(float [][] image, String roiPath)</li>
 *     <li>Call the method getFilteredImage() to get the filtered image</li>
 *     <li>Call the method getOutputImage(int outputImagePointer) to get any other images</li>
 *   </ul>
 *
 * <p>Based on
 *     <a href=https://github.com/imagej/ImageJA/blob/7f965b866c9db364b0b47140caeef4f62d5d8c15/src/main/java/ij/plugin/filter/RankFilters.java>
 *     RankFilters.java</a>
 *
 * @author Sherman Lo
 */
public class EmpiricalNullFilter implements ExtendedPlugInFilter, DialogListener {

  //STATIC FINAL VARIABLES

  //
  /**Options for what output images to show. To combine the options, either add them together or use
   *     the OR operator.
   */
  public static final int NULL_MEAN = 1, NULL_STD = 2, STD = 4, Q1 = 8, Q2 = 16, Q3 = 32;
  /**name of each output image*/
  public static final String[] OUTPUT_NAME = {"null mean", "null std", "standard deviation",
      "quantile 1", "median", "quantile 3"};

  //STATIC VARIABLES
  /**values used in the previous filtering*/
  private static double lastRadius = 0;
  /**values used in the previous filtering*/
  private static int lastNInitial = EmpiricalNull.N_INITIAL;
  /**values used in the previous filtering*/
  private static int lastNStep = EmpiricalNull.N_STEP;
  /**values used in the previous filtering*/
  private static float lastLog10Tolerance = EmpiricalNull.LOG_10_TOLERANCE;
  /**values used in the previous filtering*/
  private static float lastBandwidthParameterA = EmpiricalNull.BANDWIDTH_PARAMETER_A;
  /**values used in the previous filtering*/
  private static float lastBandwidthParameterB = EmpiricalNull.BANDWIDTH_PARAMETER_B;
  /**values used in the previous filtering*/
  private static int lastOutputImagePointer = -1;

  //MEMBER VARIABLES

  /**this filter only works on 32-bit images, this is indicated in FLAGS*/
  protected int flags = DOES_32;
  /**number of output images which can be shown*/
  protected int nImageOutput = 6;
  /**which output images to show*/
  protected int outputImagePointer = NULL_MEAN + NULL_STD;
  /**array of float processors which contains images (or statistics) which are obtained from the
   *     filter itself, eg null mean, null std, std, q1, q2, q3
   */
  protected FloatProcessor [] outputImageArray = new FloatProcessor[this.nImageOutput];
  /** radius of the kernel*/
  private double radius = 2;

  /**the image to be filtered*/
  protected ImageProcessor imageProcessor;
  /**region of interest*/
  protected Roi roi;
  /**used by showDialog, unused but needed in case deleted by automatic garbage collection*/
  private PlugInFilterRunner pfr;
  protected boolean isShowProgressBar = false;
  /**The number of passes (color channels * stack slices)*/
  protected int nPasses = 1;
  protected int pass;

  //EMPIRICAL NULL RELATED
  /**number of initial values for newton-raphson*/
  protected int nInitial = lastNInitial;
  /**number of steps in newton-raphson*/
  protected int nStep = lastNStep;
  /**stopping condition tolerance for newton-raphson*/
  protected float log10Tolerance = lastLog10Tolerance;
  /**the bandwidth parameter A where the bandwidth for the density estimate is
   *     (B x n^{-1/5} + A) * std
   */
  protected float bandwidthParameterA = lastBandwidthParameterA;
  /**the bandwidth parameter B where the bandwidth for the density estimate is
   *     (B x n^{-1/5} + A) * std
   */
  protected float bandwidthParameterB = lastBandwidthParameterB;

  /**indicate if pixels in the kernel need to be copied to a float[]*/
  protected boolean isKernelCopy = true;
  /**indicate if the kernel mean and var is required*/
  protected boolean isKernelMeanVar = true;
  /**indicate if the kernel quartiles is required*/
  protected boolean isKernelQuartile = true;

  /**seed for the rng*/
  private int seed = 1742863098;

  //MULTITHREADING RELATED
  /**number of threads*/
  private int numThreads = Prefs.getThreads();
  /**Current state of processing is in class variables. Thus, stack parallelization must be done
   *     ONLY with one thread for the image
   */
  private boolean threadWaiting; // a thread waits until it may read data

  //CONSTRUCTOR
  /**Empty constructor, used by ImageJ
   */
  public EmpiricalNullFilter() {
  }

  //IMPLEMENTED: SETUP
  /**Setup of the PlugInFilter. Returns the flags specifying the capabilities and needs
   * of the filter.
   * @param arg not used
   * @param ip not used
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
    return this.flags;
  }

  //IMPLEMENTED: RUN
  /**For the use of ExtendedPlugInFilter. Do the filtering.
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
    for (int i=0; i<this.nImageOutput; i++) {
      if ((outputImagePointer >> i) % 2 == 1) {
        ImagePlus output = new ImagePlus(OUTPUT_NAME[i], this.outputImageArray[i]);
        output.show();
      }
    }
  }

  //IMPLEMENTED: SHOW DIALOG
  /**Dialog box for setting the radius and which output images to show
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

    //checkbox each output image
    if (this.nImageOutput > 0) {
      genericDialog.addMessage("Output images");
      if (lastOutputImagePointer >= 0) {
        this.outputImagePointer = lastOutputImagePointer;
      }
      for (int i=0; i<this.nImageOutput; i++) {
        boolean defaultBoolean = (this.outputImagePointer >> i) % 2 == 1;
        genericDialog.addCheckbox("Show "+OUTPUT_NAME[i], defaultBoolean);
      }
    }

    this.showOptionsInDialog(genericDialog);

    //the DialogItemChanged method will be called on user input
    genericDialog.addDialogListener(this);
    genericDialog.showDialog(); //display the dialog
    if (genericDialog.wasCanceled()) {
      return DONE;
    }
    //protected static class variables (filter parameters) from garbage collection
    IJ.register(this.getClass());

    if (Macro.getOptions() == null) { //interactive only: remember settings
      lastRadius = this.radius;
      lastNInitial = this.nInitial;
      lastNStep = this.nStep;
      lastLog10Tolerance = this.log10Tolerance;
      lastBandwidthParameterA = this.bandwidthParameterA;
      lastBandwidthParameterB = this.bandwidthParameterB;
      lastOutputImagePointer = this.outputImagePointer;
    }

    //save a copy of pfr
    this.pfr = pfr;

    return this.flags;
  }

  public void showOptionsInDialog(GenericDialog genericDialog) {
    //add fields for the empirical null tuning parameters
    //integers do not show decimal points
    genericDialog.addMessage("Advanced options");
    genericDialog.addNumericField("number of initial values", this.getNInitial(),
        0, 6, null);
    genericDialog.addNumericField("number of steps", this.getNStep(),
        0, 6, null);
    genericDialog.addNumericField("log tolerance", this.getLog10Tolerance(),
        2, 6, null);
  }

  //IMPLEMENTED: DIALOG ITEM CHANGED
  /**Called on user input and update the radius and outputImagePointer.
   * @parm gd GUI
   * @parm e Not used
   * @return true if input values are valid, else false
   */
  @Override
  public boolean dialogItemChanged(GenericDialog genericDialog, AWTEvent e) {
    //get the radius and set it
    this.setRadius(genericDialog.getNextNumber());
    if (genericDialog.invalidNumber() || this.radius < 0) {
      return false;
    }
    //get the output image options and save it
    this.outputImagePointer = 0;
    for (int i=0; i<this.nImageOutput; i++) {
      boolean value = genericDialog.getNextBoolean();
      if (value) {
        int pointer = 1;
        pointer  <<= i;
        this.outputImagePointer += pointer;
      }
    }
    //get the empirical null parameters and set it
    //if an exception is caught, return false to indicate an invalid item change
    try {
      this.changeValueFromDialog(genericDialog);
    } catch (InvalidValueException exception) {
      return false;
    }
    return true;
  }

  protected void changeValueFromDialog(GenericDialog genericDialog) throws InvalidValueException {
    try {
      this.setNInitial((int) genericDialog.getNextNumber());
      this.setNStep((int) genericDialog.getNextNumber());
      this.setLog10Tolerance((float) genericDialog.getNextNumber());
    } catch (InvalidValueException exception) {
      throw exception;
    }
  }

  //METHOD: GET RADIUS
  /**@return the radius of the kernel
   */
  public double getRadius() {
    return this.radius;
  }

  //METHOD: SET RADIUS
  /**@param radius The radius of the kernel
   */
  public void setRadius(double radius) {
    this.radius = radius;
  }

  //METHOD: GET FILTERED IMAGE
  /**@return array of pixels of the filtered image
   */
  public float [] getFilteredImage() {
    return (float []) this.imageProcessor.getPixels();
  }

  //METHOD: GET OUTPUT IMAGE
  /**Returns an array of pixels from one of the requested output images
   * @param outputImagePointer e.g. NULL_MEAN, NULL_STD
   * @return float array containing the value of each pixel of a requested output image
   */
  public float [] getOutputImage(int outputImagePointer) {
    //for each output image
    for (int i=0; i<this.nImageOutput; i++) {
      //if the user requested this output image, return it
      if ( (outputImagePointer >> i) % 2 == 1) {
        return (float [] ) this.outputImageArray[i].getPixels();
      }
    }
    return null;
  }

  //METHOD: SET OUTPUT IMAGE
  /**Set outputImagePointer, this indicate which output images to show
   * @param pointer which output images to show, use and the static int variables,
   *     e.g. NULL_MEAN + NULL_STD
   */
  public void setOutputImage(int pointer) {
    this.outputImagePointer = pointer;
  }

  //METHOD: FILTER
  /**Call the method filter() using the image passed in the parameter
   * @param image image to be filtered
   */
  public void filter(float [][] image) {
    this.imageProcessor = new FloatProcessor(image);
    this.roi = new Roi(this.imageProcessor.getRoi());
    this.filter();
  }

  //METHOD: FILTER
  /**Call the method filter() using the image and ROI passed in the parameter
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

  /**METHOD: Perpare to call doFiltering
   */
  public void filter() {
    //setup the kernel
    Kernel.setKernel(this.radius);
    //instantiate new images for each outout
    for (int i=0; i<this.nImageOutput; i++) {
      if ((this.outputImagePointer >> i) % 2 == 1) {
        FloatProcessor outputProcessor = (FloatProcessor) this.imageProcessor.duplicate();
        float [] pixels = (float []) outputProcessor.getPixels();
        Arrays.fill(pixels, Float.NaN);
        this.outputImageArray[i] = outputProcessor;
      }
    }
    final Cache cache = this.instantiateCache();
    this.doFiltering(cache);
  }

  /**METHOD: Instantiate a cache (different cache can have paddings)
   */
  protected Cache instantiateCache() {
    return new Cache(this.imageProcessor, this.roi);
  }

  //METHOD: FILTER
  /**Do the empirical null filter using several threads.
   *
   * <p>Implementation: each thread uses the same input buffer (cache), always works on the next
   *     unfiltered line. Usually, one thread reads reads several lines into the cache, while the
   *     others are processing the data.
   *
   * <p>'aborted[0]' is set if the main thread has been interrupted (during preview) or ESC pressed.
   *     'aborted' must not be a class variable because it signals the other threads to stop;
   *     and this may be caused by an interrupted preview thread after the main calculation has been
   *     started.
   */
  protected void doFiltering(final Cache cache) {
    //returns whether interrupted during preview or ESC pressed
    final boolean[] aborted = new boolean[1];
    //roi = region of interest
    Rectangle roiRectangle = this.imageProcessor.getRoi();

    //get the number of threads
    int numThreads = Math.min(roiRectangle.height, this.numThreads);
    if (numThreads==0) {
      return;
    }

    //threads announce here which line they currently process
    final int[] yForThread = new int[numThreads];
    Arrays.fill(yForThread, roiRectangle.y-1);
    //thread number 0 is this one, not in the array
    final Thread[] threads = new Thread[numThreads-1];
    //this rng is for producing random seeds for each thread
    RandomGenerator rng = new MersenneTwister(this.seed);

    //produce a random seed for each row
    final int seeds[] = new int[roiRectangle.height];
    for (int i=0; i<roiRectangle.height; i++) {
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
    this.pass++;
  }

  //METHOD: THREAD FILTER
  /**Empirical null filter a grayscale image for a given thread.
   *
   * <p>Synchronization: unless a thread is waiting, we avoid the overhead of 'synchronized'
   *     statements. That's because a thread waiting for another one should be rare.
   *
   * <p>Data handling: The area needed for processing a line is written into the array 'cache'.
   *     This is a strip of sufficient height for all threads to have each thread processing one
   *     line, and some extra space if one thread is finished to start the next line.
   *     This array is padded at the edges of the image so that a surrounding with radius kRadius
   *     for each pixel processed is within 'cache'. Out-of-image pixels are set to NaN. When adding
   *     a new line, the lines in 'cache' are not shifted but rather the smaller array with the
   *     start and end pointers of the kernel area is modified to point at the addresses for the
   *     next line.
   *
   * <p>Notes: For mean and variance, except for very small radius, usually do not calculate the
   *     sum over all pixels. This sum is calculated for the first pixel of every line only. For the
   *     following pixels, add the new values and subtract those that are not in the sum any more.
   *
   * @param cache pointer to the cache
   * @param yForThread array indicating which y a thread is filtering
   * @param threadNumber id for this thread
   * @param seeds seeds for rng, one for each row
   * @param aborted pointer to a boolean
   */
  private void threadFilter(Cache cache, int [] yForThread, int threadNumber,int[] seeds,
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
    float[][] values = new float[this.nImageOutput+1][];
    float [] pixels = (float[]) this.imageProcessor.getPixels();
    //get the pixels of the filtered image
    values[0] = pixels;
    //get the pixels of each of the output images (only if requested)
    for (int i=0; i<this.nImageOutput; i++) {
      if ( (this.outputImagePointer >> i) % 2 == 1) {
        values[i+1] = (float[]) this.outputImageArray[i].getPixels();
      }
    }

    int numThreads = yForThread.length;
    long lastTime = System.currentTimeMillis();

    //while loop, loop over each y
    while (!aborted[0]) {

      //=====THREAD CONTROL=====

      this.updateYForThread(yForThread, threadNumber);
      int y = yForThread[threadNumber];

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

      //=====FILTER A LINE=====

      //set rng for this line and filter this line
      rng.setSeed(seeds[y - roiRectangle.y]);
      this.filterLine(values, cache, kernel, y, normal, rng);
    }// end while (!aborted[0]); loops over y (lines)
  }

  //METHOD: UPDATE Y FOR THREAD
  /**Update the array of y positions of each thread so that the y position of the current thread is
   *     at the next unfiltered row
   * @param yForThread y position of each thread
   * @param threadNumber
   */
  private synchronized void updateYForThread(int[] yForThread, int threadNumber) {
    int y = Integer.MIN_VALUE;
    for (int i=0; i<yForThread.length; i++) {
      if (yForThread[i] > y) {
        y = yForThread[i];
      }
    }
    y += 1; // y of the next line that needs processing
    yForThread[threadNumber] = y; //indicate that this thread is working on y
  }

  //METHOD: FILTER LINE
  /**Empirical null filter a line
   * @param values array of float [] for output values to be stored
   * @param cache contains pixels of the pre-filter image
   * @param kernel contains pixels in the kernel and its statistics
   * @param y current row
   * @param normal normal distribution to evaluate the normal pdf
   * @param rng random number generator, used for trying out different initial values
   */
  private void filterLine(float[][] values, Cache cache, Kernel kernel, int y,
      NormalDistribution normal, RandomGenerator rng) {

    //declare the pointer for a pixel in values
    int valuesP = this.imageProcessor.getRoi().x+y*this.imageProcessor.getWidth();
    float initialValue = Float.NaN; //initial value to be used for the newton-raphson method
    kernel.moveToNewLine(y);
    boolean isPreviousFinite = false; //boolean to indicate if the previous pixel is finite
    //do the filter while moving the kernel to the right
    do {
      if (kernel.isFinite()) {

        //if the previous pixel is not finite, then use the median as the initial value
        if (!isPreviousFinite) {
          initialValue = kernel.getQuartiles()[1];
        }
        isPreviousFinite = true;

        //get the null mean and null std
        try {
          float [] nullMeanStd = this.getNullMeanStd(initialValue, kernel, normal, rng);
          //normalise this pixel
          this.updatePixelInImage(values[0], valuesP, nullMeanStd);
          //for the next x, the initial value is this nullMean
          initialValue = nullMeanStd[0];
          //update all the output images
          this.updateOutputImages(values, valuesP, nullMeanStd, kernel);
        //if there are problems with the Newton-Raphson, then abort
        } catch (ConvergenceException exception) {
          isPreviousFinite = false;
        }
      } else { //else this pixel is not finite
        isPreviousFinite = false;
      }
      valuesP++;
    } while(kernel.moveRight());
  }

  //METHOD: UPDATE PIXEL IN IMAGE
  /**Update a pixel in the image to be filtered
   * @param values array of pixels of the iamge to be filtered
   * @param valuesP pointer to the pixel to be updated
   * @param nullMeanStd 2-element array, mode and empirical null std of this pixel
   */
  protected void updatePixelInImage(float [] values, int valuesP, float [] nullMeanStd) {
    values[valuesP] -= nullMeanStd[0];
    values[valuesP] /= nullMeanStd[1];
  }

  //METHOD: GET NULL MEAN STD
  /**Perform the Newton-Raphson on the kernel density estimate to get the empirical null mean and
   *     the empirical null std
   * @param initialValue starting value for the Newton-Raphson
   * @param kernel contains pixels in the kernel and its statistics
   * @param normal normal distribution to evaluate the normal pdf
   * @param rng random number generator for producing new initial values
   * @return empirical null mean and empirical null std
   * @throws ConvergenceException thrown when newton-raphson struggled to converge
   */
  protected float[] getNullMeanStd(float initialValue, Kernel kernel, NormalDistribution normal,
      RandomGenerator rng) throws ConvergenceException{
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
    } catch (ConvergenceException exception) {
      //exception is caught, use median as initial value this time
      try {
        empiricalNull = new EmpiricalNull(this.nInitial, this.nStep, this.log10Tolerance,
            this.bandwidthParameterA, this.bandwidthParameterB, kernel.getMedian(), kernel, normal,
            rng);
        empiricalNull.estimateNull();
      } catch (ConvergenceException exceptionAfterMedian) { //median as initial value didn't work
        throw exceptionAfterMedian;
      }
    }
    //assign null mean and null std
    nullMeanStd[0] = empiricalNull.getNullMean();
    nullMeanStd[1] = empiricalNull.getNullStd();
    return nullMeanStd;
  }

  //METHOD: UPDATE OUTPUT IMAGES
  /**Update the pixels in the output images, one pixel at a time
   * @param values array of float [] for output values to be stored
   * @param valuesP pointer to the pixel to be updated
   * @param nullMeanStd 2-element array, mode and empirical null std of this pixel
   * @param kernel contains pixels in the kernel and its statistics
   */
  protected void updateOutputImages(float[][] values, int valuesP, float[] nullMeanStd,
      Kernel kernel) {
    //for each requested output image, save that statistic
    for (int i=0; i<this.nImageOutput; i++) {
      if ((this.outputImagePointer >> i) % 2 == 1) {
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
  }

  //=====STATIC FUNCTIONS AND PROCEDURES=====

  //PROCEDURE: SET NUMBER OF INITIAL POINTS
  /**@param nInitial must be 1 or bigger
   * @throws InvalidValueException
   */
  public void setNInitial(int nInitial) throws InvalidValueException {
    if (nInitial>0) {
      this.nInitial = nInitial;
    } else {
      throw new InvalidValueException("number of initial points must be positive");
    }
  }

  //FUNCTION: GET N INITIAL
  /**@return number of initial points to try out in newton-raphson
   */
  public int getNInitial() {
    return this.nInitial;
  }

  //PROCEDURE: SET NUMBER OF STEPS
  /**@param nStep
   * @throws InvalidValueException
   */
  public void setNStep(int nStep) throws InvalidValueException {
    if (nStep>0) {
      this.nStep = nStep;
    } else {
      throw new InvalidValueException("number of steps must be positive");
    }
  }

  //FUNCTION: GET N STEP
  /**@return number of steps to do in newton-raphson
   */
  public int getNStep() {
    return this.nStep;
  }

  //PROCEDURE: SET LOG 10  TOLERANCE
  /**Stops the newton-raphson algorithm when (Math.abs(dxLnF[1])&lt;tolerance)
   * where dxLnF is the first diff of the log density
   * @param log10Tolerance
   */
  public void setLog10Tolerance(float log10Tolerance){
    this.log10Tolerance = log10Tolerance;
  }

  //METHOD: GET LOG 10 TOLERANCE
  /**@return log10 tolerance when doing newton-raphson
   */
  public float getLog10Tolerance() {
    return this.log10Tolerance;
  }

  //PROCEDURE: SET BANDWIDTH A
  /**The bandwidth for the density estimate is
   * bandwidthParameterB * Math.min(dataStd, iqr/1.34f)
        * ((float) Math.pow((double) this.n, -0.2))
        + bandwidthParameterA;
   * @param bandwidthParameterA
   * @throws InvalidValueException if the parameter is not positive
   */
  public void setBandwidthA(float bandwidthParameterA) throws InvalidValueException{
    if (bandwidthParameterA >= 0) {
      this.bandwidthParameterA = bandwidthParameterA;
    } else {
      throw new InvalidValueException("bandwidthParameterA must be non-negative");
    }
  }

  //FUNCTION: GET BANDWIDTH A
  /**The bandwidth for the density estimate is
   * bandwidthParameterB * Math.min(dataStd, iqr/1.34f)
        * ((float) Math.pow((double) this.n, -0.2))
        + bandwidthParameterA;
   * @return bandwidthParameterA
   */
  public float getBandwidthA() {
    return this.bandwidthParameterA;
  }

  //METHOD: SET BANDWIDTH B
  /**The bandwidth for the density estimate is
   * bandwidthParameterB * Math.min(dataStd, iqr/1.34f)
        * ((float) Math.pow((double) this.n, -0.2))
        + bandwidthParameterA;
   * @param bandwidthParameterB
   */
  public void setBandwidthB(float bandwidthParameterB) throws InvalidValueException{
    if (bandwidthParameterB >= 0) {
      this.bandwidthParameterB = bandwidthParameterB;
    } else {
      throw new InvalidValueException("bandwidthParameterB must be non-negative");
    }
  }

  //FUNCTION: GET BANDWIDTH B
  /**The bandwidth for the density estimate is
   * bandwidthParameterB * Math.min(dataStd, iqr/1.34f)
        * ((float) Math.pow((double) this.n, -0.2))
        + bandwidthParameterA;
   * @return bandwidthParameterB
   */
  public float getBandwidthB() {
    return this.bandwidthParameterB;
  }

  //FUNCTION: SET SEED
  /**Set the seed for the rng used to give seeds for each row
   * @param seed
   */
  public void setSeed(int seed) {
    this.seed = seed;
  }

  //METHOD: SHOW MASKS
  /**Show the kernel
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

  //METHOD: SET N PASSES
  /**This method is called by ImageJ to set the number of calls to run(ip)
   * corresponding to 100% of the progress bar
   */
  @Override
  public void setNPasses (int nPasses) {
    this.nPasses = nPasses;
    this.pass = 0;
  }

  //METHOD: SHOW PROGRESS
  /**@param percent
   */
  protected void showProgress(double percent) {
    if (this.isShowProgressBar) {
      int nPasses2 = nPasses;
      percent = (double)pass/nPasses2 + percent/nPasses2;
      IJ.showProgress(percent);
    }
  }

  //METHOD: SET PROGRESS BAR
  /**Turn the progress bar on or off
   * @param isShowProgressBar true to show the progress bar
   */
  public void setProgress(boolean isShowProgressBar) {
    this.isShowProgressBar = isShowProgressBar;
  }

}
