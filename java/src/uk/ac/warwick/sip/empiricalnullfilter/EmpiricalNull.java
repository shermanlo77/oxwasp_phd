package uk.ac.warwick.sip.empiricalnullfilter;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.RandomGenerator;

public class EmpiricalNull {
  
  //STATIC FINAL VARIABLES (these are used for default values)
  //number of times to repeat the newton-raphson using different initial values
  static final int N_INITIAL = 3;
  static final int N_STEP = 10; //number of steps in newton-raphson
  //stopping condition tolerance for newton-raphson
  static final float LOG_10_TOLERANCE = -5.0f;
  //the bandwidth for the density estimate is B x 0.9 x std x n^{-1/5} + A
  //A and B are set below
  static final float BANDWIDTH_PARAMETER_A = (float) 0.15; //intercept
  static final float BANDWIDTH_PARAMETER_B = (float) 0.90; //gradient
  
  //STATIC VERSIONS OF STATIC FINAL VARIABLES
  private int nInitial;
  private int nStep;
  private float tolerance;
  private float bandwidthParameterA;
  private float bandwidthParameterB;
  
  //MEMBER VARIABLES
  
  private float [] cache; //array of greyvalues
  private int x; //x position
  private int [] cachePointers; //array of integer pairs, pointing to the boundary of the kernel
  
  private int n = 0; //number of non-NaN data in the kernel
  private float dataStd; //the standard deviation of the pixels in the kernel
  private float iqr; //interquartile range
  
  private float initialValue; //the user requested initial value
  private float nullMean; //empirical null mean
  private float nullStd; //empirical null std
  private float bandwidth; //bandwidth for the density estimate
  private NormalDistribution normalDistribution; //standard normal distributionrng = rng;
  private RandomGenerator rng; //random number generator when a random initial value is needed
  
  /**CONSTRUCTOR
   * 
   * @param nInitial number of times to repeat the newton-raphson using different initial values
   * @param nStep number of steps in newton-raphson
   * @param log10Tolerance stopping condition tolerance for newton-raphson, stopping condition is
   *     Math.abs(dxLnF[1])<this.tolerance where dxLnF[1] is the gradient of the log density and
   *     this.tolerance is 10^log10Tolerance
   * @param bandwidthParameterA the bandwidth for the density estimate is
   *     B x 0.9 x std x n^{-1/5} + A
   * @param bandwidthParameterB the bandwidth for the density estimate is
   *     B x 0.9 x std x n^{-1/5} + A
   * @param cache array of greyvalues
   * @param x x position
   * @param cachePointers array of integer pairs, pointing to the boundary of the kernel
   * @param initialValue initial value for the newton-raphson
   * @param dataStd standard deviation of the pixels in the kernel
   * @param normalDistribution standard normal distribution object
   * @param rng random number generator when a random initial value is needed
   */
  public EmpiricalNull(int nInitial, int nStep, float log10Tolerance, float bandwidthParameterA,
      float bandwidthParameterB, float[] cache, int x, int[] cachePointers , float initialValue,
      float[] quartiles, float dataStd, int n, NormalDistribution normalDistribution,
      RandomGenerator rng) {
    this.nInitial = nInitial;
    this.nStep = nStep;
    this.tolerance = (float) Math.pow(10.0, log10Tolerance);
    this.bandwidthParameterA = bandwidthParameterA;
    this.bandwidthParameterB = bandwidthParameterB;
    this.cache = cache;
    this.x = x;
    this.cachePointers = cachePointers;
    this.initialValue = initialValue;
    this.dataStd = dataStd;
    this.iqr = quartiles[2] - quartiles[0];
    this.n = n;
    this.normalDistribution = normalDistribution;
    this.rng = rng;
  }
  
  /**METHOD: ESTIMATE NULL
   * Estimate the parameters nullMean and nullStd
   */
  public void estimateNull() {
    //get the bandwidth for the density estimate
    this.bandwidth = this.bandwidthParameterB * Math.min(dataStd, this.iqr/1.34f)
        * ((float) Math.pow((double) this.n, -0.2))
        + this.bandwidthParameterA;
    //get the initial value, if it not finite, get a random one
    float initialValue = this.initialValue;
    if (!isFinite(initialValue)) {
      initialValue = this.getRandomInitial();
    }
    
    //declare arrays, storing the max density and empirical null parameters 
    //for each new initial point
    float [] densityArray = new float[this.nInitial];
    float [] nullMeanArray = new float[this.nInitial];
    float [] nullStdArray = new float[this.nInitial];
    
    //for each initial point
    for (int i=0; i<this.nInitial; i++) {
      //find the maximum density, get the empirical null parameters and save them
      float[] densityAndNull = this.findMode(initialValue);
      densityArray[i] = densityAndNull[0];
      nullMeanArray[i] = densityAndNull[1];
      nullStdArray[i] = densityAndNull[2];
      //get a random value for the next initial point
      initialValue = this.getRandomInitial();
    }
    
    //find the maximum density
    //the empirical null parameters chosen are the ones with the highest density
    float maxDensity = Float.NEGATIVE_INFINITY;
    for (int i=0; i<this.nInitial; i++) {
      if (densityArray[i] > maxDensity) {
        maxDensity = densityArray[i];
        this.nullMean = nullMeanArray[i];
        this.nullStd = nullStdArray[i];
      }
    }
  }
  
  /**METHOD: SET NULL TO RANDOM DATA
   * Set the initial value for the newton-raphson method to a random data point
   * This is the orginal initial value plus Gaussian noise
   */
  private float getRandomInitial() {
    return this.initialValue + ((float) this.rng.nextGaussian()) * this.dataStd;
  }
  
  /**METHOD: FIND MODE
   * Find the mode of the log density using the newton-raphson method
   * @return 3-array, [0] contains the maximum density, [1] empirical null mean, [2] empirical null
   * std
   */
  private float[] findMode(float greyvalue) {
    //declare array for the output
    float[] densityAndNull = new float[3];
    //declare flag to indiciate if a solution has been found
    boolean foundSolution = false;
    //while no solution has been found
    while (!foundSolution) {
      
      //get the density, 1st and 2nd diff of the ln density at the initial value
      float [] dxLnF = this.getDLnDensity(greyvalue);
      
      //for n_step
      for (int i=0; i<this.nStep; i++) {
        
        //update the solution to the mode
        greyvalue -= dxLnF[1]/dxLnF[2];
        //get the density, 1st and 2nd diff of the ln density at the new value
        dxLnF = this.getDLnDensity(greyvalue);
        //if this gradient is within tolerance, break the i_step for loop
        if (Math.abs(dxLnF[1])<this.tolerance) {
          break;
        }
        //if any of the variables are not finite, stop the algorithm
        if (!isFinite(dxLnF[0])) {
          break;
        } else if (!isFinite(dxLnF[1])) {
          break;
        }  else if (!isFinite(dxLnF[2])) {
          break;
        } else if (!isFinite(greyvalue)) {
          break;
        }
      }
      
      //check if the solution to the mode is a maxima by looking at the 2nd diff
      //if any of the variables are not finite, start again with a random initial value
      if (EmpiricalNull.isFinite(dxLnF[0]) && EmpiricalNull.isFinite(dxLnF[1])
          && EmpiricalNull.isFinite(dxLnF[2]) && EmpiricalNull.isFinite(greyvalue)
          && (dxLnF[2] < 0)) {
        foundSolution = true;
        densityAndNull[0] = dxLnF[0];
        densityAndNull[1] = greyvalue;
        densityAndNull[2] = (float) Math.pow((double) -dxLnF[2], -0.5);
      }
      greyvalue = this.getRandomInitial();
    }
    
    return densityAndNull;
  }
  
  /**METHOD: GET D LN DENSITY
   * Return a 3 element array containing:
   * 0. the density (ignore any constant multiplied to it) (NOT THE LOG)
   * 1. the first derivative of the log density
   * 2. the second derivative of the log density
   * @param greyValue the value of the derivative to be evaluated at
   * @return 3 element array containing derivatives
   */
  private float [] getDLnDensity(float greyValue) {
    
    //declare array for storing 3 sums where
    //z = (x - x_i) / h where x is the point of evaluation, x_i is a data point, h is the bandwidth
    //0. sum phi(z)
    //1. sum phi(z)z
    //2. sum phi(z)z^2
    //where phi is the Gaussian pdf
    float [] sumKernel = new float [3];
    float z;
    float phiZ;
    
    //for each non-NaN pixel in the kernel
    for (int kk=0; kk<this.cachePointers.length; kk++) {
      for (int p=this.cachePointers[kk++]+x; p<=this.cachePointers[kk]+x; p++) {
        if (!Float.isNaN(this.cache[p])) {
          
          //get phi(z)
          z = (cache[p] - greyValue) / this.bandwidth;
          phiZ = (float) this.normalDistribution.density((double) z);
          //update the sum of the kernels
          sumKernel[0] += phiZ;
          sumKernel[1] += phiZ * z;
          sumKernel[2] += phiZ * z * z;
          
        }
      }
    }
    
    //declare array for storing the 3 derivatives of the log density
    //0. the density up to a constant
    //1. 1st derivative
    //2. 2nd derivative
    float [] dxLnF = new float[3];
    dxLnF[0] = sumKernel[0];
    dxLnF[1] = sumKernel[1]/(this.bandwidth*sumKernel[0]);
    dxLnF[2] = (sumKernel[0]*(sumKernel[2] - sumKernel[0]) - sumKernel[1]*sumKernel[1]) 
        / ((float)Math.pow((double)(this.bandwidth*sumKernel[0]),2.0));
    
    return dxLnF;
  }
  
  /**METHOD: GET NULL MEAN
   * @return null mean after calling estimateNull()
   */
  public float getNullMean() {
    return this.nullMean;
  }
  
  /**METHOD: GET NULL STD
   * @return null std after calling estimateNull()
   */
  public float getNullStd() {
    return this.nullStd;
  }
  
  /**FUNCTION: IS FINITE
   * Indicate if the float is finite, ie not nan and not infinite
   * @param f
   * @return
   */
  public static boolean isFinite(float f) {
    return !(Float.isNaN(f) || Float.isInfinite(f));
  }
}
