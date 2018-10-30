package uk.ac.warwick.sip.empiricalNullFilter;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.omg.DynamicAny.DynAnyPackage.InvalidValue;

public class EmpiricalNull {
  
  //STATIC VARIABLES
  
  //number of times to repeat the newton-raphson using different initial values
  private static int nInitial = 20;
  private static int nStep = 10; //number of steps in newton-raphson
  //stopping condition tolerance for newton-raphson
  private static float log10Tolerance = -5.0f;
  private static float tolerance = (float) Math.pow(10.0, log10Tolerance);
  
  //the bandwidth for the density estimate is B x 0.9 x std x n^{-1/5} + A
  //A and B are set below
  private static float bandwidthParameterA = (float) 0.16; //intercept
  private static float bandwidthParameterB = (float) 0.9; //gradient
  
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
   * @param cache array of greyvalues
   * @param x x position
   * @param cachePointers array of integer pairs, pointing to the boundary of the kernel
   * @param initialValue initial value for the newton-raphson
   * @param dataStd standard deviation of the pixels in the kernel
   * @param normalDistribution standard normal distribution object
   * @param rng random number generator when a random initial value is needed
   */
  public EmpiricalNull(float[] cache, int x, int[] cachePointers , float initialValue,
      float[] quartiles, float dataStd, int n, NormalDistribution normalDistribution,
      RandomGenerator rng) {
    this.cache = cache;
    this.x = x;
    this.cachePointers = cachePointers;
    this.initialValue = initialValue;
    this.dataStd = dataStd;
    this.iqr = quartiles[2] - quartiles[0];
    this.n = n;
    this.normalDistribution = normalDistribution;
    this.rng = rng;
    this.bandwidth = EmpiricalNull.bandwidthParameterB * Math.min(dataStd, iqr/1.34f)
        * ((float) Math.pow((double) this.n, -0.2))
        + EmpiricalNull.bandwidthParameterA;
  }
  
  /**METHOD: ESTIMATE NULL
   * Estimate the parameters nullMean and nullStd
   */
  public void estimateNull() {
    //get the initial value, if it not finite, get a random one
    float initialValue = this.initialValue;
    if (!Float.isFinite(initialValue)) {
      initialValue = this.getRandomInitial();
    }
    
    //declare arrays, storing the max density and empirical null parameters 
    //for each new initial point
    float [] densityArray = new float[EmpiricalNull.nInitial];
    float [] nullMeanArray = new float[EmpiricalNull.nInitial];
    float [] nullStdArray = new float[EmpiricalNull.nInitial];
    
    //for each initial point
    for (int i=0; i<EmpiricalNull.nInitial; i++) {
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
    for (int i=0; i<EmpiricalNull.nInitial; i++) {
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
      for (int i=0; i<EmpiricalNull.nStep; i++) {
        
        //update the solution to the mode
        greyvalue -= dxLnF[1]/dxLnF[2];
        //get the density, 1st and 2nd diff of the ln density at the new value
        dxLnF = this.getDLnDensity(greyvalue);
        //if this gradient is within tolerance, break the i_step for loop
        if (Math.abs(dxLnF[1])<EmpiricalNull.tolerance) {
          break;
        }
        //if any of the variables are not finite, stop the algorithm
        if (!Float.isFinite(dxLnF[0])) {
          break;
        } else if (!Float.isFinite(dxLnF[1])) {
          break;
        }  else if (!Float.isFinite(dxLnF[2])) {
          break;
        } else if (!Float.isFinite(greyvalue)) {
          break;
        }
      }
      
      //check if the solution to the mode is a maxima by looking at the 2nd diff
      //if any of the variables are not finite, start again with a random initial value
      if (Float.isFinite(dxLnF[0]) && Float.isFinite(dxLnF[1]) && Float.isFinite(dxLnF[2])
          && Float.isFinite(greyvalue) && (dxLnF[2] < 0)) {
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
  
  //=====STATIC FUNCTIONS AND PROCEDURES=====
  
  /**PROCEDURE: SET NUMBER OF INITIAL POINTS
   * @param nInitial must be 1 or bigger
   * @throws InvalidValue
   */
  public static void setNInitial(int nInitial) throws InvalidValue {
    if (nInitial>0) {
      EmpiricalNull.nInitial = nInitial;
    } else {
      throw new InvalidValue("number of initial points must be positive");
    }
  }
  
  /**FUNCTION: GET N INITIAL
   * @return number of initial points to try out in newton-raphson
   */
  public static int getNInitial() {
    return nInitial;
  }
  
  /**PROCEDURE: SET NUMBER OF STEPS
   * @param nStep
   * @throws InvalidValue
   */
  public static void setNStep(int nStep) throws InvalidValue {
    if (nStep>0) {
      EmpiricalNull.nStep = nStep;
    } else {
      throw new InvalidValue("number of steps must be positive");
    }
  }
  
  /**FUNCTION: GET N STEP
   * @return number of steps to do in newton-raphson
   */
  public static int getNStep() {
    return nStep;
  }
  
  /**PROCEDURE: SET LOG 10  TOLERANCE
   * Stops the newton-raphson algorithm when (Math.abs(dxLnF[1])<tolerance)
   * where dxLnF is the first diff of the log density
   * @param log10Tolerance
   */
  public static void setLog10Tolerance(float log10Tolerance){
    EmpiricalNull.log10Tolerance = log10Tolerance;
    tolerance = (float) Math.pow(10, (double)log10Tolerance);
  }
  
  /**METHOD: GET LOG 10 TOLERANCE
   * @return log10 tolerance when doing newton-raphson
   */
  public static float getLog10Tolerance() {
    return log10Tolerance;
  }
  
  /**PROCEDURE: SET BANDWIDTH A
   * The bandwidth for the density estimate is
   * bandwidthParameterB * Math.min(dataStd, iqr/1.34f)
        * ((float) Math.pow((double) this.n, -0.2))
        + bandwidthParameterA;
   * @param bandwidthParameterA
   */
  public static void setBandwidthA(float bandwidthParameterA) {
    EmpiricalNull.bandwidthParameterA = bandwidthParameterA;
  }
  
  /**FUNCTION: GET BANDWIDTH A
   * The bandwidth for the density estimate is
   * bandwidthParameterB * Math.min(dataStd, iqr/1.34f)
        * ((float) Math.pow((double) this.n, -0.2))
        + bandwidthParameterA;
   * @return bandwidthParameterA
   */
  public static float getBandwidthA() {
    return bandwidthParameterA;
  }
  
  /**METHOD: SET BANDWIDTH B
   * The bandwidth for the density estimate is
   * bandwidthParameterB * Math.min(dataStd, iqr/1.34f)
        * ((float) Math.pow((double) this.n, -0.2))
        + bandwidthParameterA;
   * @param bandwidthParameterB
   */
  public static void setBandwidthB(float bandwidthParameterB) {
    EmpiricalNull.bandwidthParameterB = bandwidthParameterB;
  }
  
  /**FUNCTION: GET BANDWIDTH B
   * The bandwidth for the density estimate is
   * bandwidthParameterB * Math.min(dataStd, iqr/1.34f)
        * ((float) Math.pow((double) this.n, -0.2))
        + bandwidthParameterA;
   * @return bandwidthParameterB
   */
  public static float getBandwidthB() {
    return bandwidthParameterB;
  }
  
}
