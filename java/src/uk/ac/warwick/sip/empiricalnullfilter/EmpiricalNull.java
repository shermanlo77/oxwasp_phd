//MIT License
//Copyright (c) 2019 Sherman Lo

package uk.ac.warwick.sip.empiricalnullfilter;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

//CLASS: EMPIRICAL NULL
/**Fits a kernel density on the histogram, finds the mode using the Newton-Raphson method. The null
 *     std uses the 2nd derivate of the log density at the mode. The mode (null mean) and the null
 *     std can be returned so that it can used to normalise the statistics.
 *
 * <p>How to use:
 *   <ul>
 *     <li>Pass the array of statistics to the constructor as well as other parameters required for
 *         the kernel density estimate and the Newton-Raphson method</li>
 *     <li>Call the method estimateNull()</li>
 *     <li>Call the method getNullMean() and getNullStd() to get the null parameters</li>
 *   </ul>
 *
 * @author Sherman Lo
 */
public class EmpiricalNull {

  //STATIC FINAL VARIABLES (these are used for default values)
  /**DEFAULT number of times to repeat the newton-raphson using different initial values*/
  static final int N_INITIAL = 3;
  /**DEFAULT number of steps in newton-raphson*/
  static final int N_STEP = 10;
  /**DEFAULT stopping condition tolerance for newton-raphson*/
  static final float LOG_10_TOLERANCE = -5.0f;
  /**DEFAULT bandwidth for the density estimate is (B x n^{-1/5} + A) * std */
  static final float BANDWIDTH_PARAMETER_A = (float) 0.16; //intercept
  /**DEFAULT bandwidth for the density estimate is (B x n^{-1/5} + A) * std */
  static final float BANDWIDTH_PARAMETER_B = (float) 0.9; //gradient

  //MEMBER VARIABLES

  /**member variable copies of the static final variables*/
  private int nInitial;
  /**member variable copies of the static final variables*/
  private int nStep;
  /**member variable copies of the static final variables*/
  private float log10Tolerance;
  /**member variable copies of the static final variables*/
  private float bandwidthParameterA;
  /**member variable copies of the static final variables*/
  private float bandwidthParameterB;

  /**array of statistics*/
  protected final float[] zArray;
  /**number of non nan*/
  protected final int n;
  /**std of the statistics*/
  private float dataStd;
  /**interquartile range of the statistics*/
  private float iqr;
  /**the user requested initial value*/
  private final float initialValue;
  /**std used to generate new initial values*/
  private float initialSigma;

  /**empirical null mean*/
  private float nullMean;
  /**empirical null std*/
  private float nullStd;
  /**bandwidth for the density estimate*/
  private float bandwidth;
  /**standard normal distribution for evaluating density*/
  private final NormalDistribution normalDistribution;
  /**random number generator when a random initial value is needed*/
  private final RandomGenerator rng;

  //CONSTRUCTOR
  /**To be used by MATLAB, default values are provided for you
   * @param zArray float array of z statistics to be corrected
   * @param initialValue initial value for the newton-raphson
   * @param quartiles array containing the 3 quartiles of the data
   * @param dataStd standard deviation of the data
   * @param n number of non-nan in the kernel at this position
   * @param seed seed for the random number generator
   */
  public EmpiricalNull(float[] zArray, float initialValue, float[] quartiles, float dataStd,
      int n, long seed) {

    this.nInitial = N_INITIAL;
    this.nStep = N_STEP;
    this.log10Tolerance = LOG_10_TOLERANCE;
    this.bandwidthParameterA = BANDWIDTH_PARAMETER_A;
    this.bandwidthParameterB = BANDWIDTH_PARAMETER_B;

    this.initialValue = initialValue;
    this.zArray = zArray;
    this.n = n;
    this.dataStd = dataStd;
    this.iqr = quartiles[2] - quartiles[0];

    this.normalDistribution = new NormalDistribution();
    this.rng = new MersenneTwister(seed);

    this.setBandwidth();
  }

  //CONSTRUCTOR
  /**To be used by EmpiricalNullFilter. All parameters must be provided.
   * @param nInitial number of times to repeat the newton-raphson using different initial values
   * @param nStep number of steps in newton-raphson
   * @param log10Tolerance stopping condition tolerance for newton-raphson, stopping condition is
   *     Math.abs(dxLnF[1])&lt;this.tolerance where dxLnF[1] is the gradient of the log density and
   *     this.tolerance is 10^log10Tolerance
   * @param bandwidthParameterA the bandwidth for the density estimate is
   *     (B x n^{-1/5} + A) * std
   * @param bandwidthParameterB the bandwidth for the density estimate is
   *     (B x n^{-1/5} + A) * std
   * @param initialValue initial value for the newton-raphson
   * @param kernel contains the statistics to do the empirical null on
   * @param normalDistribution standard normal distribution object
   * @param rng random number generator when a random initial value is needed
   */
  public EmpiricalNull(int nInitial, int nStep, float log10Tolerance, float bandwidthParameterA,
      float bandwidthParameterB, float initialValue, Kernel kernel,
      NormalDistribution normalDistribution, RandomGenerator rng) {
    this.nInitial = nInitial;
    this.nStep = nStep;
    this.log10Tolerance = log10Tolerance;
    this.bandwidthParameterA = bandwidthParameterA;
    this.bandwidthParameterB = bandwidthParameterB;

    this.initialValue = initialValue;
    this.zArray = kernel.getPixels();
    this.n = kernel.getNFinite();
    this.dataStd = kernel.getStd();
    this.iqr = kernel.getQuartiles()[2] - kernel.getQuartiles()[0];

    this.normalDistribution = normalDistribution;
    this.rng = rng;

    this.setBandwidth();
  }

  //METHOD: ESTIMATE NULL
  /**Estimate the parameters nullMean and nullStd
   * @throws ConvergenceException if Newton-Raphson failed to converge
   */
  public void estimateNull() throws ConvergenceException{

    //get the initial value, if it not finite, get a random one
    float initialValue = this.initialValue;

    //declare arrays, storing the max density, mode and the 2nd div of the log density at mode
    //for each new initial point
    float [] densityArray = new float[this.nInitial];
    float [] nullMeanArray = new float[this.nInitial];
    float [] secondDivArray = new float[this.nInitial];

    //for each initial point
    for (int i=0; i<this.nInitial; i++) {
      //find the maximum density
      //densityAtMode contains the density at the mode, the position of the mode and the second
          //derivative of the log density at the mode in that order
      try {
        float[] densityAtMode = this.findMode(initialValue);
        densityArray[i] = densityAtMode[0];
        nullMeanArray[i] = densityAtMode[1];
        secondDivArray[i] = densityAtMode[2];
        //get a random value for the next initial point
        initialValue = this.getRandomInitial();
      } catch (ConvergenceException exception) {
        throw exception;
      }

    }

    //the empirical null parameters chosen are the ones with the highest density
    float maxDensity = Float.NEGATIVE_INFINITY;
    int maxPointer = -1; //pointer to the initial value with the highest density
    //find the maximum density for each initial value
    for (int i=0; i<this.nInitial; i++) {
      if (densityArray[i] > maxDensity) {
        maxDensity = densityArray[i];
        maxPointer = i;
      }
    }
    //save the empirical null mean and empirical null std
    this.nullMean = nullMeanArray[maxPointer];
    this.nullStd = this.estimateNullStd(nullMeanArray[maxPointer], secondDivArray[maxPointer]);
  }

  //METHOD: ESTIMATE NULL STD
  /**Return the null std given the mode and the second derivative of the log density
   * @param mode location of the mode (not used in this class, subclasses may do)
   * @param secondDiv second derivative of the log density at the mode
   * @return empirical null std
   */
  protected float estimateNullStd(float mode, float secondDiv) {
    return (float) Math.pow(-secondDiv, -0.5);
  }

  //METHOD: SET NULL TO RANDOM DATA
  /**Set the initial value for the newton-raphson method to a random data point.
   * @return original initial value plus Gaussian noise.
   */
  private float getRandomInitial() {
    return this.initialValue + ((float) this.rng.nextGaussian()) * this.initialSigma;
  }

  //METHOD: FIND MODE
  /**Find the mode of the log density using the newton-raphson method
   * @param greyvalue initial value for the newton-raphson
   * @return 3-array, [0] the maximum density, [1] mode, [2] 2nd div of log density
   * @throws ConvergenceException when the algorithm fails to converge
   */
  private float[] findMode(float greyvalue) throws ConvergenceException{
    //declare array for the output
    float[] densityAtMode = new float[3];
    //declare flag to indiciate if a solution has been found
    boolean foundSolution = false;

    int counter = 0; //count the number of times a new initial value has been generated
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
        if ( Math.log10((double)Math.abs(dxLnF[1])) < this.log10Tolerance ) {
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
        densityAtMode[0] = dxLnF[0];
        densityAtMode[1] = greyvalue;
        densityAtMode[2] = dxLnF[2];
      }
      //generate a new initial value and start from there
      greyvalue = this.getRandomInitial();
      counter++; //count the number of times a initial value has been generated
      //if too many initial values has been generated, throw an exception
      if (counter > 10*this.nInitial) {
        throw new ConvergenceException();
      }
    }
    return densityAtMode;
  }

  //METHOD: GET D LN DENSITY
  /**Return an array containing derivatives of the density estimate evaluated at a point.
   * The elements are:
   * <ol>
   *   <li>the density (ignore any constant multiplied to it) (NOT THE LOG)</l>
   *   <li>the first derivative of the log density</li>
   *   <li>the second derivative of the log density</li>
   * </ol>
   * @param greyvalue the value of the derivative to be evaluated at
   * @return 3 element array containing derivatives
   */
  private float [] getDLnDensity(float greyvalue) {

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
    for (int i=0; i<this.n; i++) {
      z = this.zArray[i];
      if (!Float.isNaN(z)) {
        //get phi(z)
        z = (z - greyvalue) / this.bandwidth;
        phiZ = (float) this.normalDistribution.density((double) z);
        //update the sum of the kernels
        sumKernel[0] += phiZ;
        sumKernel[1] += phiZ * z;
        sumKernel[2] += phiZ * z * z;
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

  //METHOD: GET NULL MEAN
  /**@return null mean after calling estimateNull()
   */
  public float getNullMean() {
    return this.nullMean;
  }

  //METHOD: GET NULL STD
  /**@return null std after calling estimateNull()
   */
  public float getNullStd() {
    return this.nullStd;
  }

  //FUNCTION: IS FINITE
  /**Indicate if the float is finite, ie not nan and not infinite
   * @param f
   * @return
   */
  public static boolean isFinite(float f) {
    return !(Float.isNaN(f) || Float.isInfinite(f) || Float.isInfinite(-f));
  }

  //METHOD: GET N INITIAL
  /**@return number of initial points used in newton-raphson
   */
  public int getNInitial() {
    return this.nInitial;
  }

  //METHOD: SET N INITIAL
  /**@param nInitial number of initial points used in newton-raphson
   */
  public void setNInitial(int nInitial) {
    this.nInitial = nInitial;
  }

  //METHOD: GET N STEP
  /**@return number of steps used in newton-raphson
   */
  public int getNStep() {
    return this.nStep;
  }

  //METHOD: SET N STEP
  /**@param nStep number of steps used in newton-raphson
   */
  public void setNStep(int nStep) {
    this.nStep = nStep;
  }

  //METHOD: GET LOG 10 TOLERANCE
  /**@return stopping condition tolerance for newton-raphson
   * stopping condition is Math.abs(dxLnF[1])&lt;this.tolerance where dxLnF[1] is the gradient of
   *     the log density and this.tolerance is 10^log10Tolerance
   */
  public float getLog10Tolerance() {
    return this.log10Tolerance;
  }

  //METHOD: SET LOG 10 TOLERANCE
  /**@param log10Tolerance stopping condition tolerance for newton-raphson
   * stopping condition is Math.abs(dxLnF[1])&lt;this.tolerance where dxLnF[1] is the gradient of
   *     the log density and this.tolerance is 10^log10Tolerance
   */
  public void setLog10Tolerance(float log10Tolerance) {
    this.log10Tolerance = log10Tolerance;
  }

  //METHOD: SET BANDWIDTH
  /**Set the bandwidth for the kernel density using the bandwidth parameters.
   * Change datastd and the iqr if needed, for example if they are zero
   */
  private void setBandwidth() {
    //if the std is zero, then all the numbers are the same, probably an integer
    //use std of uniform
    if (Float.compare(this.dataStd, 0.0f) == 0)  {
      this.dataStd = 0.289f;
    }
    //if iqr is zero, then use std instead
    if (Float.compare(this.iqr, 0.0f) == 0) {
      this.iqr = this.dataStd * 1.34f;
    }
    //set initialSigma, used for generating new initial values
    this.initialSigma = this.dataStd;
    //get the bandwidth for the density estimate
    this.bandwidth = (this.bandwidthParameterB * ( (float) Math.pow((double) this.n, -0.2))
        + this.bandwidthParameterA) * Math.min(this.dataStd, this.iqr/1.34f);
  }

  //METHOD: SET BANDIWDTH
  /**Set the bandwidth for the kernel density directly
   * @param bandwidth
   */
  public void setBandwidth(float bandwidth) {
    this.bandwidth = bandwidth;
  }

  //METHOD: GET BANDWIDTH PARAMETER A
  /**@return the bandwidth parameter A where the bandwidth for the density estimate is
   *     (B x n^{-1/5} + A) * std
   */
  public float getBandwidthParameterA() {
    return this.bandwidthParameterA;
  }

  //METHOD: SET BANDWIDTH PARAMETER A
  /**Recalculate the bandwidth by changing the bandwidth parameter A
   * @param bandwidthParameterA the bandwidth parameter A where the bandwidth for the density
   *     estimate is (B x n^{-1/5} + A) * std
   */
  public void setBandwidthParameterA(float bandwidthParameterA) {
    this.bandwidthParameterA = bandwidthParameterA;
    this.setBandwidth();
  }

  //METHOD: GET BANDWIDTH PARAMETER B
  /**@return the bandwidth parameter B where the bandwidth for the density estimate is
   *     (B x n^{-1/5} + A) * std
   */
  public float getBandwidthParameterB() {
    return this.bandwidthParameterB;
  }

  //METHOD: SET BANDWIDTH PARAMETER B
  /**Recalculate the bandwidth by changing the bandwidth parameter B
   * @param bandwidthParameterB the bandwidth parameter A where the bandwidth for the density
   *     estimate is (B x n^{-1/5} + A) * std
   */
  public void setBandwidthParameterB(float bandwidthParameterB) {
    this.bandwidthParameterB = bandwidthParameterB;
    this.setBandwidth();
  }

}
