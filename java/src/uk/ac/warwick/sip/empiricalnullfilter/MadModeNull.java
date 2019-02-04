package uk.ac.warwick.sip.empiricalnullfilter;

import java.util.Arrays;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.RandomGenerator;

public class MadModeNull extends EmpiricalNull{
  
  static final float MAD_FACTOR = 1.4826f;
  
  /**CONSTRUCTOR
   * To be used by MATLAB, default values are provided for you
   * @param zArray float array of z statistics to be corrected
   * @param initialValue initial value for the newton-raphson
   * @param quartiles array the 3 quartiles of the data in the kernel at this position
   * @param dataStd standard deviation of the pixels in the kernel t this position
   * @param n number of non-nan in the kernel at this position
   * @param seed seed for the random number generator
   */
  public MadModeNull(float[] zArray, float initialValue, float[] quartiles, float dataStd,
      int n, long seed) {
    super(zArray, initialValue, quartiles, dataStd, n, seed);
  }
  
  /**CONSTRUCTOR
   * To be used by EmpiricalNullFilter
   * All parameters must be provided
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
   * @param quartiles array the 3 quartiles of the data in the kernel at this position
   * @param dataStd standard deviation of the pixels in the kernel t this position
   * @param n number of non-nan in the kernel at this position
   * @param normalDistribution standard normal distribution object
   * @param rng random number generator when a random initial value is needed
   */
  public MadModeNull(int nInitial, int nStep, float log10Tolerance, float bandwidthParameterA,
      float bandwidthParameterB, float initialValue, Kernel kernel,
      NormalDistribution normalDistribution, RandomGenerator rng) {
    super(nInitial, nStep, log10Tolerance, bandwidthParameterA, bandwidthParameterB, initialValue,
        kernel, normalDistribution, rng);
  }
  
  /**METHOD: ESTIMATE NULL STD
   * Return the null std given the mode and the second derivative of the log density
   * @param mode location of the mode
   * @param secondDiv second derivative of the log density at the mode
   * @return empirical null std
   */
  protected float estimateNullStd(float mode, float secondDiv) {
    for (int i=0; i<this.n; i++) {
      this.zArray[i] -= mode;
      this.zArray[i] = Math.abs(this.zArray[i]);
    }
    Arrays.sort(this.zArray);
    if ((this.n%2) == 1) {
      return MAD_FACTOR * this.zArray[(this.n-1)/2];
    } else
      return MAD_FACTOR * (this.zArray[this.n/2] + this.zArray[this.n/2 + 1]) / 2;
    
  }
  
}
