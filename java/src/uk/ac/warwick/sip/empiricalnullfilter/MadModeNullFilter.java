package uk.ac.warwick.sip.empiricalnullfilter;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.RandomGenerator;

public class MadModeNullFilter extends EmpiricalNullFilter {
  
  public MadModeNullFilter() {
    super();
  }
  
  /**METHOD: GET NULL MEAN STD
   * Returns the null mean and null std
   * @param values array of float [] for output values to be stored
   * @param cache contains pixels of the pre-filter image
   * @param x position
   * @param cachePointers pointers used by the kernel
   * @param cacheLineP pointer for the current y line in the cache
   * @param initialValue for the Newton-Raphson method
   * @param quartiles 3-vector
   * @param mean (not used)
   * @param std
   * @param nData number of non-nan data
   * @param normal
   * @param rng
   * @return 2-vector, [null mean, null std]
   */
  @Override
  protected float[] getNullMeanStd(float initialValue, Cache cache, Kernel kernel,
      NormalDistribution normal, RandomGenerator rng) throws ConvergenceException{
    //declare 2 vector to store the null mean and null std
    float[] nullMeanStd = new float[2];
    //get the empirical null
    EmpiricalNull empiricalNull;
    
    //estimate the null and get it
    //try using the empirical null,if an exception is caught, then use the median as the initial 
    //value and try again, if another exception is caught, then throw exception
    try {
      empiricalNull = new MadModeNull(this.nInitial, this.nStep, this.log10Tolerance,
          this.bandwidthParameterA, this.bandwidthParameterB, initialValue, kernel, normal,
          rng);
      empiricalNull.estimateNull();
    } catch (Exception exception1) {
      try {
        empiricalNull = new MadModeNull(this.nInitial, this.nStep, this.log10Tolerance,
            this.bandwidthParameterA, this.bandwidthParameterB, kernel.getMedian(), kernel, normal,
            rng);
        empiricalNull.estimateNull();
      } catch (Exception exceptionAfterMedian) {
        throw exceptionAfterMedian;
      }
    }
      
    nullMeanStd[0] = empiricalNull.getNullMean();
    nullMeanStd[1] = empiricalNull.getNullStd();
    return nullMeanStd;
  }
  
}
