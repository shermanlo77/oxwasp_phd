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
  protected float[] getNullMeanStd(float initialValue, Cache cache, Kernel kernel,
      NormalDistribution normal, RandomGenerator rng) {
    //declare 2 vector to store the null mean and null std
    float[] nullMeanStd = new float[2];
    //get the empirical null
    EmpiricalNull empiricalNull = new MadModeNull(this.nInitial, this.nStep, this.log10Tolerance,
        this.bandwidthParameterA, this.bandwidthParameterB, initialValue, kernel, normal,
        rng);
    //estimate the null and get it
    empiricalNull.estimateNull();
    nullMeanStd[0] = empiricalNull.getNullMean();
    nullMeanStd[1] = empiricalNull.getNullStd();
    return nullMeanStd;
  }
  
}
