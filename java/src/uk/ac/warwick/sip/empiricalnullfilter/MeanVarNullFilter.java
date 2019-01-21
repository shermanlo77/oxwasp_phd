package uk.ac.warwick.sip.empiricalnullfilter;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.RandomGenerator;

/**CLASS: MEAN VAR NULL FILTER
 * Superclass of EmpiricalNullFilter
 * Replaces the empirical null mean with mean
 * Replaces the empirical null variance with variance
 * @author sherman
 */
public class MeanVarNullFilter extends EmpiricalNullFilter {
  
  public MeanVarNullFilter() {
  }
  
  /**METHOD: GET NULL MEAN STD
   * Returns the mean and std
   * @param values NOT USED
   * @param cache NOT USED
   * @param x NOT USED
   * @param cachePointers NOT USED
   * @param cacheLineP NOT USED
   * @param initialValue NOT USED
   * @param quartiles NOT USED
   * @param mean 
   * @param std
   * @param nData number of non-nan data
   * @param normal NOT USED
   * @param rng NOT USED
   * @return 2-vector, [null mean, null std]
   */
  protected float[] getNullMeanStd(float[][] values, float[] cache, int x, int[] cachePointers,
      int cacheLineP, float initialValue, float[] quartiles, float mean, float std, int nData,
      NormalDistribution normal, RandomGenerator rng) {
    //declare 2 vector to store the null mean and null std
    float[] nullMeanStd = new float[2];
    //get the empirical null
    nullMeanStd[0] = mean;
    nullMeanStd[1] = std;
    return nullMeanStd;
  }
  
  
}
