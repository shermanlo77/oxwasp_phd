package uk.ac.warwick.sip.empiricalnullfilter;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.RandomGenerator;

/**CLASS: MEDIAN IQR NULL FILTER
 * Superclass of EmpiricalNullFilter
 * Replaces the empirical null mean with median
 * Replaces the empirical null std with iqr/1.3490
 * @author sherman
 */
public class MedianIqrNullFilter extends EmpiricalNullFilter {
  
  public MedianIqrNullFilter() {
  }
  
  /**METHOD: GET NULL MEAN STD
   * Returns the mean and std
   * @param values NOT USED
   * @param cache NOT USED
   * @param x NOT USED
   * @param cachePointers NOT USED
   * @param cacheLineP NOT USED
   * @param initialValue NOT USED
   * @param quartiles stores quartiles in a kernel (size 3)
   * @param mean NOT USED
   * @param std NOT USED
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
    nullMeanStd[0] = quartiles[1];
    nullMeanStd[1] = (quartiles[2]-quartiles[0]) / 1.349f;
    return nullMeanStd;
  }
  
}
