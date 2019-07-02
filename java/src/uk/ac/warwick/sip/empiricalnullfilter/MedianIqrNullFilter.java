package uk.ac.warwick.sip.empiricalnullfilter;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.RandomGenerator;

//CLASS: MEDIAN IQR NULL FILTER
/**Subclass of EmpiricalNullFilter
 * Replaces the empirical null mean with median
 * Replaces the empirical null std with iqr/1.3490
 * @author sherman
 */
public class MedianIqrNullFilter extends EmpiricalNullFilter {
  
  public MedianIqrNullFilter() {
    super();
    //this filter does not need to work out the kernel mean and variance
    this.isKernelMeanVar = false;
  }
  
  @Override
  protected float[] getNullMeanStd(float initialValue, Cache cache, Kernel kernel,
      NormalDistribution normal, RandomGenerator rng) {
  //declare 2 vector to store the null mean and null std
    float[] nullMeanStd = new float[2];
    //get the empirical null
    nullMeanStd[0] = kernel.getMedian();
    nullMeanStd[1] = kernel.getIqr() / 1.349f;
    return nullMeanStd;
  }
  
}
