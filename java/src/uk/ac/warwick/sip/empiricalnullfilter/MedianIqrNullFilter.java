//MIT License
//Copyright (c) 2019 Sherman Lo

package uk.ac.warwick.sip.empiricalnullfilter;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.RandomGenerator;

//CLASS: MEDIAN IQR NULL FILTER
/**Subclass of EmpiricalNullFilter, replaces the empirical null mean with median, replaces the 
 *     empirical null std with iqr/1.3490.
 * @author Sherman Lo
 */
public class MedianIqrNullFilter extends EmpiricalNullFilter {
  
  public MedianIqrNullFilter() {
    super();
    //this filter does not need to work out the kernel mean and variance
    this.isKernelMeanVar = false;
  }
  
  @Override
  protected float[] getNullMeanStd(float initialValue, Kernel kernel, NormalDistribution normal,
      RandomGenerator rng) {
  //declare 2 vector to store the null mean and null std
    float[] nullMeanStd = new float[2];
    //get the empirical null
    nullMeanStd[0] = kernel.getMedian();
    nullMeanStd[1] = kernel.getIqr() / 1.349f;
    return nullMeanStd;
  }
  
}
