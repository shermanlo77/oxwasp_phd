//MIT License
//Copyright (c) 2019 Sherman Lo

package uk.ac.warwick.sip.empiricalnullfilter;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.RandomGenerator;

//CLASS: MEAN VAR NULL FILTER
/**Subclass of EmpiricalNullFilter, replaces the empirical null mean with mean, replaces the 
 *     empirical null variance with variance.
 * @author Sherman Lo
 */
public class MeanVarNullFilter extends EmpiricalNullFilter {
  
  public MeanVarNullFilter() {
    super();
    //this filter does not need to copy kernel pixels or work out quantiles
    this.isKernelCopy = false;
    this.isKernelQuartile = false;
  }
  
  @Override
  protected float[] getNullMeanStd(float initialValue, Kernel kernel, NormalDistribution normal,
      RandomGenerator rng) {
  //declare 2 vector to store the null mean and null std
    float[] nullMeanStd = new float[2];
    //get the empirical null
    nullMeanStd[0] = kernel.getMean();
    nullMeanStd[1] = kernel.getStd();
    return nullMeanStd;
  }
}
