//MIT License
//Copyright (c) 2019 Sherman Lo

package uk.ac.warwick.sip.empiricalnullfilter;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.RandomGenerator;

//CLASS: MAD MODE NULL FILTER
/**Subclass of EmpiricalNullFilter, replaces the empirical null std with median around the mode.
 * @author Sherman Lo
 */
public class MadModeNullFilter extends EmpiricalNullFilter {
  
  public MadModeNullFilter() {
    super();
  }
  
  @Override
  protected float[] getNullMeanStd(float initialValue, Kernel kernel, NormalDistribution normal,
      RandomGenerator rng) throws ConvergenceException{
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
