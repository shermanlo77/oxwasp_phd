package uk.ac.warwick.sip.empiricalnullfilter;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

class RankFilters extends EmpiricalNullFilter {

  public RankFilters() {
    this.outputImagePointer = STD + Q1 + Q2 + Q3;
  }

  @Override
  protected Cache instantiateCache() {
    return new CacheReflect(this.imageProcessor, this.roi);
  }

  @Override
  protected void updatePixelInImage(float [] values, int valuesP, float [] nullMeanStd) {
    //do nothing
  }

  @Override
  protected float[] getNullMeanStd(float initialValue, Kernel kernel, NormalDistribution normal,
      RandomGenerator rng) throws ConvergenceException{
    float[] dummy = new float[1];
    dummy[0] = initialValue;
    return dummy;
  }

}
