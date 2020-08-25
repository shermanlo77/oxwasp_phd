package uk.ac.warwick.sip.empiricalnullfilter;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

class RankFilters extends EmpiricalNullFilter {

  private int[] nFinite;

  public RankFilters() {
    this.outputImagePointer = STD + Q1 + Q2 + Q3;
  }

  public void filter() {
    this.nFinite = new int[this.imageProcessor.getPixelCount()];
    super.filter();
  }

  @Override
  protected void updatePixelInImage(float [] values, int valuesP, float [] nullMeanStd) {
    //do nothing
  }

  @Override
  protected void updateOutputImages(float[][] values, int valuesP, float[] nullMeanStd,
      Kernel kernel) {
    super.updateOutputImages(values, valuesP, nullMeanStd, kernel);
    this.nFinite[valuesP] = kernel.getNFinite();
  }

  @Override
  protected float[] getNullMeanStd(float initialValue, Kernel kernel, NormalDistribution normal,
      RandomGenerator rng) throws ConvergenceException{
    float[] dummy = new float[1];
    dummy[0] = initialValue;
    return dummy;
  }

  public int[] getNFinite() {
    return this.nFinite;
  }

}
