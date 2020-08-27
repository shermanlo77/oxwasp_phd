/**RANK FILTERS: Used by GPU versions of the filters
 *
 * A dummy class to get the std, median and quantile filters using the CPU.
 * Also keep tracks of the number of finite pixels in a kernel, see nFinite
 */

package uk.ac.warwick.sip.empiricalnullfilter;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.RandomGenerator;

class RankFilters extends EmpiricalNullFilter {

  private int[] nFinite; //number of finite pixels in the kernel (when centred for each pixel)

  public RankFilters() {
    this.outputImagePointer = STD + Q1 + Q2 + Q3;
  }

  @Override
  public void filter() {
    this.nFinite = new int[this.imageProcessor.getPixelCount()];
    super.filter();
  }

  @Override
  protected void updatePixelInImage(float [] values, int valuesP, float [] nullMeanStd) {
    //do nothing
    //only used for perparing GPU filters
  }

  //override to in addition count the number of finite pixels in the kernel
  @Override
  protected void updateOutputImages(float[][] values, int valuesP, float[] nullMeanStd,
      Kernel kernel) {
    super.updateOutputImages(values, valuesP, nullMeanStd, kernel);
    this.nFinite[valuesP] = kernel.getNFinite();
  }

  //override to not do empirical null filter (from superclass)
  //the method updatePixelInImage was overridden to ensure these dummy results are handled
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
