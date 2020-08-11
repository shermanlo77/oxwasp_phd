//MIT License
//Copyright (c) 2019 Sherman Lo

import ij.IJ;
import uk.ac.warwick.sip.empiricalnullfilter.EmpiricalNullFilterGpu;

//EMPIRICAL NULL FILTER
/**
 * @author Sherman Lo
 */
public class Empirical_Null_Filter_GPU extends EmpiricalNullFilterGpu{

  //METHOD: SHOW PROGRESS
  /**Use ImageJ to show progress bar
   * @param percent
   */
  @Override
  protected void showProgress(double percent) {
    int nPasses2 = nPasses;
    percent = (double)pass/nPasses2 + percent/nPasses2;
    IJ.showProgress(percent);
  }

}
