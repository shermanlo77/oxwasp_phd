//MIT License
//Copyright (c) 2019 Sherman Lo

import ij.IJ;
import uk.ac.warwick.sip.empiricalnullfilter.ModeFilterGpu;

//EMPIRICAL NULL FILTER
/**
 * @author Sherman Lo
 */
public class Mode_Filter_GPU extends ModeFilterGpu{

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

  //MAIN
  public static void main(String[] args){
    System.out.println("MIT License - please see LICENSE");
    System.out.println("Copyright (c) 2019 Sherman Lo");
    System.out.println("Please see https://github.com/shermanlo77/oxwasp_phd or README.md");
  }
}