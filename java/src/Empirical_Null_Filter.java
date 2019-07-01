import ij.IJ;
import uk.ac.warwick.sip.empiricalnullfilter.EmpiricalNullFilter;

public class Empirical_Null_Filter extends EmpiricalNullFilter{
  
  /**METHOD: SHOW PROGRESS
   * Use ImageJ to show progress bar
   * @param percent
   */
  @Override
  protected void showProgress(double percent) {
    int nPasses2 = nPasses;
    percent = (double)pass/nPasses2 + percent/nPasses2;
    IJ.showProgress(percent);
  }
  
  public static void main(String[] args){
    System.out.println("MIT License\nCopyright (c) 2019 Sherman Lo");
  }
}
