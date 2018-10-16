import org.apache.commons.math3.distribution.NormalDistribution;

public class EmpiricalNull {
  
  protected static int nStep = 10; //number of steps in newton-raphson
  //stopping condition tolerance for newton-raphson
  protected static float tolerance = (float) Math.pow(10.0, -5.0);
  //the bandwidth for the density estimate is B x 0.9 x std x n^{-1/5} + A
  //A and B are set below
  protected static float bandwidthParameterA = (float) 0.16; //intercept
  protected static float bandwidthParameterB = (float) 0.9; //gradient
  
  protected float [] cache; //array of greyvalues
  protected int x; //x position
  protected int [] cachePointers; //array of integer pairs, pointing to the boundary of the kernel
  
  protected int n = 0; //number of non-NaN data in the kernel
  protected float nullMean; //empirical null mean
  protected float nullStd; //empirical null std
  protected float bandwidth; //bandwidth for the density estimate
  protected float initialValue; //initial value for the newton-raphson
  protected NormalDistribution normalDistribution; //standard normal distribution
  
  /**CONSTRUCTOR
   * @param cache array of greyvalues
   * @param x x position
   * @param cachePointers array of integer pairs, pointing to the boundary of the kernel
   * @param initialValue initial value for the newton-raphson
   */
  public EmpiricalNull(float[] cache, int x, int[] cachePointers , float initialValue,
      NormalDistribution normalDistribution) {
    this.cache = cache;
    this.x = x;
    this.cachePointers = cachePointers;
    this.initialValue = initialValue;
    this.normalDistribution = normalDistribution;
    this.countData();
  }
  
  
  
  /**METHOD: COUNT DATA
   * Count the number of non-NaN greyvalues in the kernel
   * @return  number of non-NaN greyvalues in the kernel
   */
  public void countData() {
    for (int kk=0; kk<this.cachePointers.length; kk++) {
      for (int p=this.cachePointers[kk++]+x; p<=this.cachePointers[kk]+x; p++) {
        if (!Float.isNaN(this.cache[p])) {
          this.n++;
        }
      }
    }
  }
  
  /**METHOD: GET D LN DENSITY
   * Return a 2 element array containing:
   * 0. the first derivative of the log density
   * 1. the second derivative of the log density
   * @param greyValue the value of the derivative to be evaluated at
   * @return 2 element array containing derivatives
   */
  public float [] getDLnDensity(float greyValue) {
    
    //declare array for storing 3 sums where
    //z = (x - x_i) / h where x is the point of evaluation, x_i is a data point, h is the bandwidth
    //0. sum phi(z)
    //1. sum phi(z)z
    //2. sum phi(z)z^2
    //where phi is the Gaussian pdf
    float [] sumKernel = new float [3];
    float z;
    float phiZ;
    
    //for each non-NaN pixel in the kernel
    for (int kk=0; kk<this.cachePointers.length; kk++) {
      for (int p=this.cachePointers[kk++]+x; p<=this.cachePointers[kk]+x; p++) {
        if (!Float.isNaN(this.cache[p])) {
          
          //get phi(z)
          z = (cache[p] - greyValue) / this.bandwidth;
          phiZ = (float) this.normalDistribution.density((double) z);
          //update the sum of the kernels
          sumKernel[0] += phiZ;
          sumKernel[1] += phiZ * z;
          sumKernel[2] += phiZ * z * z;
          
        }
      }
    }
    
    //declare array for storing the 2 derivatives of the log density
    //0. 1st derivative
    //1. 2nd derivative
    float [] dxLnF = new float[2];
    dxLnF[0] = sumKernel[1]/(this.bandwidth*sumKernel[0]);
    dxLnF[1] = (sumKernel[0]*(sumKernel[2] - sumKernel[0]) - sumKernel[1]*sumKernel[1]) 
        / ((float)Math.pow((double)(this.bandwidth*sumKernel[0]),2.0));
    
    return dxLnF;
  }
  
}
