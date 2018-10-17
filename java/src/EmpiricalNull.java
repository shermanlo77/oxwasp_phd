import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.MersenneTwister;

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
  protected float dataStd; //the standard deviation of the pixels in the kernel
  
  protected float nullMean; //empirical null mean
  protected float nullStd; //empirical null std
  protected float bandwidth; //bandwidth for the density estimate
  protected NormalDistribution normalDistribution; //standard normal distributionrng = rng;
  protected MersenneTwister rng; //random number generator when a random initial value is needed
  
  /**CONSTRUCTOR
   * @param cache array of greyvalues
   * @param x x position
   * @param cachePointers array of integer pairs, pointing to the boundary of the kernel
   * @param initialValue initial value for the newton-raphson
   * @param dataStd standard deviation of the pixels in the kernel
   * @param normalDistribution standard normal distribution object
   * @param rng random number generator when a random initial value is needed
   */
  public EmpiricalNull(float[] cache, int x, int[] cachePointers , float initialValue,
      float dataStd, NormalDistribution normalDistribution, MersenneTwister rng) {
    this.cache = cache;
    this.x = x;
    this.cachePointers = cachePointers;
    this.nullMean = initialValue;
    this.dataStd = dataStd;
    this.normalDistribution = normalDistribution;
    this.rng = rng;
    this.countData();
    this.bandwidth = EmpiricalNull.bandwidthParameterB * dataStd
        * ((float) Math.pow((double) this.n, -0.2))
        + EmpiricalNull.bandwidthParameterA;
  }
  
  /**METHOD: ESTIMATE NULL
   * Estimate the parameters nullMean and nullStd
   */
  public void estimateNull() {
    //this.findMode returns a boolean, true if found a valid solution
    while (this.findMode()) {
      //if this.findMode failed to find a valid solution, change the initial value
      this.setNullToRandomData();
    }
  }
  
  /**METHOD: SET NULL TO RANDOM DATA
   * Set the initial value for the newton-raphson method to a random data point
   */
  public void setNullToRandomData() {
    //get a random pointer to a data
    int dataPointer = this.rng.nextInt(this.n);
    //count through the data and retrieve the random starting point
    int nCounter = 0;
    for (int kk=0; kk<this.cachePointers.length; kk++) {
      for (int p=this.cachePointers[kk++]+x; p<=this.cachePointers[kk]+x; p++) {
        if (!Float.isNaN(this.cache[p])) {
          //if found the random data, set it to nullMean and break out of all loops
          if (dataPointer == nCounter) {
            this.nullMean = this.cache[p];
            kk = this.cachePointers.length-1;
            p = this.cachePointers[kk];
          }
          nCounter ++;
        }
      }
    }
  }
  
  /**METHOD: FIND MODE
   * Find the mode of the log density using the newton-raphson method
   * Solution is stored in this.nullMean
   * @return true if the solution is valid
   */
  public boolean findMode() {
    //get the 1st and 2nd diff of the ln density at the initial value
    float [] dxLnF = this.getDLnDensity(this.nullMean);
    //for n_step
    for (int i=0; i<EmpiricalNull.nStep; i++) {
      //update the solution to the mode
      this.nullMean -= dxLnF[0]/dxLnF[1];
      //get the 1st and 2nd diff of the ln density at the new value
      dxLnF = this.getDLnDensity(this.nullMean);
      //if this gradient is within tolerance, break the i_step for loop
      if (Math.abs(dxLnF[0])<EmpiricalNull.tolerance) {
        break;
      }
      //if any of the variables are nan, break the loop as well
      if (Float.isNaN(dxLnF[0])) {
        return false;
      } else if (Float.isNaN(dxLnF[1])) {
        return false;
      } else if (Float.isNaN(this.nullMean)) {
        return false;
      }
    }
    //check if the solution to the mode is a maxima by looking at the 2nd diff
    //return true is solution is valid and work out the null std
    if (dxLnF[1] < 0) {
      this.nullStd = (float) Math.pow((double) dxLnF[1], -0.5);
      return true;
    } else {
      return false;
    }
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
