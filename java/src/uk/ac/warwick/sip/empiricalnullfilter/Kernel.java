package uk.ac.warwick.sip.empiricalnullfilter;

import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import ij.gui.Roi;

//CLASS: KERNEL
/**A circular kernel, contains statistics and methods using the pixels captured by the circular
 *     kernel. The kernel can move right or start on a new row, the member variables are updated
 *     when the kernel moves
 * <p>
 * Modified from the RankFilters.java
 *     See https://github.com/imagej/ImageJA/blob/master/src/main/java/ij/plugin/filter/
 *         RankFilters.java
 * <p>
 * How to use:
 *   <ul>
 *     <li>Pass the required parameters into the constructor</li>
 *     <li>Call the method moveToNewLine(int y) to move the kernel onto a line</li>
 *     <li>Call the method moveRight() to move the kernel one pixel to the right</li>
 *     <li>Call the getter methods to get required kernel statistics</li>
 *   </ul>
 */
class Kernel {
  
  //=====STATIC VARIABLES=====//
  private static boolean isSmallKernel; //indicate if this kernel is small of not
  private static int kNPoints; //number of points in the kernel
  private static int kRadius; //radius of the kernel
  private static int kHeight; //height of the kernel (2*kRadius + 1);
  //pairs of pointers to be used in iteration, see method setKernel
  private static int [] kernelPointer;
  
  //=====MEMBER VARIABLES=====//
  private final double[] sums; //[0] sum of greyvalues, [1] sum of greyvalues squared
  private final float[] pixels; //array of pixel greyvalues in the kernel
  private float mean;
  private float std;
  private float[] quartiles; //[0,1,2] 1st 2nd and 3rd quartiles greyvalues of pixels in the kernel
  private boolean isFullCalculation; //true to do full sum
  private int x; //x position of kernel, increments with the method moveKernel()
  private int y; //y position of kernel, set using method moveToNewLine()
  private int previousY;
  private int nFinite;
  private boolean isFinite; //false if the kernel is centred on a non-ROI pixel or NaN calculations
  
  //indicate if to copy pixels to the member variable pixels
  //must be true if isQuartile is true
  private boolean isCopy;
  private boolean isMeanStd; //indicate if to calculate sums
  private boolean isQuartile; //indicate if to do quartile calculations
  
  private final Cache cache; //contains copy of pixels of the orginial image
  //pairs of integers which points to the cache according to the shape of the kernel
  private final int[] cachePointers;
  Roi roi; //region of interest
  
  //CONTRUCTOR
  /**@param cache The kernel is placed on this Cache
   * @param roi Region of interest of the image being filtered
   * @param isCopy true if the kernel to copy pixels from cache
   * @param isMeanStd true if to work out mean and standard deviation
   * @param isQuartile true if to work out quartiles
   */
  public Kernel(Cache cache, Roi roi, boolean isCopy, boolean isMeanStd, boolean isQuartile) {
    this.cache = cache;
    
    //make the cache points
    //pairs of integers which points the start and end of the current row to the cache according to
        //the shape of the kernel
    this.cachePointers = new int[2*Kernel.getKHeight()];
    for (int i=0; i<Kernel.getKHeight(); i++) {
      cachePointers[2*i] =
          i*cache.getCacheWidth()+Kernel.getKRadius() + Kernel.getKernelPointer()[2*i];
      cachePointers[2*i+1] =
          i*cache.getCacheWidth()+Kernel.getKRadius() + Kernel.getKernelPointer()[2*i+1];
    }
    this.roi = roi;
    
    this.isQuartile = isQuartile;
    this.isMeanStd = isMeanStd;
    //pixels must be copied when working out quartiles
    if (isQuartile) {
      this.isCopy = true;
    } else {
      this.isCopy = isCopy;
    }
    
    //arrays to store calculations of pixels contained in a kernel
    this.sums = new double[2];
    this.pixels = new float[kNPoints]; //stores greyvalues of pixels in a kernel
    
    this.quartiles = new float[3];
    this.isFullCalculation = true;
    
    this.previousY = Kernel.getKHeight()/2-cache.getCacheHeight();
    
  }
  
  //METHOD: MOVE TO NEW LINE
  /**Position the kernel centred at (x=0, y)
   * Member variables are updated
   * @param y Row number
   */
  public void moveToNewLine(int y) {
    this.y = y;
    this.x = 0;
    this.isFullCalculation = true;
    //shift the cachePointers
    for (int i=0; i<this.cachePointers.length; i++) {  //shift kernel pointers to new line
      this.cachePointers[i] = (this.cachePointers[i] +
          this.cache.getCacheWidth()*(y-this.previousY))%this.cache.getCache().length;
    }
    this.previousY = y;
    this.updateStatistics();
  }
  
  //METHOD: MOVE RIGHT
  /**Move the kernel one pixel to the right
   * Member variables are updated
   * @return true if the move was successful (ie within the roi bounding box)
   */
  public boolean moveRight() {
    if (this.x<this.roi.getBounds().width) {
      this.x++;
      this.updateStatistics();
      return true;
    } else {
      return false;
    }
  }
  
  //METHOD: UPDATE STATISTICS
  /**Update the following member variables: sums, pixels, mean, std, quartiles
   */
  private void updateStatistics() {
    //if this pixel is not in the roi, for the next pixel do a full calculation as the summation
        //cannot be propagate
    //else this pixel is in the roi and filter this pixel
    if (!this.roi.contains(this.roi.getBounds().x+x, y)) {
      this.isFullCalculation = true;
      this.isFinite = false;
    } else {
      this.isFinite = true;
    }
    
    //if centred on finite pixel
    if (this.isFinite) {
      
      //if request for mean and std calculations
      if (this.isMeanStd) {
        
        if (this.isFullCalculation) {
          //for small kernel, always use the full area, not incremental algorithm
          this.isFullCalculation = isSmallKernel;
          this.sumArea();
        } else {
          this.sumSides();
          //avoid perpetuating NaNs into remaining line
          if (Double.isNaN(this.sums[0])) {
            this.isFullCalculation = true;
            this.isFinite = false;
          }
        }
        
        //at least 2 data points are required for standard deviation calculations
        if (this.nFinite < 2) {
          this.isFullCalculation = true;
          this.isFinite = false;
        } else {
          //calculate the mean and standard deviation
          this.mean = (float) (this.sums[0]/this.nFinite);
          this.std = (float) Math.sqrt(((this.sums[1] - this.sums[0]*this.sums[0]/this.nFinite) 
              / (this.nFinite-1)));
        }
      }
      
      //if request to copy pixels from cache to kernel
      if (this.isCopy) {
        this.copyPixels();
      }
      
      //if request for quartile calculations
      if (this.isQuartile) {
        if (this.nFinite < 2) {
          this.isFinite = false;
        } else {
          this.calculateQuartiles();
        }
      }
    }
  }
  
  //METHOD: GET AREA SUMS
  /**Get sum of values and values squared within the kernel area.
   * x between 0 and cacheWidth-1
   * Output is written to array sums[0] = sum; sums[1] = sum of squares
   * Ignores nan
   */
  //MODIFIES nFinite
  private void sumArea() {
    this.sums[0] = 0;
    this.sums[1] = 0;
    this.nFinite = 0;
    //y within the cache stripe (we have 2 kernel pointers per cache line)
    for (int kk=0; kk<this.cachePointers.length; kk++) {
      for (int p=this.cachePointers[kk++]+this.x; p<=this.cachePointers[kk]+this.x; p++) {
        double greyvalue = this.cache.getCache()[p];
        if (!Double.isNaN(greyvalue)) {
          this.sums[0] += greyvalue;
          this.sums[1] += greyvalue*greyvalue;
          this.nFinite++;
        }
      }
    }
  }
  
  //METHOD: ADD SIDE SUMS
  /**Add all values and values squared at the right border inside minus at the left border outside
   * the kernal area.
   * Output is added or subtracted to/from array sums[0] += sum; sums[1] += sum of squares  when at
   * the right border, minus when at the left border
   */
  private void sumSides() {
    //for each row
    for (int kk=0; kk<this.cachePointers.length; /*k++;k++ below*/) {
      //this value is not in the kernel area any more
      double greyvalue = this.cache.getCache()[this.cachePointers[kk++]+(this.x-1)];
      if (!Double.isNaN(greyvalue)) {
        this.sums[0] -= greyvalue;
        this.sums[1] -= greyvalue*greyvalue;
        this.nFinite--;
      }
      //this value comes into the kernel area
      greyvalue = this.cache.getCache()[this.cachePointers[kk++]+this.x];
      if (!Double.isNaN(greyvalue)) {
        this.sums[0] += greyvalue;
        this.sums[1] += greyvalue*greyvalue;
        this.nFinite++;
      }
    }
  }
  
  //METHOD: COPY PIXELS
  /**Deep copy pixels captured by the kernel from the cache
   */
  private void copyPixels() {
    this.nFinite = 0;
    for (int kk=0; kk<this.cachePointers.length; kk++) {
      for (int p=this.cachePointers[kk++]+this.x; p<=this.cachePointers[kk]+this.x; p++) {
        float greyvalue = this.cache.getCache()[p];
        if (!Float.isNaN(greyvalue)) {
          this.pixels[this.nFinite] = greyvalue;
          this.nFinite++;
        }
        
      }
    }
  }
  
  /**METHOD: GET QUARTILES
   * Get the quartiles of values within kernel-sized neighborhood.
   * nan values are ignored
   */
  private void calculateQuartiles() {
    Percentile percentile = new Percentile();
    //convert float to double
    double[] pixels = new double[this.pixels.length];
    for (int i=0; i<this.nFinite; i++) {
      pixels[i] = (double) this.pixels[i];
    }
    percentile.setData(pixels, 0, this.nFinite);
    for (int i=0; i<3; i++) {
      this.quartiles[i] = (float) percentile.evaluate((i+1) * 25.0);
    }
  }
  
  public int getY() {
    return this.y;
  }
  
  public float getMean() {
    return this.mean;
  }
  
  public float getStd() {
    return this.std;
  }
  
  public int getNFinite() {
    return this.nFinite;
  }
  
  public boolean isFinite() {
    return this.isFinite;
  }
  
  public float getMedian() {
    return this.quartiles[1];
  }
  
  public float[] getQuartiles() {
    return this.quartiles;
  }
  
  public float getIqr() {
    return this.quartiles[2] - this.quartiles[0];
  }
  
  public int getX() {
    return this.x;
  }
  
  public float[] getPixels() {
    return this.pixels;
  }
  
  public int[] getCachePointers() {
    return this.cachePointers;
  }
  
  //FUNCTION: MAKE LINE RADII
  /**Set the static variables given the kernel radius: isSmallKernel, kNPoints, kRadius, kHeight.
   *     kernelPointer
   * kernelPointer: the output is an array that gives the length of each line of the structuring
   *     element (kernel) to the left (negative) and to the right (positive):
   *         [0] left in line 0, [1] right in line 0,
   *         [2] left in line 2, ...
   *     Array elements at the end:
   *         length-2: nPoints, number of pixels in the kernel area
   *         length-1: kernelRadius in x direction (kernel width is 2*kernelRadius+1)
   * Kernel height: (array length - 1)/2 (odd number);
   * Kernel radius: kernel height/2 (truncating integer division).
   * Note that kernel width and height are the same for the circular kernels used here,
   * but treated separately for the case of future extensions with non-circular kernels.
   * e.g. r=0.5 will return [0,0,-1,1,0,0,nPoints, kernelRadius]
   * e.g. r=3 will return [-1,1,-2,2,-3,3,-3,3,-3,3,-2,2,-1,1,nPoints, kernelRadius]
   * @param radius of the kernel
   *     Radius = 0.5 includes the 4 neighbours of the pixel in the centre,
   *     radius = 1 corresponds to a 3x3 kernel
   */
  public static void setKernel(double radius) {
    if (radius>=1.5 && radius<1.75) {//this code creates the same sizes as the previous RankFilters
      radius = 1.75;
    } else if (radius>=2.5 && radius<2.85) {
      radius = 2.85;
    }
    int r2 = (int) (radius*radius) + 1;
    kRadius = (int)(Math.sqrt(r2+1e-10));
    kHeight = 2*kRadius + 1;
    kernelPointer = new int[2*kHeight];
    kernelPointer[2*kRadius] = -kRadius;
    kernelPointer[2*kRadius+1] =  kRadius;
    kNPoints = 2*kRadius+1;
    for (int y=1; y<=kRadius; y++) { //lines above and below centre together
      int dx = (int)(Math.sqrt(r2-y*y+1e-10));
      kernelPointer[2*(kRadius-y)] = -dx;
      kernelPointer[2*(kRadius-y)+1] =  dx;
      kernelPointer[2*(kRadius+y)] = -dx;
      kernelPointer[2*(kRadius+y)+1] =  dx;
      kNPoints += 4*dx+2; //2*dx+1 for each line, above&below
    }
    isSmallKernel = kRadius < 2; //indicate if this kernel is small
  }

  public static boolean getIsSmallKernel() {
    return isSmallKernel;
  }

  public static int getKNPoints() {
    return kNPoints;
  }

  public static int getKRadius() {
    return kRadius;
  }

  public static int getKHeight() {
    return kHeight;
  }

  public static int[] getKernelPointer() {
    return kernelPointer;
  }
  
}
