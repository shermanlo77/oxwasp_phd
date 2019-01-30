package uk.ac.warwick.sip.empiricalnullfilter;

import java.awt.Rectangle;

import org.apache.commons.math3.stat.descriptive.rank.Percentile;

import ij.gui.Roi;

public class Kernel {
  
  private double[] sums; //[0] sum of greyvalues, [1] sum of greyvalues squared
  private double[] pixels; //array of pixel greyvalues in the kernel
  private float mean;
  private float std;
  private float[] quartiles; //[0,1,2] 1st 2nd and 3rd quartiles greyvalues of pixels in the kernel
  private boolean isSmallKernel; //indicate if this kernel is small of not
  private boolean isFullCalculation; //true to do full sum
  private int x; //x position of kernel, increments with the method moveKernel()
  private int y; //y position of kernel, fixed
  private int width;
  private int nFinite;
  private boolean isFinite;
  
  private boolean isMeanStd; //indicate if to calculate sums
  private boolean isQuartile; //indicate if to do quartile calculations
  
  private float[] cache; //contains copy of pixels of the orginial image
  //pairs of integers which points to the cache according to the shape of the kernel
  private int[] cachePointers;
  Roi roi; //region of interest
  
  /**CONSTRUCTOR
   * 
   * @param y
   * @param cache
   * @param cachePointers
   * @param kNPoints
   * @param kRadius
   * @param roi
   * @param isMeanStd
   * @param isQuartile
   */
  public Kernel(int y, float[] cache, int[] cachePointers, int kNPoints, float kRadius, Roi roi,
      int width, boolean isMeanStd, boolean isQuartile) {
    
    this.x = 0;
    this.y = y;
    this.cache = cache;
    this.cachePointers = cachePointers;
    this.roi = roi;
    this.width  = width;
    
    this.isMeanStd = isMeanStd;
    this.isQuartile = isQuartile;
    
    //arrays to store calculations of pixels contained in a kernel
    this.sums = new double[2];
    this.pixels = new double[kNPoints]; //stores greyvalues of pixels in a kernel
    
    this.quartiles = new float[3];
    this.isSmallKernel = kRadius < 2; //indicate if this kernel is small
    this.isFullCalculation = true;
    
    this.updateStatistics();
    
  }
  
  public boolean moveRight() {
    if (this.x<this.roi.getBounds().width) {
      this.x++;
      this.updateStatistics();
      return true;
    } else {
      return false;
    }
  }
  
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
    
    if (this.isFinite) {
      if (this.isMeanStd) {
        if (this.isFullCalculation) {
          //for small kernel, always use the full area, not incremental algorithm
          this.isFullCalculation = this.isSmallKernel;
          this.sumArea();
        } else {
          this.sumSides();
          //avoid perpetuating NaNs into remaining line
          if (Double.isNaN(this.sums[0])) {
            this.isFullCalculation = true;
            this.isFinite = false;
          }
        }
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
      
      if (this.isQuartile) {
        this.copyPixels();
        if (this.nFinite < 2) {
          this.isFinite = false;
        } else {
          this.calculateQuartiles();
        }
      }
    }
  }
  
  /**METHOD: GET AREA SUMS
   * Get sum of values and values squared within the kernel area.
   * x between 0 and cacheWidth-1
   * Output is written to array sums[0] = sum; sums[1] = sum of squares
   * Ignores nan
   * Returns the number of non-nan numbers
   * @param cache
   * @param xCache0
   * @param kernel
   * @param sums modified
   * @return  number of non-nan numbers
   */
  private void sumArea() {
    this.sums[0] = 0;
    this.sums[1] = 0;
    this.nFinite = 0;
    //y within the cache stripe (we have 2 kernel pointers per cache line)
    for (int kk=0; kk<this.cachePointers.length; kk++) {
      for (int p=this.cachePointers[kk++]+this.x; p<=this.cachePointers[kk]+this.x; p++) {
        double greyvalue = cache[p];
        if (!Double.isNaN(greyvalue)) {
          this.sums[0] += greyvalue;
          this.sums[1] += greyvalue*greyvalue;
          this.nFinite++;
        }
      }
    }
  }
  
  /**METHOD: ADD SIDE SUMS
   * Add all values and values squared at the right border inside minus at the left border outside
   * the kernal area.
   * Output is added or subtracted to/from array sums[0] += sum; sums[1] += sum of squares  when at
   * the right border, minus when at the left border
   * @param cache
   * @param xCache0
   * @param kernel
   * @param sums modified
   * @param nData
   * @return number of non-nan numbers
   */
  private void sumSides() {
    //for each row
    for (int kk=0; kk<this.cachePointers.length; /*k++;k++ below*/) {
      double greyvalue = this.cache[this.cachePointers[kk++]+(this.x-1)]; //this value is not in the kernel area any more
      if (!Double.isNaN(greyvalue)) {
        this.sums[0] -= greyvalue;
        this.sums[1] -= greyvalue*greyvalue;
        this.nFinite--;
      }
      greyvalue = this.cache[this.cachePointers[kk++]+this.x]; //this value comes into the kernel area
      if (!Double.isNaN(greyvalue)) {
        this.sums[0] += greyvalue;
        this.sums[1] += greyvalue*greyvalue;
        this.nFinite++;
      }
    }
  }
  
  private void copyPixels() {
    this.nFinite = 0;
    for (int kk=0; kk<this.cachePointers.length; kk++) {
      for (int p=this.cachePointers[kk++]+this.x; p<=this.cachePointers[kk]+this.x; p++) {
        float greyvalue = cache[p];
        if (!Float.isNaN(greyvalue)) {
          this.pixels[this.nFinite] = (double) greyvalue;
          this.nFinite++;
        }
        
      }
    }
  }
  
  /**METHOD: GET QUARTILES
   * Get the quartiles of values within kernel-sized neighborhood.
   * nan values are ignored
   * @param cache
   * @param xCache0
   * @param kernel
   * @param quartileBuf
   * @param kNPoints
   * @param quartiles modified
   */
  private void calculateQuartiles() {
    Percentile percentile = new Percentile();
    percentile.setData(this.pixels, 0, this.nFinite);
    for (int i=0; i<3; i++) {
      this.quartiles[i] = (float) percentile.evaluate((i+1) * 25.0);
    }
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

  public float[] getQuartiles() {
    return this.quartiles;
  }
  
  public int getX() {
    return this.x;
  }
  
}
