//MIT License
//Copyright (c) 2019 Sherman Lo

package uk.ac.warwick.sip.empiricalnullfilter;

import java.awt.Rectangle;
import java.util.Arrays;

import ij.gui.Roi;
import ij.process.ImageProcessor;

//CLASS: CACHE
/**A image containing a deep copy of the image to be filtered.
 * 
 * <p>Pixels are copied from the image to the cache.
 * 
 * <p>Modified from 
 *     <a href=https://github.com/imagej/ImageJA/blob/7f965b866c9db364b0b47140caeef4f62d5d8c15/src/main/java/ij/plugin/filter/RankFilters.java>
 *     RankFilters.java</a>
 * 
 * @author Sherman Lo
 */
class Cache {
  
  /**image to be working on*/
  private final ImageProcessor ip;
  /**contains deep copy of a section of the image*/
  private final float[] cache;
  /**width of cache*/
  private final int cacheWidth;
  /**height of cache*/
  private final int cacheHeight;
  
  //CONSTRUCTOR
  /**@param ip The image to be filtered
   * @param roi The image region of interest
   */
  public Cache(ImageProcessor ip, Roi roi) {
    
    this.ip = ip;
    Rectangle roiRectangle = ip.getRoi();
    int width = ip.getWidth();
    float[] pixels = (float[]) ip.getPixels();
    
    //get properties of the kernel and the cache
    this.cacheWidth = roiRectangle.width+2*Kernel.getKRadius();
    this.cacheHeight = roiRectangle.height+2*Kernel.getKRadius();
    //'cache' is the input buffer. Each line y in the image is mapped onto cache line y%cacheHeight
    this.cache = new float[cacheWidth*cacheHeight];
    
    //for each row
    for (int iRow = 0; iRow < this.cacheHeight; iRow++) {
      //get the y coordinate
      int y = iRow - Kernel.getKRadius();
      //if y is outside the roi, fill this row with NaN
      if ( y < 0 ||  y >= roiRectangle.height ) {
        Arrays.fill(this.cache, iRow*this.cacheWidth, (iRow+1)*this.cacheWidth, Float.NaN);
      //else it is inside the roi
      } else {
        //pad left and right with nan
        Arrays.fill(this.cache, iRow*this.cacheWidth,
            iRow*this.cacheWidth+Kernel.getKRadius(), Float.NaN);
        Arrays.fill(this.cache, (iRow+1)*this.cacheWidth-Kernel.getKRadius(),
            (iRow+1)*this.cacheWidth, Float.NaN);
        //copy pixels between the padding
        for (int x=0; x<roi.getBounds().width; x++){
          float greyvalue = Float.NaN;
          if (roi.contains(x+roi.getBounds().x, y+roi.getBounds().y)){
            greyvalue = pixels[(y+roi.getBounds().y)*width + x+roi.getBounds().x];
          }
          this.cache[iRow*this.cacheWidth+Kernel.getKRadius()+x] = greyvalue;
        }
        
      }
    }
    
  }
  
  public float[] getCache() {
    return this.cache;
  }

  public int getCacheWidth() {
    return this.cacheWidth;
  }

  public int getCacheHeight() {
    return this.cacheHeight;
  }
  
}
