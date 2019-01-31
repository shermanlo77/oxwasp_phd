package uk.ac.warwick.sip.empiricalnullfilter;

import java.awt.Rectangle;
import java.util.Arrays;

import ij.gui.Roi;
import ij.process.ImageProcessor;

public class Cache {
  
  private int highestYInCache;
  
  final private ImageProcessor ip;
  final private Roi roi;
  final private float[] cache;
  final private int cacheWidth;
  final private int cacheHeight;
  final private int xMin;
  final private int xMax;
  final private int padLeft;
  final private int padRight;
  final private int xMinInside;
  final private int xMaxInside;
  final private int widthInside;
  final private boolean isMultiThread;
  
  private boolean copyingToCache = false;
  
  public Cache(int numThreads, ImageProcessor ip, Roi roi) {
    
    this.ip = ip;
    this.roi = roi;
    Rectangle roiRectangle = ip.getRoi();
    int width = ip.getWidth();
    
    //get properties of the kernel and the cache
    this.cacheWidth = roiRectangle.width+2*Kernel.getKRadius();
    this.cacheHeight = Kernel.getKHeight() + (numThreads>1 ? 2*numThreads : 0);
    //'cache' is the input buffer. Each line y in the image is mapped onto cache line y%cacheHeight
    this.cache = new float[cacheWidth*cacheHeight];
    //this line+1 will be read into the cache first
    this.highestYInCache = Math.max(roiRectangle.y-Kernel.getKHeight()/2, 0) - 1;
    
    //get the boundary
    this.xMin = roiRectangle.x - Kernel.getKRadius();
    this.xMax = roiRectangle.x + roiRectangle.width + Kernel.getKRadius();
    
    //pad out the image, eg when the kernel is on the boundary of the image
    this.padLeft = this.xMin<0 ? -this.xMin : 0;
    this.padRight = this.xMax>width? this.xMax-width : 0;
    this.xMinInside = this.xMin>0 ? this.xMin : 0;
    this.xMaxInside = this.xMax<width ? this.xMax : width;
    this.widthInside = this.xMaxInside - this.xMinInside;
    
    this.isMultiThread = numThreads > 1;
  }
  
  public void readIntoCache(ImageProcessor ip, int[] yForThread, Kernel kernel) {
    int y = kernel.getY();
    Rectangle roiRectangle = ip.getRoi();
    if (!this.isMultiThread) {
      int yStartReading = y==roiRectangle.y ? Math.max(roiRectangle.y-Kernel.getKHeight()/2, 0) : y+Kernel.getKHeight()/2;
      for (int yNew = yStartReading; yNew<=y+Kernel.getKHeight()/2; yNew++) { //only 1 line except at start
        this.readLineToCacheOrPad(yNew);
      }
    } else {
      if (!this.copyingToCache || this.highestYInCache < y+Kernel.getKHeight()/2) synchronized(cache) {
        this.copyingToCache = true; // copy new line(s) into cache
        while (this.highestYInCache < arrayMinNonNegative(yForThread) - Kernel.getKHeight()/2 + cacheHeight - 1) {
          int yNew = this.highestYInCache + 1;
          this.readLineToCacheOrPad(yNew);
          this.highestYInCache = yNew;
        }
        this.copyingToCache = false;
      }
    }
  }
  
  /**METHOD: READ LINE TO CACHE OR PAD
   * Read a line into the cache (including padding in x), anything outside the boundary is nan
   * @param pixels
   * @param width
   * @param height
   * @param roiY
   * @param xminInside
   * @param widthInside
   * @param cache modified
   * @param cacheWidth
   * @param cacheHeight
   * @param padLeft
   * @param padRight
   * @param kHeight
   * @param y
   */
  private void readLineToCacheOrPad(int y) {
    int lineInCache = y%this.cacheHeight;
    if (y < this.ip.getHeight()) {
      this.readLineToCache(y);
      if (y==0) {
        for (int prevY = this.ip.getRoi().y-Kernel.getKHeight()/2; prevY<0; prevY++) {  //for y<0, pad with nan
          int prevLineInCache = this.cacheHeight+prevY;
          Arrays.fill(cache, prevLineInCache*this.cacheWidth, prevLineInCache*this.cacheWidth + this.cacheWidth,
              Float.NaN);
        }
      }
    } else {
      Arrays.fill(this.cache, lineInCache*this.cacheWidth, lineInCache*this.cacheWidth + this.cacheWidth, Float.NaN);
    }
  }
  
  /**METHOD: READ LINE TO CACHE
   * Read a line into the cache (includes conversion to flaot)
   * Pad with nan if necessary
   * @param pixels
   * @param pixelLineP
   * @param y
   * @param xminInside
   * @param widthInside
   * @param cache modified
   * @param cacheLineP
   * @param padLeft
   * @param padRight
   */
  private void readLineToCache(int y) {
    
    float[] pixels = (float[]) this.ip.getPixels();
    int pixelLineP = y*this.ip.getWidth();
    int lineInCache = y%this.cacheHeight;
    int cacheLineP = lineInCache*this.cacheWidth;
    
    //for each pixel in the line
    for (int x=0; x<this.widthInside; x++) {
      //if this pixel is in the roi, copy it to the cache, else put nan in the cache
      float toCopytoCache;
      if (!this.roi.contains(this.xMinInside + x, y)) {
        toCopytoCache = Float.NaN;
      } else {
        toCopytoCache = pixels[pixelLineP+this.xMinInside + x];
      }
      this.cache[cacheLineP+this.padLeft+x] = toCopytoCache;
    }
    //Padding contains NaN
    for (int cp=cacheLineP; cp<cacheLineP+this.padLeft; cp++) {
      this.cache[cp] = Float.NaN;
    }
    for (int cp=cacheLineP+padLeft+widthInside; cp<cacheLineP+this.padLeft+this.widthInside+this.padRight; cp++) {
      this.cache[cp] = Float.NaN;
    }
  }
  
  /**METHOD: ARRAY MIN NON NEGATIVE
   * Used by thread control in threadFilter
   * @param array
   * @return the minimum of the array, but not less than 0
   */
  private static int arrayMinNonNegative(int[] array) {
    int min = Integer.MAX_VALUE;
    for (int i=0; i<array.length; i++) {
      if (array[i]<min) {
        min = array[i];
      }
    }
    return min<0 ? 0 : min;
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
