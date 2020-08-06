//MIT License
//Copyright (c) 2019 Sherman Lo

package uk.ac.warwick.sip.empiricalnullfilter;

import java.awt.Rectangle;
import java.util.Arrays;

import ij.ImagePlus;
import ij.process.FloatProcessor;
import ij.gui.Roi;
import ij.process.ImageProcessor;

class CacheReflect extends Cache {

  //CONSTRUCTOR
  /**@param ip The image to be filtered
   * @param roi The image region of interest
   */
  public CacheReflect(ImageProcessor ip, Roi roi) {
    super(ip, roi);
  }

  @Override
  protected void copyToCache(float[] pixels, Rectangle roiRectangle, Roi roi) {

    int kRadius = Kernel.getKRadius();
    int width = roiRectangle.width;
    int height = roiRectangle.height;

    //for each row
    for (int iRow = 0; iRow < this.cacheHeight; iRow++) {

      //get the y coordinate
      int y = iRow - Kernel.getKRadius();

      //if y is outside the roi, fill from next rows (top and bottom padding)
      if (y < 0 || y >= roiRectangle.height) {
        int copyFrom;
        if (y<0) {
          copyFrom = (-y-1)*roiRectangle.width;
        } else {
          copyFrom = (2*roiRectangle.height-y-1)*roiRectangle.width;
        }
        int copyTo;
        for (int x=0; x<roiRectangle.width; x++) {
          copyTo = iRow*this.cacheWidth + Kernel.getKRadius();
          this.cache[copyTo+x] = pixels[copyFrom+x];
        }
      //else it is inside the roi
      } else {
        //copy pixels between the padding
        for (int x=0; x<roi.getBounds().width; x++) {
          float greyvalue = Float.NaN;
          if (roi.contains(x+roi.getBounds().x, y+roi.getBounds().y)){
            greyvalue = pixels[(y+roi.getBounds().y)*width + x+roi.getBounds().x];
          }
          this.cache[iRow*this.cacheWidth+Kernel.getKRadius()+x] = greyvalue;
        }
      }

      int paddingIndex;

      //left padding
      paddingIndex = iRow*this.cacheWidth+Kernel.getKRadius();
      for (int x=0; x<Kernel.getKRadius(); x++) {
        this.cache[paddingIndex-x-1] = this.cache[paddingIndex+x];
      }
      //right padding
      paddingIndex = (iRow+1)*this.cacheWidth-Kernel.getKRadius();
      for (int x=Kernel.getKRadius()-1; x>=0; x--) {
        this.cache[paddingIndex+x] = this.cache[paddingIndex-x-1];
      }
    }
  }
}
