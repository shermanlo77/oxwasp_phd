//MIT License
//Copyright (c) 2019 Sherman Lo

package uk.ac.warwick.sip.empiricalnullfilter;

import ij.ImageJ;
import ij.ImagePlus;
import ij.process.FloatProcessor;

import org.apache.commons.math3.random.MersenneTwister;

/**Test class
 * Produce random image and filters it
 * Shows image before and after filtering
 * Shows null mean and null std
 */
public class Test {
  
  public static void main(String[] args) {
    
    DebugPrint.newFile("Test");
    
    //random image
    ImageJ.main(null);
    FloatProcessor processorOrginal = new FloatProcessor(255, 250);
    MersenneTwister rng = new MersenneTwister(1018526);
    for (int i=0; i<processorOrginal.getPixelCount(); i++) {
      processorOrginal.setf(i, (float) rng.nextGaussian());
    }
    ImagePlus image = new ImagePlus("float version" , processorOrginal);
    //show the random image
    ImagePlus org = image.duplicate(); //copy of original before filtering
    org.show();
    
    double radius = 20; //kernel radius
    long time = System.currentTimeMillis(); //start timing
    
    //filter the image
    EmpiricalNullFilter filter = new EmpiricalNullFilter();
    filter.setup(null, image);
    filter.setRadius(radius);
    filter.setOutputImage(3); //show null mean and null std
    filter.run(image.getProcessor());
    image.show();
    
    System.out.println("time "+(System.currentTimeMillis() - time) + " ms");
    
    DebugPrint.close();
  }
  
}
