//MIT License
//Copyright (c) 2019 Sherman Lo

package uk.ac.warwick.sip.empiricalnullfilter;

import ij.ImageJ;
import ij.ImagePlus;
import ij.plugin.filter.PlugInFilterRunner;
import ij.process.FloatProcessor;
import uk.ac.warwick.sip.empiricalnullfilter.EmpiricalNullFilter;

import org.apache.commons.math3.random.MersenneTwister;

/**Test Interactive class
 * Produce random image and filters it using plugin
 * Shows image before and after filtering
 * Shows options for what output images to show
 */
public class TestInteractive {
  
  public static void main(String[] args) {
    
    DebugPrint.newFile("TestInteractive");
    
    //produce random image
    ImageJ.main(null);
    FloatProcessor processorOrginal = new FloatProcessor(255, 250);
    MersenneTwister rng = new MersenneTwister(1018526);
    for (int i=0; i<processorOrginal.getPixelCount(); i++) {
      processorOrginal.setf(i, (float) rng.nextGaussian());
    }
    
    //copy random image
    ImagePlus image = new ImagePlus("float version" , processorOrginal);
    ImagePlus org = image.duplicate();
    org.show(); //show random image
    
    double radius = 20; //kernel radius
    long time = System.currentTimeMillis(); //timing
    
    //filter the image using it as a plugin
    EmpiricalNullFilter filter = new EmpiricalNullFilter();
    PlugInFilterRunner pfr = new PlugInFilterRunner(filter, "empirical null filter", null);
    image.show();
    
    //filter again, the settings should be the same as last time
    pfr = new PlugInFilterRunner(filter, "empirical null filter", null);
    
    System.out.println("time "+(System.currentTimeMillis() - time) + " ms");
    
    DebugPrint.close();
    
  }
  
}