package test;

import ij.ImageJ;
import ij.ImagePlus;
import ij.plugin.filter.PlugInFilterRunner;
import ij.process.FloatProcessor;
import uk.ac.warwick.sip.empiricalNullFilter.EmpiricalNullFilter;

import org.apache.commons.math3.random.MersenneTwister;

public class GlobalInteractive {
  
  public static void main(String[] args) {
    
    System.out.println("Hello");
    ImageJ.main(null);
    FloatProcessor processorOrginal = new FloatProcessor(255, 250);
    MersenneTwister rng = new MersenneTwister(1018526);
    for (int i=0; i<processorOrginal.getPixelCount(); i++) {
      processorOrginal.setf(i, (float) rng.nextGaussian());
    }
    
    ImagePlus image = new ImagePlus("float version" , processorOrginal);
    
    double radius = 20;
    
    ImagePlus org = image.duplicate();
    org.show();
    
    long time = System.currentTimeMillis();
    
    EmpiricalNullFilter filter = new EmpiricalNullFilter();
    PlugInFilterRunner pfr = new PlugInFilterRunner(filter, "empirical null filter", null);
    image.show();
    pfr = new PlugInFilterRunner(filter, "empirical null filter", null);
    
    System.out.println("time "+(System.currentTimeMillis() - time) + " ms");
    
  }
  
}