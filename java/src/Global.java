import org.apache.commons.math3.random.MersenneTwister;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.plugin.filter.PlugInFilterRunner;
import ij.plugin.filter.RankFilters;
import ij.process.FloatProcessor;

public class Global {
  
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
    new PlugInFilterRunner(filter, "empirical null filter", null);
    image.show();
    
    System.out.println("time "+(System.currentTimeMillis() - time) + " ms");
    
  }
  
}
