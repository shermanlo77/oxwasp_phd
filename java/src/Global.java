import org.apache.commons.math3.random.MersenneTwister;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
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
    filter.rank(image.getProcessor(), radius, RankFilters.MEDIAN);
    image.show();
    
    ImagePlus output;
    output = new ImagePlus("null mean", new FloatProcessor(image.getWidth(), image.getHeight(),
        filter.getOutputImage(EmpiricalNullFilter.NULL_MEAN)));
    output.show();
    
    output = new ImagePlus("null std", new FloatProcessor(image.getWidth(), image.getHeight(),
        filter.getOutputImage(EmpiricalNullFilter.NULL_STD)));
    output.show();
    
    output = new ImagePlus("std", new FloatProcessor(image.getWidth(), image.getHeight(),
        filter.getOutputImage(EmpiricalNullFilter.STD)));
    output.show();
    
    output = new ImagePlus("q1", new FloatProcessor(image.getWidth(), image.getHeight(),
        filter.getOutputImage(EmpiricalNullFilter.Q1)));
    output.show();
    
    output = new ImagePlus("q2", new FloatProcessor(image.getWidth(), image.getHeight(),
        filter.getOutputImage(EmpiricalNullFilter.Q2)));
    output.show();
    
    output = new ImagePlus("q3", new FloatProcessor(image.getWidth(), image.getHeight(),
        filter.getOutputImage(EmpiricalNullFilter.Q3)));
    output.show();
    
    System.out.println("time "+(System.currentTimeMillis() - time) + " ms");
    
  }
  
}
