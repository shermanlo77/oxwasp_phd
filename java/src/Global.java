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
    MersenneTwister rng = new MersenneTwister(System.currentTimeMillis());
    for (int i=0; i<processorOrginal.getPixelCount(); i++) {
      processorOrginal.setf(i, (float) rng.nextGaussian());
    }
    //processorOrginal.noise(1.0);
    
    ImagePlus image = new ImagePlus("float version" , processorOrginal);
    
    double radius = 50;
    
    ImagePlus org = image.duplicate();
    org.show();
    
    ImagePlus stdImage = image.duplicate();
    RankFilters varianceFilter = new RankFilters();
    varianceFilter.rank(stdImage.getProcessor(), radius, RankFilters.VARIANCE);
    stdImage.getProcessor().sqrt();
    stdImage.show();
    
    EmpiricalNullFilter filter = new EmpiricalNullFilter();
    filter.rank(image.getProcessor(), radius, RankFilters.MEDIAN, (FloatProcessor) stdImage.getProcessor());
    image.show();
    
    ImagePlus output;
    output = new ImagePlus("null mean", new FloatProcessor(image.getWidth(), image.getHeight(),
        filter.getOutputImage(EmpiricalNullFilter.NULL_MEAN)));
    output.show();
    
    output = new ImagePlus("null std", new FloatProcessor(image.getWidth(), image.getHeight(),
        filter.getOutputImage(EmpiricalNullFilter.NULL_STD)));
    output.show();
    
  }
  
}
