package uk.ac.warwick.sip.empiricalnullfilter;

import ij.ImagePlus;
import ij.io.Opener;
import ij.process.ImageConverter;
import ij.process.ImageProcessor;

/**BENCH MARKER
 * For timing it takes to do a mode filter on a given image
 *   The image is converted to float and then filtered. Shows the image before and after and display
 *     the time it took.
 */
public class BenchMarker {

  /**
   * @param fileName location of the benchmark image
   * @param isCpu indicate to use cpu or gpu version
   * @param radius radius of the kernel
   * @param nInitial number of initial points for Newton-Raphson
   * @param dimX gpu block dimension (only ised if isCpu is false)
   * @param dimY gpu block dimension (only ised if isCpu is false)
   */
  public static void benchmark(String fileName, boolean isCpu, Float radius, int nInitial,
      int dimX, int dimY) throws Exception {

    //convert the image to float
    Opener opener = new Opener();
    ImagePlus image = opener.openImage(fileName);
    ImageConverter imageConverter = new ImageConverter(image);
    imageConverter.convertToGray32();
    image.show();

    //instantiate filter
    EmpiricalNullFilter modeFilter;
    if (isCpu) {
      modeFilter = new ModeFilter();
    } else {
      ModeFilterGpu modeFilterGpu = new ModeFilterGpu();
      modeFilterGpu.setBlockDimX(dimX);
      modeFilterGpu.setBlockDimY(dimY);
      modeFilter = (EmpiricalNullFilter) modeFilterGpu;
    }
    modeFilter.setRadius(radius);
    modeFilter.setNInitial(nInitial);
    modeFilter.setOutputImage(0); //do not need other output images

    //do filtering
    ImagePlus imageAfter = image.duplicate();
    ImageProcessor imageProcessor = imageAfter.getProcessor();
    modeFilter.setup("", imageAfter);
    long startTime = System.currentTimeMillis();
    modeFilter.run(imageProcessor);
    long endTime = System.currentTimeMillis();
    System.out.println("Time: " + (endTime - startTime) + " ms");
    imageAfter.show();
  }

}
