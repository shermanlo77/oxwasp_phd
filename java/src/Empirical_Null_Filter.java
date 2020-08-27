//MIT License
//Copyright (c) 2019 Sherman Lo

import uk.ac.warwick.sip.empiricalnullfilter.EmpiricalNullFilter;
import uk.ac.warwick.sip.empiricalnullfilter.BenchMarker;

//EMPIRICAL NULL FILTER
/**Copy of EmpiricalNullFilter, the class name has underscores so that it can be used by ImageJ
 * @author Sherman Lo
 */
public class Empirical_Null_Filter extends EmpiricalNullFilter{

  public Empirical_Null_Filter() {
    this.setProgress(true);
  }

  //MAIN
  //-b for benchmark
  public static void main(String[] args) throws Exception{
    System.out.println("MIT License - please see LICENSE");
    System.out.println("Copyright (c) 2019 Sherman Lo");
    System.out.println("Please see https://github.com/shermanlo77/oxwasp_phd or README.md");

    if (args.length > 0) {

      //benchmark
      //see Benchmarker for useage
      if (args[0].equals("-b")) {

        String fileName = null;
        boolean isCpu = false;
        float radius = 0.0f;
        int nInitial = 0;
        int dimX = 0;
        int dimY = 0;
        boolean gotParameters = false;

        //check the content of args
        try {
          fileName = args[1];
          if (args[2].equals("cpu")) {
            isCpu = true;
          } else if (args[2].equals("gpu")){
            isCpu = false;
          } else {
            throw new Exception();
          }
          radius = Float.parseFloat(args[3]);
          nInitial = Integer.parseInt(args[4]);
          if (!isCpu) {
            if (args.length == 7) {
              dimX = Integer.parseInt(args[5]);
              dimY = Integer.parseInt(args[6]);
            } else {
              throw new Exception();
            }
          }
          gotParameters = true;
        } catch (Exception exception) {
          System.out.println("");
          printBenchmarkManual();
        }

        //then do the benchmark
        if (gotParameters) {
          try {
            BenchMarker.benchmark(fileName, isCpu, radius, nInitial, dimX, dimY);
          } catch (Exception exception) {
            System.out.println("");
            System.out.println("Exception caught");
            System.out.println("");
            printBenchmarkManual();
            System.out.println("");
            throw exception;
          }
        }

      }
    }
  }

  public static void printBenchmarkManual() {
    System.out.println("-b benchmark useage");
    System.out.println("-b fileName cpu/gpu radius nInitial dimX dimY");
    System.out.println("fileName: location of the benchmark image");
    System.out.println("cpu/gpu: the text either cpu or gpu");
    System.out.println("radius: radius of the kernel");
    System.out.println("nInitial: number of initial points for Newton-Raphson");
    System.out.println("dimX: gpu block dimension");
    System.out.println("dimY: gpu block dimension");
  }
}
