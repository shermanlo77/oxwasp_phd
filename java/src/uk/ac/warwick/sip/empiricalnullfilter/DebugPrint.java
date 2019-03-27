package uk.ac.warwick.sip.empiricalnullfilter;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.nio.file.Paths;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

public class DebugPrint {
  
  static final String directory = "/home/sherman/Documents/oxwasp_phd/debugReport";
  static String name;
  static BufferedWriter bufferedWriter;
  
  public static void newFile(String prefix) {
    DateFormat dateFormat = new SimpleDateFormat("yyyymmddhhmmss");
    Date date = new Date();
    name = prefix + "_" + dateFormat.format(date);
    
    try {
      FileWriter fileWriter = 
          new FileWriter(Paths.get(directory,name+".txt").toFile());
      bufferedWriter = new BufferedWriter(fileWriter);
      write(name + " debug report:");
    } catch(Exception exception) {
      System.out.println("Exception caught when making new file with prefix "+prefix);
      exception.printStackTrace();
    }
  }
  
  public static void write(String string) {
    if (bufferedWriter == null) {
      System.out.println(string);
    } else {
      try {
        bufferedWriter.write(string);
        bufferedWriter.newLine();
        flush();
      } catch (Exception exception) {
        System.out.println("Exception caught when writing "+string);
        exception.printStackTrace();
      }
    }
  }
  
  public static void flush() {
    if (bufferedWriter != null) {
      try {
        bufferedWriter.flush();
      } catch (Exception exception) {
        System.out.println("Exception caught when flushing");
        exception.printStackTrace();
      }
    }
  }
  
  public static void close() {
    if (bufferedWriter != null) {
      try {
        flush();
        bufferedWriter.close();
        bufferedWriter = null;
      } catch (Exception exception) {
        System.out.println("Exception caught when closing");
        exception.printStackTrace();
      }
    }
  }
  
}
