package uk.ac.warwick.sip.empiricalnullfilter;

public class InvalidValueException extends Exception{

  public InvalidValueException() {
    super();
  }
  
  public InvalidValueException(String string) {
    super(string);
  }
  
}