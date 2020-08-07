package uk.ac.warwick.sip.empiricalnullfilter;

public class EmpiricalNullFilterGpu extends EmpiricalNullFilter {

  public EmpiricalNullFilterGpu() {
  }

  @Override
  protected Cache instantiateCache() {
    return new CacheReflect(this.imageProcessor, this.roi);
  }

}
