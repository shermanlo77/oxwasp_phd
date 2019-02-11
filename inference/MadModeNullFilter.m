classdef MadModeNullFilter < EmpiricalNullFilter
  
  methods (Access = public)
    
    function this = MadModeNullFilter(radius)
      this@EmpiricalNullFilter(radius);
      this.javaFilter = uk.ac.warwick.sip.empiricalnullfilter.MadModeNullFilter();
      this.javaFilter.setRadius(radius);
    end
    
  end
  
end

