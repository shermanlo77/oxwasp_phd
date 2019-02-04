classdef MadModeNullFilter < EmpiricalNullFilter
  
  methods (Access = public)
    
    function this = MadModeNullFilter(radius)
      this@EmpiricalNullFilter(radius);
    end
    
  end
  
  methods (Access = protected)
    
    %METHOD: GET FILTER
    function getFilter(this)
      this.javaFilter = uk.ac.warwick.sip.empiricalnullfilter.MadModeNullFilter();
    end
    
  end
  
end

