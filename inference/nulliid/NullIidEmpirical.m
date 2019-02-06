classdef NullIidEmpirical < NullIid
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@NullIid(uint32(2288468478));
    end
    
    function [nullMean, nullStd] = getNull(this, z)
      empiricalNull = EmpiricalNull(z, 0, ...
          this.randStream.randi([intmin('int32'),intmax('int32')],'int32'));
      empiricalNull.estimateNull();
      nullMean = empiricalNull.getNullMean();
      nullStd = empiricalNull.getNullStd();
    end
    
  end
  
end

