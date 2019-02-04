classdef NullIidMixture2Empirical < NullIidMixture2
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@NullIidMixture2(uint32(3750354808));
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

