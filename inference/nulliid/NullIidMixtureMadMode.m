classdef NullIidMixtureMadMode < NullIidMixture

  methods (Access = protected)
    
    function setup(this)
      this.setup@NullIidMixture(uint32(4284207830));
    end
    
    function [nullMean, nullStd] = getNull(this, z)
      empiricalNull = MadModeNull(z, 0, ...
          this.randStream.randi([intmin('int32'),intmax('int32')],'int32'));
      empiricalNull.estimateNull();
      nullMean = empiricalNull.getNullMean();
      nullStd = empiricalNull.getNullStd();
    end
    
  end
  
end

