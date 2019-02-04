classdef (Abstract) NullIidMixture2 < NullIid
  
  methods (Access = protected)
    
    function setup(this, seed)
      this.setup@NullIid(seed);
    end
    
    function z = getSample(this, n)
      isNull = this.randStream.rand(n,1) < 0.9;
      z = zeros(n,1);
      z(isNull) = this.randStream.randn(sum(isNull),1);
      z(~isNull) = this.randStream.randn(sum(~isNull),1) + (this.randStream.randi([0,1])*2-1)*3;
    end
    
  end
  
end

