classdef (Abstract) NullIidMixture < NullIid
  
  methods (Access = protected)
    
    function setup(this, seed)
      this.setup@NullIid(seed);
    end
    
    function z = getSample(this, n)
      n0 = round(0.9*n);
      n1 = n - n0;
      z = [this.randStream.randn(n0,1);  this.randStream.randn(n1,1)+3];
    end
    
  end
  
end

