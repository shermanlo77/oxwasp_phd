%MIT License
%Copyright (c) 2019 Sherman Lo

classdef (Abstract) NullIidMixture < NullIid
  
  methods (Access = protected)
    
    function setup(this, seed)
      this.setup@NullIid(seed);
    end
    
    function yLim = getYLim(this, index)
      switch index
        case 1
          yLim = [-0.3, 0.6];
        case 2
          yLim = [0.8, 1.8];
        otherwise
          yLim = this.getYLim@NullIid(index);
      end
    end
    
    %OVERRIDE: GET SAMPLE
    %Return mixture of Gaussian
    function z = getSample(this, n)
      isNull = this.randStream.rand(n,1) < 0.9;
      z = zeros(n,1);
      z(isNull) = this.randStream.randn(sum(isNull),1);
      z(~isNull) = this.randStream.randn(sum(~isNull),1)+3;
    end
    
  end
  
end

