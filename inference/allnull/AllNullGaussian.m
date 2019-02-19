%CLASS: EXPERIMENT ALL NULL GAUSSIAN
%See superclass Experiment_AllNull
%Experiment with the empirical null filter on a gaussian image
classdef AllNullGaussian < AllNull
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = AllNullGaussian()
      this@AllNull();
    end
    
  end
  
  methods (Access = protected)
    
    %OVERRIDE: SETUP
    function setup(this, seed)
      this.setup@AllNull(seed);
    end
    
    %IMPLEMENTED: GET IMAGE
    function image = getImage(this)
      %return pure gaussian image
      image = this.randStream.randn(this.imageSize(1), this.imageSize(2));
    end
    
  end
  
end

