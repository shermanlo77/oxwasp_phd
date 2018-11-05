%CLASS: EXPERIMENT ALL NULL GAUSSIAN
%See superclass Experiment_AllNull
%Experiment with the empirical null filter on a gaussian image
classdef Experiment_AllNullGaussian < Experiment_AllNull
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = Experiment_AllNullGaussian()
      this@Experiment_AllNull("Experiment_AllNullGaussian");
    end
    
  end
  
  methods (Access = protected)
    
    %METHOD: SETUP
    function setup(this)
      this.setup@Experiment_AllNull(uint32(3499211588));
    end
    
    %METHOD: GET IMAGE
    function image = getImage(this)
      %return pure gaussian image
      image = this.randStream.randn(this.imageSize(1), this.imageSize(2));
    end
    
  end
  
end

