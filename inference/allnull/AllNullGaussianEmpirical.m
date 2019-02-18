%CLASS: EXPERIMENT ALL NULL GAUSSIAN
%See superclass Experiment_AllNull
%Experiment with the empirical null filter on a gaussian image
classdef AllNullGaussianEmpirical < AllNull
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = AllNullGaussianEmpirical()
      this@AllNull();
    end
    
  end
  
  methods (Access = protected)
    
    %METHOD: SETUP
    function setup(this)
      this.setup@AllNull(uint32(3499211588));
    end
    
    %METHOD: GET IMAGE
    function image = getImage(this)
      %return pure gaussian image
      image = this.randStream.randn(this.imageSize(1), this.imageSize(2));
    end
    
    %METHOD: GET FILTER
    %instantiate an empirical null filter with that radius
    function filter = getFilter(this, radius)
      filter = EmpiricalNullFilter(radius);
      filter.setSeed(this.randStream.randi([intmin('int32'),intmax('int32')],'int32'));
    end
    
  end
  
end

