classdef AllNullGaussianMedianIqr  < AllNull
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = AllNullGaussianMedianIqr()
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
    %instantiate an medianiqr null filter with that radius
    function filter = getFilter(this, radius)
      filter = MedianIqrNullFilter(radius);
    end
    
  end
  
end

