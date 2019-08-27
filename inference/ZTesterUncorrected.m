%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: Z TESTER UNCORRECTED
%Z tester using the uncorrected method
classdef ZTesterUncorrected < ZTester
  
  properties (SetAccess = protected)
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = ZTesterUncorrected(z_image)
      this@ZTester(z_image);
    end
    
    %OVERRIDE: DO TEST
    %Does hypothesis using the p values, uncorrected for multiple hypothesis testing
    %Saves positive pixels in the member variable pImage
    function doTest(this)
      %calculate the p values
      this.pImage = 2*(normcdf(-abs(this.getZCorrected())));
      %save the results, sig_image and size_corrected
      this.positiveImage = this.pImage < this.threshold; %2d boolean of positive pixels
      this.sizeCorrected = this.threshold; %size of test
    end
    
  end
  
end

