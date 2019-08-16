%MIT License
%Copyright (c) 2019 Sherman Lo

%SCAN SINGLE FLAT
%See super class Scan_Single and Scan
%Contains an image of only one greyvalue
classdef ScanSingleFlat < ScanSingle
  
  properties (SetAccess = private)
    
    %greyvalue of the flat colour
    greyvalue;
    
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = ScanSingleFlat(width, height, voltage, power, timeExposure, greyvalue)
      %calls superclass constructor
      this@ScanSingle([], [], width, height, voltage, power, timeExposure);
      this.greyvalue = greyvalue;
    end
    
    %OVERRIDE: LOAD IMAGE
    %Return a flat colour sample image
    %PARAMETERS
    function slice = loadImage(this, ~)
      slice = this.greyvalue * ones(this.height, this.width);
    end
    
  end
  
end

