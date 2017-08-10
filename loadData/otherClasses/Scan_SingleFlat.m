%SCAN SINGLE FLAT
%See super class Scan_Single and Scan
%Contains an image of only one greyvalue
classdef Scan_SingleFlat < Scan_Single
    
    properties
        
        %greyvalue of the flat colour
        greyvalue;
        
    end
    
    methods
        
        %CONSTRUCTOR
        function this = Scan_SingleFlat(width, height, voltage, power, time_exposure, greyvalue)
            %calls superclass constructor
            this@Scan_Single([], [], width, height, voltage, power, time_exposure);
            this.greyvalue = greyvalue;
        end
        
        %OVERRIDE: LOAD IMAGE
        %Return a flat colour sample image
        %PARAMETERS
            %not used
        function slice = loadImage(this,~)
            slice = this.greyvalue * ones(this.height, this.width);
        end
        
    end
    
end

