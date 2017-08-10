%SCAN SINGLE CLASS
%See superclass Scan
%This is a special case of Scan where there is only one image
%Overrides the method loadImage, does not have a integer suffix in the file name
classdef Scan_Single < Scan
    
    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = Scan_Single(folder_location, file_name, width, height, voltage, power, time_exposure)
            %calls superclass constructor
            this@Scan(folder_location, file_name, width, height, 1, voltage, power, time_exposure);
        end
        
        %OVERRIDE: LOAD IMAGE
        %Return a sample image
        %PARAMETERS
            %not used
        function slice = loadImage(this,~)
            slice = imread(strcat(this.folder_location,this.file_name,'.tif'));
            slice = this.shadingCorrect(double(slice));
        end
        
    end
    
end
