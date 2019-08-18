%MIT License
%Copyright (c) 2019 Sherman Lo

classdef ShadingCorrectorNull < ShadingCorrector
    %SHADINGCORRECTOR_NULL Stores an array of refererence scans, does not do
    %shading correction
    %   A stack of reference scans (white, grey, black images) is passed to the
    %   object via the constructor.
    
    %MEMBER VARIABLES
    properties
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %PARAMETERS:
            %reference_image_array: stack of reference scans
        function this = ShadingCorrectorNull()
            this = this@ShadingCorrector();
        end
        
        %CALIBRATE
        %Does nothing
        function calibrate(this)   
        end
        
        %SHADE CORRECT
        %PARAMETERS:
            %scan_image: scan_image
        function scan_image = shadingCorrect(this,scan_image)
            scan_image = scan_image;
        end
        
        %RETURN THE NAME OF THIS SHADING CORRECTOR
        function name = getName(this)
            name = 'null';
        end
        
    end
    
end

