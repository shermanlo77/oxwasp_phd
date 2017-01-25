classdef ShadingCorrector_null < ShadingCorrector
    %SHADINGCORRECTOR_NULL Stores an array of blank scans, does not do
    %shading correction
    %   A stack of blank scans (white, grey, black images) is passed to the
    %   object via the constructor.
    
    %MEMBER VARIABLES
    properties
    end
    
    %METHODS
    methods
        
         %CONSTRUCTOR
        %PARAMETERS:
            %reference_image_array: stack of blank scans
        function this = ShadingCorrector_null(reference_image_array)
            this = this@ShadingCorrector(reference_image_array);
        end
        
        %CALIBRATE
        %Does nothing
        function calibrate(this)   
        end
        
        %SHADE CORRECT
        %PARAMETERS:
            %scan_image: scan_image
        function scan_image = shadeCorrect(this,scan_image)
            scan_image = scan_image;
        end
        
    end
    
end

