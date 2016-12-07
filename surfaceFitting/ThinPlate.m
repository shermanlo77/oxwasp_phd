classdef ThinPlate < SurfaceFitting
    %LOWESS Fit thin plate to each panel
    %   Subclass of SurfaceFitting
    
    properties
    end
    
    methods
        
        %CONSTRUCTOR
        %PARAMETERS
            %total_size: [height, width] of the size of the detector image
            %active_size: [height, width] of the size of the cropped image
            %n_panel: [height, width] of the number of panels in the image
        function this = ThinPlate(total_size,active_size,n_panel_column)
            this = this@SurfaceFitting(total_size,active_size,n_panel_column);
        end
        
        %FIT THIN PLATE
        %Fit thin plate
        %PARAMETERS:
            %x: vector of x coordinates
            %y: vector of y coordinates
            %z: vector of greyvalues
            %p: unused
        %RETURN:
            %sfit_obj: sfit object (surface fit)
        function sfit_obj = fitPanel(this,x,y,z,p)
            sfit_obj = fit([x,y],z,'thinplateinterp');
        end
        
    end
    
end

