classdef Lowess < SurfaceFitting
    %LOWESS Fit lowess to each panel
    %   Subclass of SurfaceFitting
    
    properties
    end
    
    methods
        
        %CONSTRUCTOR
        %PARAMETERS
            %total_size: [height, width] of the size of the detector image
            %active_size: [height, width] of the size of the cropped image
            %n_panel: [height, width] of the number of panels in the image
        function this = Lowess(total_size,active_size,n_panel_column)
            this = this@SurfaceFitting(total_size,active_size,n_panel_column);
        end
        
        %FIT LOWESS
        %Fit lowess with span p
        %PARAMETERS:
            %x: vector of x coordinates
            %y: vector of y coordinates
            %z: vector of greyvalues
            %p: span
        %RETURN:
            %sfit_obj: sfit object (surface fit)
        function sfit_obj = fitPanel(this,x,y,z,p)
            sfit_obj = fit([x,y],z,'lowess','Span',p);
        end
        
    end
    
end

