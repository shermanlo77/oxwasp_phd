classdef ShadingCorrector_smooth < ShadingCorrector
    %SHADINGCORRECTOR_SMOOTH Abstract class for a shading_corrector which
    %smooths the reference images
    %   To implement a subclass, the method smoothPanel(index,corner_position,parameter)
    %   must be implemented. This modifies the member variable reference_image_array
    %   by smoothing a panel of the index-th image using the parameter.
    
    %MEMBER VARIABLE
    properties
        
        %the orginial array of reference images
        orginial_reference_array;
        parameter;
        panel_counter;
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %PARAMETERS:
            %reference_image_array: stack of blank scans
        function this = ShadingCorrector_smooth(reference_image_array, panel_counter, parameter)
            this = this@ShadingCorrector(reference_image_array);
            this.can_smooth = true;
            %make a copy of the reference images
            this.orginial_reference_array = reference_image_array;
            this.panel_counter = panel_counter;
            this.parameter = parameter;
            
            this.smoothEachPanel();
        end
        
        %MEAN SQUARED ERROR
        %Calculate the mean squared difference between the orginial and
        %smoothed reference image
        %PARAMETER:
            %index: index pointing to an image in the reference stack
        %RETURN:
            %mse: mean squared error
%         function mse = meanSquaredError(this,index)
%             %calculate the mean squared error
%             mse = sum(sum((this.getResidualImage(index)).^2)) / (this.image_size(1) * this.image_size(2));
%         end
        
        %GET RESIUDAL IMAGE
        %Get image of reference - smoothed reference image and scaled to
        %have std 1
        %PARAMETER:
            %index: index pointing to an image in the reference stack
        function residual_image = getZResidualImage(this,index)
            %get the residual of each pixel
            residual_image = this.reference_image_array(:,:,index) - this.orginial_reference_array(:,:,index);
            %get the std of the residuals
            scale = std(reshape(residual_image,[],1));
            %scale the residual
            residual_image = residual_image/scale;
        end
        
        %SMOOTH PANELS
        %Call this method is a shading corrector was provided and to use it
        %to smooth the reference images panel by panel.
        function smoothEachPanel(this)
            
            %start counting the panels
            this.panel_counter.resetPanelCorner();
            %for each panel
            while this.panel_counter.hasNextPanelCorner()
                %get the coordinates of the panel
                panel_corners = this.panel_counter.getNextPanelCorner();
                %for each reference image, smooth that panel
                for i_index = 1:this.n_image
                    this.smoothPanel(i_index,panel_corners,this.parameter(i_index));
                end
            end
        end
        
    end
    
    %ABSTRACT METHODS
    methods (Abstract)
        
        %SMOOTH PANEL
        %Smooth a panel of an image in this.reference_image_array
        %PARAMETERS:
            %index: pointer to an image in this.reference_image_array
            %corner_position: 2x2 matrix, 1st column represent the top left
            %of the panel, 2nd column represent the bottom right of the
            %panel, all inclusive
            %parameter: parameter for the surface or smoothing to be
            %implemented
        smoothPanel(index,corner_position,parameter);
    
    end
    
end
