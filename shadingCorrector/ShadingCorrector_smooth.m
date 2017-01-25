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
        
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %PARAMETERS:
            %reference_image_array: stack of blank scans
        function this = ShadingCorrector_smooth(reference_image_array)
            this = this@ShadingCorrector(reference_image_array);
            this.can_smooth = true;
            this.orginial_reference_array = reference_image_array;
        end
        
        
        %MEAN SQUARED ERROR
        %Calculate the mean squared difference between the orginial and
        %smoothed reference image
        %PARAMETER:
            %index: index pointing to an image in the reference stack
        %RETURN:
            %mse: mean squared error
        function mse = meanSquaredError(this,index)
            %calculate the mean squared error
            mse = sum(sum((this.getResidualImage(index)).^2)) / (this.image_size(1) * this.image_size(2));
        end
        
        %GET RESIUDAL IMAGE
        %Get image of reference - smoothed reference image and scaled to
        %have std 1
        %PARAMETER:
            %index: index pointing to an image in the reference stack
        function residual_image = getZResidualImage(this,index)
            residual_image = this.reference_image_array(:,:,index) - this.orginial_reference_array(:,:,index);
            r = reshape(residual_image,[],1);
            scale = std(r);
            residual_image = (residual_image)/scale;
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
