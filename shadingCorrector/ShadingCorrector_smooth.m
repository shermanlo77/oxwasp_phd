%SHADINGCORRECTOR_SMOOTH Abstract class for a shading_corrector which smooths the reference images panel by panel
    %   To implement a subclass, the method smoothPanel(index,corner_position,parameter) must be implemented.
    %   This method modifies the member variable reference_image_array by smoothing a panel of the index-th image using the parameter.
    %
    %   To set up the shading corrector, a panel_counter object must be provided via the method addPanelCounter
    %   Then provide reference images via the method addReferenceImages()
    %   Then call the method calibrate() 
classdef ShadingCorrector_smooth < ShadingCorrector
    
    %MEMBER VARIABLE
    properties
        %the orginial array of reference images
        orginial_reference_array;
        %vector of parameters
            %dim 1: for each image in orginial_reference_array
        parameter;
        %panel counter object
            %used for extracting the coordinates of each panel
        panel_counter;
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %PARAMETERS:
            %parameters: vector of parameters for smoothing each reference image
                %dim 1: for each image in orginial_reference_array
        function this = ShadingCorrector_smooth(parameter)
            %call super class
            this = this@ShadingCorrector();
            %assign member variables
            this.can_smooth = true;
            this.parameter = parameter;
        end
        
        %ADD REFERENCE IMAGES
        %PARAMETERS:
            %reference_image_array: stack of blank scans (see superclass)
        function addReferenceImages(this, reference_image_array)
            %make a copy of the reference images
            this.orginial_reference_array = reference_image_array;
            %call the superclass version of addReferenceImages
            this.addReferenceImages@ShadingCorrector(reference_image_array);
        end
        
        %CALIBRATE
        %Work out the parameters for shading correction
        function calibrate(this)
            %smooth each image panel by panel
            this.smoothEachPanel();
            %call the superclass version of calibrate
            this.calibrate@ShadingCorrector();
        end
        
        %ADD PANEL COUNTER
        %Add a panel counter object, to be used to extract the coordinates of each panel
        %PARAMETERS:
            %panel_counter: panel_counter object
        function addPanelCounter(this, panel_counter)
            %assign member variable
            this.panel_counter = panel_counter;
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
        
        %SMOOTH EACH PANEL
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
