%SCAN Abstract class for handling x-ray images
classdef Scan < handle

    %MEMBER VARIABLES
    properties
        width; %width of the image
        height; %height of the image
        area; %area of the image
        n_sample; %number of images
        file_name; %... name of sample image
        folder_location; %location of the dataset
        reference_image_array; %array of reference scan objects (black, white then grey)
        panel_counter %panel counter object
        shading_corrector; %shading corrector object
        want_shading_correction; %boolean, true to do shading correction, default false, automatically turns to true if a shading corrector is added
        want_remove_dead_pixels; %boolean, true to remove dead pixels, default false
        
        min_greyvalue; %minimum possible greyvalue
    end
    
    methods
        
        %CONSTRUCTOR
        %PARAMETERS:
            %folder_location: location of the images
            %file_name: name of the image files
            %width: width of the image
            %height: height of the image
            %n_sample: number of images
        function this = Scan(folder_location, file_name, width, height, n_sample)
            %assign member variable if parameters are provided
            if nargin > 0
                this.folder_location = folder_location;
                this.file_name = file_name;
                this.width = width;
                this.height = height;
                this.area = width * height;
                this.n_sample = n_sample;
                this.want_shading_correction = false;
                this.want_remove_dead_pixels = false;
            end
        end
        
        %LOAD SAMPLE IMAGE
        %Return a sample image
        %PARAMETERS
            %index: index of image (scalar)
        function slice = loadSample(this,index)
            slice = imread(strcat(this.folder_location,this.file_name,num2str(index),'.tif'));
            slice = this.shadingCorrection(double(slice));
        end
        
        %LOAD SAMPLE IMAGE STACK
        %Return stack of sample images
        %PARAMETERS:
            %range (optional): vector of indices of images requested
                %if empty return the full range
        function stack = loadSampleStack(this,range)
            %if range not provided, provide the full range
            if nargin == 1
                range = 1:this.n_sample;
            end
            %declare stack of images
            stack = zeros(this.height,this.width,numel(range));
            %for each image, put it in the stack
            for index = 1:numel(range)
                stack(:,:,index) = this.loadSample(range(index));
            end
        end
        
        %SHADING CORRECTION
        %if want_shading_correction, does shading correction on the
        %provided image and returns it
        %if want_remove_dead_pixels, remove dead pixels
        function slice = shadingCorrection(this,slice)
            if this.want_shading_correction
                slice = this.shading_corrector.shadeCorrect(slice);
            end
            if this.want_remove_dead_pixels
                slice = removeDeadPixels(slice);
            end
        end
        
        %ADD SHADING CORRECTOR
        %Assign a shading corrector to the member variable and calibrate it for shading correction
        %The reference images used are determined by the parameter reference_index
        %PARAMETERS:
            %shading_corrector: ShadingCorrector object
            %reference_index: matrix of index (optional)
                %dim 1: image index (integers)
                %dim 2: (black, white, grey)
                %for each column, (eg 1st column for black images)
                %the mean of black images specified by dim 1 are used for shading corrector
                %
                %if not provided, use the mean of all images, black, grey and white
        function addShadingCorrector(this,shading_corrector,reference_index)
            
            %turn off shading correction to obtain the reference images
            this.turnOffShadingCorrection();
            
            %if reference_index is not provided
            if nargin == 2
                %use all black, white and grey images
                n_reference = numel(this.reference_image_array);
            %else get the number of colours requested
            else            
                n_reference = numel(reference_index(1,:));
            end
            
            %declare a stack of reference images
            reference_stack = zeros(this.height,this.width,n_reference);
            
            %for each reference image (or colour)
            for i = 1:n_reference
                %if reference_index is not provided, take the mean of all images
                if nargin == 2
                    reference_stack(:,:,i) = mean(this.reference_image_array(i).loadSampleStack(),3);
                %else take the mean of all images specified in the ith column of reference_index
                else
                    reference_i_index = reference_index(:,i);
                    reference_i_index(reference_i_index==0) = [];
                    reference_stack(:,:,i) = mean(this.reference_image_array(i).loadSampleStack(reference_i_index),3);
                end
            end

            %add the reference_stack to the shading corrector
            shading_corrector.addReferenceImages(reference_stack);
            
            %add the shading corrector to the member variable
            this.addManualShadingCorrector(shading_corrector);
            
        end
        
        %ADD MANUAL SHADING CORRECTOR
        %Assign a provided shading corrector to the member variable and calibrate it
        %for shading correction
        %PARAMETERS:
            %shading_corrector: shading_corrector object
        function addManualShadingCorrector(this,shading_corrector)
            %assign the provided shading corrector to the member variable
            this.shading_corrector = shading_corrector;

            %get the minimum possible greyvalue to the shading corrector
            this.shading_corrector.min_greyvalue = this.min_greyvalue;
            
            %if the shading corrector can smooth the reference images panel by panel
                %add the panel counter
            if this.shading_corrector.can_smooth
                shading_corrector.addPanelCounter(this.panel_counter);
            end
            
            %calibrate the shading corrector to do shading correction
            this.shading_corrector.calibrate();
            %set shading correction to be on
            this.turnOnShadingCorrection();
        end
        
        %TURN ON SHADING CORRECTION
        %Set the member variable want_shading_correction to be true
        function turnOnShadingCorrection(this)
            this.want_shading_correction = true;
        end
        
        %TURN OFF SHADING CORRECTION
        %Set the memebr variable want_shading_correction to be false
        function turnOffShadingCorrection(this)
            this.want_shading_correction = false;
        end
        
        %TURN ON REMOVE DEAD PIXELS
        function turnOnRemoveDeadPixels(this)
            this.want_remove_dead_pixels = true;
        end
        
        %TURN OFF REMOVE DEAD PIXELS
        function turnOffRemoveDeadPixels(this)
            this.want_remove_dead_pixels = false;
        end
        
        %TURN ON SET EXTREME TO NAN
        %Set the shading corrector to set extreme greyvalues to be NaN
        function turnOnSetExtremeToNan(this)
            this.shading_corrector.turnOnSetExtremeToNan();
        end
        
        %TURN OFF SET EXTREME TO NAN
        %Set the shading corrector to keep extreme greyvalues
        function turnOffSetExtremeToNan(this)
            this.shading_corrector.turnOffSetExtremeToNan();
        end
        
        %LOAD REFERENCE
        %Return a b/g/w image
        %PARAMETER
            %reference_index: 1 for black, 2 for white, 3 for grey...
            %image_index: scalar integer, index for image
        function slice = loadReference(this,reference_index,image_index)
            slice = this.reference_image_array(reference_index).loadSample(image_index);
            slice = this.shadingCorrection(slice);
        end
        
        %LOAD REFERENCE STACK
        %Return a stack of b/g/w images
        %PARAMETERS:
            %reference_index: 1 for black, 2 for white, 3 for grey...
            %image_index_array: vector for scalar integers, index for images
        function stack = loadReferenceStack(this,reference_index,image_index_array)
            %if range not provided, provide the full range
            if nargin == 2
                image_index_array = 1:this.reference_image_array(reference_index).n_sample;
            end
            %declare stack of images
            stack = zeros(this.height,this.width,numel(image_index_array));
            for i_image = 1:numel(image_index_array)
                stack(:,:,i_image) = this.loadReference(reference_index,image_index_array(i_image));
            end
        end
        
        %LOAD BLACK IMAGE
        %Return a black image
        %PARAMETERS
            %index: index of image
        function slice = loadBlack(this,index)
            slice = this.loadReference(1,index);
        end
        
        %LOAD GREY IMAGE
        %Return a grey image
        %PARAMETERS
            %index: index of image
        function slice = loadGrey(this,index)
            slice = this.loadReference(3,index);
        end
        
        %LOAD WHITE IMAGE
        %Return a white image
        %PARAMETERS
            %index: index of image
        function slice = loadWhite(this,index)
            slice = this.loadReference(2,index);
        end
        
        %LOAD BLACK IMAGE STACK
        %Return stack of black images
        %PARAMETERS:
            %range (optional): vector of indices of images requested, if
            %empty return the full range
        function stack = loadBlackStack(this,range)
            if nargin == 2
                stack = this.loadReferenceStack(1,range);
            else
                stack = this.loadReferenceStack(1);
            end
        end
        
        %LOAD GREY IMAGE STACK
        %Return stack of grey images
        %PARAMETERS:
            %range (optional): vector of indices of images requested, if
            %empty return the full range
        function stack = loadGreyStack(this,range)
            if nargin == 2
                stack = this.loadReferenceStack(3,range);
            else
                stack = this.loadReferenceStack(3);
            end
        end
        
        %LOAD WHITE IMAGE STACK
        %Return stack of white images
        %PARAMETERS:
            %range (optional): vector of indices of images requested, if
            %empty return the full range
        function stack = loadWhiteStack(this,range)
            if nargin == 2
                stack = this.loadReferenceStack(2,range);
            else
                stack = this.loadReferenceStack(2);
            end
        end
        
    end
    
end

