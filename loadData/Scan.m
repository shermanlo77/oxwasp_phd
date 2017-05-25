classdef Scan < handle
    %SCAN Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        width; %width of the image
        height; %height of the image
        area; %area of the image
        n_sample; %number of images
        file_name; %... name of sample image
        folder_location; %location of the dataset
        reference_image_array; %array of reference scan objects
        panel_counter %panel counter object
        shading_corrector; %shading corrector object
        want_shading_correction; %boolean, true to do shading correction, default false, automatically turns to true if a shading corrector is added
        want_remove_dead_pixels; %boolean, true to remove dead pixels, default false
        
        min_greyvalue; %minimum possible greyvalue
    end
    
    methods
        
        %CONSTRUCTOR
        function this = Scan(folder_location, file_name, width, height, n_sample)
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
            %index: index of image
        function slice = loadSample(this,index)
            slice = imread(strcat(this.folder_location,this.file_name,num2str(index),'.tif'));
            slice = this.shadingCorrection(double(slice));
        end
        
        %LOAD SAMPLE IMAGE STACK
        %Return stack of sample images
        %PARAMETERS:
            %range (optional): vector of indices of images requested, if
            %empty return the full range
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
        %Instantise a new shading corrector to the member variable and calibrate it
        %for shading correction
        %PARAMETERS:
            %shading_corrector_class: function name or function handle of
            %a ShadingCorrector class
            %want_grey: boolean, true if want to consider grey images
            %parameters (optional): a vector of parameters for panel fitting (one for each reference image)
        function addShadingCorrector(this,shading_corrector_class,want_grey,parameters)
            
            %turn off shading correction to obtain the reference images
            this.turnOffShadingCorrection();
            
            %declare an array reference_stack which stores the mean black and mean
            %white image
            reference_stack = zeros(this.height,this.width,2+want_grey);
            %load and save the mean black image
            reference_stack(:,:,1) = mean(this.reference_image_array(1).loadSampleStack(),3);
            %load and save the mean white image
            reference_stack(:,:,2) = mean(this.reference_image_array(2).loadSampleStack(),3);
            %if want_grey, load and save the mean grey image
            if want_grey
                reference_stack(:,:,3) = mean(this.reference_image_array(3).loadSampleStack(),3);
            end

            %instantise a shading corrector and set it up using reference images
            if nargin == 3
                shading_corrector_temp = feval(shading_corrector_class,reference_stack);
            elseif nargin == 4
                shading_corrector_temp = feval(shading_corrector_class,reference_stack, this.panel_counter, parameters);
            end
            
            this.addManualShadingCorrector(shading_corrector_temp);
            
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
        
        function slice = loadReference(this,reference_index,image_index)
            slice = this.reference_image_array(reference_index).loadSampleStack(image_index);
        end
        
        function stack = loadReference_stack(this,reference_index,image_index)
            %if range not provided, provide the full range
            if nargin == 2
                image_index = 1:this.reference_image_array(reference_index).n_sample;
            end
            %declare stack of images
            stack = this.loadReference(this,reference_index,image_index);
        end
        
        %LOAD BLACK IMAGE
        %Return a black image
        %PARAMETERS
            %index: index of image
        function slice = loadBlack(this,index)
            slice = this.loadReference(1,index);
            slice = this.shadingCorrection(double(slice));
        end
        
        %LOAD GREY IMAGE
        %Return a grey image
        %PARAMETERS
            %index: index of image
        function slice = loadGrey(this,index)
            slice = this.loadReference(3,index);
            slice = this.shadingCorrection(double(slice));
        end
        
        %LOAD WHITE IMAGE
        %Return a white image
        %PARAMETERS
            %index: index of image
        function slice = loadWhite(this,index)
            slice = this.loadReference(2,index);
            slice = this.shadingCorrection(double(slice));
        end
        
        %LOAD BLACK IMAGE STACK
        %Return stack of black images
        %PARAMETERS:
            %range (optional): vector of indices of images requested, if
            %empty return the full range
        function stack = loadBlackStack(this,range)
            stack = this.loadReference(1,range);
        end
        
        %LOAD GREY IMAGE STACK
        %Return stack of grey images
        %PARAMETERS:
            %range (optional): vector of indices of images requested, if
            %empty return the full range
        function stack = loadGreyStack(this,range)
            stack = this.loadReference(3,range);
        end
        
        %LOAD WHITE IMAGE STACK
        %Return stack of white images
        %PARAMETERS:
            %range (optional): vector of indices of images requested, if
            %empty return the full range
        function stack = loadWhiteStack(this,range)
            stack = this.loadReference(2,range);
        end
        
    end
    
end

