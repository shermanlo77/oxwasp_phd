classdef ShadingCorrector < handle
    %SHADINGCORRECTOR Stores an array of blank scans and use it to do
    %shading correction
    %   A stack of blank scans (white, grey, black images) is passed to the
    %   object via the constructor. Use the method calibrate() to work out
    %   parameters to do shading correction. The image to be shading
    %   corrected is passed to the method shadeCorrect()
    
    %MEMBER VARIABLES
    properties
        
        reference_image_array; %array of blank images
        image_size; %two vector representing the size of the image [height, width]
        n_image; %number of images in reference_image_array
        
        reference_mean; %vector of the mean greyvalue of each image in reference_image_array
        target_mean; %mean of all greyvalues in reference_image_array
        b_array; %image of the gradients
        
        can_smooth; %boolean, false, it cannot smooth
        
        set_extreme_to_nan; %boolean, set extreme shading corrected values to nan
        
        min_greyvalue; %the minimum possible greyvalue
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %PARAMETERS:
            %reference_image_array: stack of blank scans
        function this = ShadingCorrector(reference_image_array)
            %assign member variables
            this.reference_image_array = reference_image_array;
            %get the size of the images and the number of images in the stack
            [height,width,this.n_image] = size(reference_image_array);
            %assign member variable
            this.image_size = [height,width];
            this.can_smooth = false;
            this.set_extreme_to_nan = true;
            this.min_greyvalue = 0;
        end
        
        %CALIBRATE
        %Perpare statistics for shading correction
        function calibrate(this)
            %declare vector (one element for each image)
            target_array = zeros(1,this.n_image);
            %for each image
            for i_image = 1:this.n_image
                %get the mean grey value and save it to target_array
                target_array(i_image) = mean(reshape(this.reference_image_array(:,:,i_image),[],1));
            end
            
            %target_image_array is a stack of this.n_image images
            %each image is completely one greyvalue, taking values in
            %this.mean_population_array
            target_image_array = repmat(reshape(target_array,1,1,[]),this.image_size);
            
            %reference_mean_array is an image, each pixel is the sample
            %mean over all reference images
            this.reference_mean = mean(this.reference_image_array,3);
            
            %target_mean is the mean of (the mean greyvalues of each image)
            this.target_mean = mean(target_array);
            
            %work out the sum of squares of reference image - reference mean
            s_xx = sum((this.reference_image_array - repmat(this.reference_mean,1,1,this.n_image)).^2,3);
            %work out the covariance between reference nad target
            s_xy = sum((this.reference_image_array - repmat(this.reference_mean,1,1,this.n_image)) .* (target_image_array - this.target_mean),3);
            
            %work out the gradient
            this.b_array = s_xy./s_xx;
                
        end
        
        %TURN ON SET EXTREME TO NAN
        function turnOnSetExtremeToNan(this)
            this.set_extreme_to_nan = true;
        end
        
        %TURN OFF SET EXTREME TO NAN
        function turnOffSetExtremeToNan(this)
            this.set_extreme_to_nan = false;
        end
        
        %SHADE CORRECT
        %PARAMETERS:
            %scan_image: image to be shading corrected
        function scan_image = shadeCorrect(this,scan_image)
            %use linear interpolation for shading correction
            scan_image = this.b_array .* (scan_image - this.reference_mean) + this.target_mean;
            
            %if this object requires extreme values to be replaced by NaN
            if this.set_extreme_to_nan
            
                %any negative grey values to be set to nan
                scan_image(scan_image < this.min_greyvalue) = nan;

                %set any overflow grey values to be nan
                scan_image(scan_image > intmax('uint16')) = nan;
                
            end
        end
        
    end
    
end

