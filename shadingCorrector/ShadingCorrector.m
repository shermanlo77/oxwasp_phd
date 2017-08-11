%SHADINGCORRECTOR Stores an array of reference scans and use it to do shading correction
%   Use the constructor to instantise a shading corrector
%
%   A reference scan is passed to the shading corrector via the method addScan().
%   Use the method calibrate() to work out parameters to do shading correction.
%   Shading correction is done via the method shadingCorrect()
%
%   Outlying pixels can be set to NaN by calling the method turnOnSetExtremeToNan()
%   Outlying pixels are pixels with greyvalues less than this.min_greyvalue and more than intmax('uint16')
classdef ShadingCorrector < handle
    
    %MEMBER VARIABLES
    properties
        
        %array of reference images (3 dimensions)
            %dim 1 and dim 2 for the image
            %dim 3: for each image
        reference_image_array; 
        image_size; %two vector representing the size of the image [height, width]
        n_image; %number of images in reference_image_array
        i_image;
        
        between_reference_mean; %image: between reference mean greyvalue
        global_mean; %scalar: mean of all greyvalues in reference_image_array
        b_array; %image of the gradients
        
        can_smooth; %boolean, false if it cannot smooth the images panel by panel in reference_image_array
        
        set_extreme_to_nan; %boolean, set extreme shading corrected values to nan, default false
        
        min_greyvalue; %the minimum possible greyvalue, used to determine if a greyvalue is small enough to be NaN
        
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        function this = ShadingCorrector()
            %assign member variables
            this.can_smooth = false;
            this.set_extreme_to_nan = false;
            this.min_greyvalue = 0;
        end
        
        %INITALISE
        %Set the member variables n_image, i_image, image_size, reference_image_array
        %PARAMETERS:
            %n_image: number of reference scans
            %height: height of the image
            %width: width of the image
        function initalise(this, n_image, height, width)
            this.n_image = n_image;
            this.i_image = 1;
            this.image_size = [height, width];
            this.reference_image_array = zeros(height, width, n_image);
        end
        
        %ADD SCAN
        %Add a reference scan to the shading correction
        %PARAMETERS:
            %scan: Scan object
            %index: integer vector, pointing to which images in that scan to use
        function addScan(this, scan, index)
            %if index is not supplied, load all images
            if nargin == 2
                reference_stack = scan.loadImageStack();
            %else load the images supplied by the parameter index
            else
                reference_stack = scan.loadImageStack(index);
            end
            %save the mean image stack of references for this stack
            this.reference_image_array(:,:,this.i_image) = mean(reference_stack,3);
            %increment the member variable i_image
            this.i_image = this.i_image + 1;
        end
        
        
        %CALIBRATE
        %Perpare statistics for shading correction
        function calibrate(this)
            this.n_image = this.i_image - 1;
            
            %declare vector (one element for each reference image) for the within image mean
            %this is the target greyvalue of the unshaded greyvalue for each reference image
            within_reference_mean = zeros(1,this.n_image);
            %for each image
            for i = 1:this.n_image
                %get the mean within image grey value and save it to within_reference_mean
                within_reference_mean(i) = mean(reshape(this.reference_image_array(:,:,i),[],1));
            end
            
            %target_image_array is a stack of this.n_image images
            %each image is completely one greyvalue, using the values in within_reference_mean
            target_image_array = repmat(reshape(within_reference_mean,1,1,[]),this.image_size);
            
            %between_reference_mean is an image representing the between reference image mean
            this.between_reference_mean = mean(this.reference_image_array,3);
            
            %global_mean is the mean of all greyvalues
            this.global_mean = mean(within_reference_mean);
            
            %work out the sum of squares of reference image - between reference mean
            %proportional to the between reference variance
            %s_xx is an images
            s_xx = sum((this.reference_image_array - repmat(this.between_reference_mean,1,1,this.n_image)).^2,3);
            %work out the covariance of between reference images and the target greyvalues
            %s_xy is an image
            s_xy = sum((this.reference_image_array - repmat(this.between_reference_mean,1,1,this.n_image)) .* (target_image_array - this.global_mean),3);
            
            %work out the gradient
            %b_array is an image
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
        function scan_image = shadingCorrect(this,scan_image)
            %use linear interpolation for shading correction
            scan_image = this.b_array .* (scan_image - this.between_reference_mean) + this.global_mean;
            
            %if this object requires extreme values to be replaced by NaN
            if this.set_extreme_to_nan
            
                %any negative grey values to be set to nan
                scan_image(scan_image < this.min_greyvalue) = nan;

                %set any overflow grey values to be nan
                scan_image(scan_image > intmax('uint16')) = nan;
                
            end
        end
        
        %RETURN THE NAME OF THIS SHADING CORRECTOR
        function name = getName(this)
            if this.n_image == 2
                name = 'bw';
            else
                name = 'linear';
            end
        end
        
    end
    
end

