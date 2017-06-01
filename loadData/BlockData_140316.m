classdef BlockData_140316 < Scan
    %BLOCKDATA
    %Class for obtaining images for the 140316 data
    
    %MEMBER VARIABLES
    properties        
        
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        function this = BlockData_140316()
            
            this@Scan('data/140316/', 'block_', 1996, 1996, 100);
            %assign member variables
            this.panel_counter = PanelCounter_Brass();
            this.min_greyvalue = 5.7588E3;
            
            reference_image_array(3) = Scan();
            this.reference_image_array = reference_image_array;
            this.reference_image_array(1) = Scan('data/140316_bgw/black/', 'black_140316_', this.width, this.height, 20);
            this.reference_image_array(2) = Scan('data/140316_bgw/white/', 'white_140316_', this.width, this.height, 20);
            this.reference_image_array(3) = Scan('data/140316_bgw/grey/', 'grey_140316_', this.width, this.height, 20);
        end
        
        %GET SAMPLE MEAN VARIANCE DATA (using top half of the images)
        %PARAMETERS:
            %index (optional): vector of indices of images requested to be
            %used in mean and variance estimation, if not provided all
            %images shall be considered
        function [sample_mean,sample_var] = getSampleMeanVar_topHalf(this,index)

            %if index not provided, index points to all images
            if nargin == 1
                index = 1:this.n_sample;
            end
            
            %load the stack of images, indicated by the vector index
            stack = this.loadSampleStack(index);
            %crop the stack, keeping the top half
            stack = stack(1:(this.height/2),:,:);
            %work out the sample mean and convert it to a vector
            sample_mean = reshape(mean(stack,3),[],1);
            %work out the sample variance and convert it to a vector
            sample_var = reshape(var(stack,[],3),[],1);

        end

    end
    
    methods (Static)
        
        %GET THRESHOLD TOP HALF
        %Return a logical matrix which segments the sample from the
        %background of the top half of the scans. 1 indicate the
        %background, 0 for the sample.
        %
        %Method: does shading correction with median filter applied to the
        %reference images, take the mean of the shading corrected images
        %and then threshold at 4.7E4
        function threshold = getThreshold_topHalf()
            
            %load the data
            block_data = BlockData_140316();

            %add shading correction
            block_data.addShadingCorrector(ShadingCorrector_median([3,3,3]));

            %load the images
            slice = block_data.loadSampleStack();
            %crop the images to keep the top half
            slice = slice(1:(round(block_data.height/2)),:,:);
            %take the mean over all images
            slice = mean(slice,3);
            %remove dead pixels
            slice = removeDeadPixels(slice);

            %indicate pixels with greyvalues more than 4.7E4
            threshold = slice>4.7E4;
        end
    end
    
end
