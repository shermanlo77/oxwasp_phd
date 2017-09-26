classdef AbsBlock_Mar16 < Scan
    %BLOCKDATA
    %Class for obtaining images for the 140316 data
    
    %MEMBER VARIABLES
    properties        
        
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        function this = AbsBlock_Mar16()
            this@Scan('data/absBlock_Mar16/', 'block_', 1996, 1996, 100, 100, 33, 500);
            %assign member variables
            this.panel_counter = PanelCounter_Brass();
            %this.min_greyvalue = 5.7588E3;
        end
        
        %OVERRIDE: SEGMENTATION ONLY TOP HALF
        function segmentation = getSegmentation(this)
            segmentation = this.getSegmentation@Scan();
            segmentation((floor(this.height/2)+1):end,:) = false;
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
            stack = this.loadImageStack(index);
            %crop the stack, keeping the top half
            stack = stack(1:(this.height/2),:,:);
            %work out the sample mean and convert it to a vector
            sample_mean = reshape(mean(stack,3),[],1);
            %work out the sample variance and convert it to a vector
            sample_var = reshape(var(stack,[],3),[],1);

        end
        
        

    end
    
end
