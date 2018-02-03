%CLASS: MULTI GRID TESTER
%Class for using the grid tester for multiple translations of the grid
%Signficant pixels are stored in the member variables combined_sig_array and local_sig_array
%p values are stored in p_image_array
%
%How to use:
%Pass parameters and the z image in the constructor MultiGridTester(z_image, sub_height, sub_width, shift_array)
%Call the method doTest()
%Results are then stored in the member variables
classdef MultiGridTester < handle

    %MEMBER VARIABLES
    properties (SetAccess = private)
        sub_height; %height of square in grid
        sub_width; %width of square in grid
        size; %size of the test (0 uses default value)
        z_image; %image of z statistics
        
        %array of emperical null parameters
        mean_null_array; %emperical null mean parameter
        var_null_array; %emperical null variance parameter
        
        %array of boolean images, true for significant pixel
        combined_sig_array; %using combined p value analysis
        local_sig_array; %using local p value analysis
        p_image_array; %array of p value images
        n_shift; %number of translations of the grid to do
        
        %array of translations
            %dim 1 (size 2): for each dimension
            %dim 2 (size n_shift): for each translation
        shift_array;
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %z_image: image of z statistics
            %sub_height: height of square in grid
            %sub_width: width of square in grid
            %shift: two vector which translates the grid
                %each element is positive and cannot be more than [sub_height, sub_width]
        function this = MultiGridTester(z_image, sub_height, sub_width, shift_array)
            %assign member variables
            this.sub_height = sub_height;
            this.sub_width = sub_width;
            this.z_image = z_image;
            this.shift_array = shift_array; 
            this.n_shift = numel(shift_array(1,:));
            [height, width] = size(z_image);
            this.mean_null_array = zeros(height, width, this.n_shift);
            this.var_null_array = zeros(height, width, this.n_shift);
            this.p_image_array = zeros(height, width, this.n_shift);
            this.combined_sig_array = zeros(height, width, this.n_shift);
            this.local_sig_array = zeros(height, width, this.n_shift);
            this.size = 0;
        end
        
        %METHOD: SET SIZE
        %Set the size of the hypothesis test
        %PARAMETERS:
            %size: size of the test
        function setSize(this, size)
            this.size = size;
        end
        
        %METHOD: DO TEST
        %Does the multiple hypothesis test on the grid using each translation
        %PARAMETERS:
            %n_linspace: number of points to investigate to find the mode for emperical null
        function doTest(this, n_linspace)
            %for every shift
            for i_shift = 1:this.n_shift
                %get the translation vector
                shift = this.shift_array(:,i_shift);
                %instantise a grid tester
                grid_tester = GridTester(this.z_image, this.sub_height, this.sub_width, shift);
                %set the size of the test if required
                if this.size ~= 0
                    this.grid_tester.setSize(this.size);
                end
                %do the test
                grid_tester.doTest(n_linspace);
                %extract the emperical null parameters
                this.mean_null_array(:,:,i_shift) = grid_tester.mean_null;
                this.var_null_array(:,:,i_shift) = grid_tester.var_null;
                %extract the significant pixels and p values
                %save it in the member variable arrays
                this.combined_sig_array(:,:,i_shift) = grid_tester.combined_sig;
                this.local_sig_array(:,:,i_shift) = grid_tester.local_sig;
                this.p_image_array(:,:,i_shift) = grid_tester.p_image;
            end
        end
        
        %METHOD: MAJORITY LOCAL VOTE
        %Get boolean image, true if the majority of the translation voted significant
        %The FDR analaysis here are not combined
        %RETURN:
            %sig_image: boolen image, true if signficant
        function sig_image = majorityLocalVote(this)
            sig_image = this.majorityLocalVoteIndex(this.n_shift);
        end
        
        %METHOD: MAJORITY COMBINED VOTE
        %Get boolean image, true if the majority of the translation voted significant
        %The FDR analaysis here are combined
        %RETURN:
            %sig_image: boolen image, true if signficant
        function sig_image = majorityCombinedVote(this)
            sig_image = this.majorityCombinedVoteIndex(this.n_shift);
        end
        
        %METHOD: MAJORITY LOCAL VOTE WITH INDEX
        %Get boolean image, true if the majority of the translation (selected by index) voted significant
        %The FDR analaysis here are not combined
        %PARAMETERS:
            %index: vector of index 1 - this.n_shift
        %RETURN:
            %sig_image: boolen image, true if signficant
        function sig_image = majorityLocalVoteIndex(this, index)
            sig_image = sum(this.local_sig_array(:,:,index),3) >= round(numel(index)/2);
        end
        
        %METHOD: MAJORITY COMBINED VOTE
        %Get boolean image, true if the majority of the translation (selected by index) voted significant
        %The FDR analaysis here are combined
        %PARAMETERS:
            %index: vector of index 1 - this.n_shift
        %RETURN:
            %sig_image: boolen image, true if signficant
        function sig_image = majorityCombinedVoteIndex(this, index)
            sig_image = sum(this.combined_sig_array(:,:,index),3) >= round(numel(index)/2);
        end
        
    end
    
end

