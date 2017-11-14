classdef GridTester < handle
    
    %MEMBER VARIABLES
    properties (SetAccess = protected)
        
        sub_height; %height of square in grid
        sub_width; %width of square in grid
        
        shift; %two vector which translates the grid, each element is positive and cannot be more than [sub_height, sub_width]
        %zero shift will have the grid and top left square top left corner algined
        
        z_image; %image of z statistics
        z0_image; %image of z statistics, corrected for the null hypothesis
        p_image; %image of p values, corrected for the null hypothesis
        z_tester_array; %cell array of ZTester objects, one for each grid
        n_row; %number of rows in the grid
        n_col; %number of columns in the grid
        height; %height of the z_image
        width; %width of the z_image
        
        combined_sig; %boolean image, highlighted significant pixels using combined FDR analysis
        local_sig; %boolean image, highlighted significant pixels using the local FDR analysis
        
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
        function this = GridTester(z_image, sub_height, sub_width, shift)
            %assign member variables
            this.sub_height = sub_height;
            this.sub_width = sub_width;
            this.z_image = z_image;
            this.shift = shift;
            [this.height, this.width] = size(z_image);
            this.n_row = ceil((this.height+this.shift(1))/this.sub_height);
            this.n_col = ceil((this.width+this.shift(2))/this.sub_width);
            this.z0_image = nan(this.height, this.width);
            this.p_image = zeros(this.height, this.width);
            this.combined_sig = zeros(this.height, this.width);
            this.local_sig = zeros(this.height, this.width);
            
            %assign z_tester_array
            %this is an array of ZTester objects, one for each square in the grid
            this.z_tester_array = cell(this.n_row, this.n_col);
            
            %for each column
            for i_col = 1:this.n_col
                %for each row
                for i_row = 1:this.n_row
                    %get the coordinates of the square
                    [top_left, bottom_right] = this.getGridCoordinates(i_row, i_col);
                    %extract the z statistics in that square and instantise ZTester using that
                    this.z_tester_array{i_row,i_col} = ZTester(this.z_image(top_left(1):bottom_right(1), top_left(2):bottom_right(2)));
                end
            end
            
        end
        
        %METHOD: GET GRID COORDINATES
        %Returns the coordinates of the top left and bottom right corner of a square in the grid
        %All coordinates are inclusive to the area of the square so that they can be used for indexing
        %PARAEMETERS:
            %i_row: index pointing to the square living in the i_row
            %i_col: index pointing to the square living in the i_col
        %RETURN:
            %top_left: 2 column vector of the top_left corner of the specified square
            %bottom_right: 2 column vector of the bottom_right corner of the specified square
        function [top_left, bottom_right] = getGridCoordinates(this, i_row, i_col)
            
            %declare 2 column vectors for the top left and bottom right corners
            top_left = zeros(2,1);
            bottom_right = zeros(2,1);
            
            %get the coordinates of the top left corner
            top_left(1) = (i_row - 1) * this.sub_height + 1;
            top_left(2) = (i_col - 1) * this.sub_width + 1;
            
            %get the coordinates of the bottom right corner
            bottom_right(1) = i_row * this.sub_height;
            bottom_right(2) = i_col * this.sub_width;
            
            %shift the coordinates by this.shift
            top_left = top_left - this.shift;
            bottom_right = bottom_right - this.shift;
            
            %boundary check, if beyound left and top of the image
            %set that coordinate to 1
            top_left(top_left < 1) = 1;
            
            %boundary check, if beyound right and bottom of the image
            %set that coordinate to the size of the z_image
            if bottom_right(1) > this.height
                bottom_right(1) = this.height;
            end
            if bottom_right(2) > this.width
                bottom_right(2) = this.width;
            end
        end
        
        %METHOD: DO TEST
        function doTest(this, n_linspace)
            %for each column
            for i_col = 1:this.n_col
                %for each row
                for i_row = 1:this.n_row
                    %get the coordinates of the square
                    [top_left, bottom_right] = this.getGridCoordinates(i_row, i_col);
                    
                    %do the test
                    this.z_tester_array{i_row,i_col}.estimateNull(n_linspace);
                    this.z_tester_array{i_row,i_col}.doTest();
                    
                    %extract the local significant pixels
                    this.local_sig(top_left(1):bottom_right(1), top_left(2):bottom_right(2)) = this.z_tester_array{i_row,i_col}.sig_image;
                    
                    %extract the z statistics
                    this.z0_image(top_left(1):bottom_right(1), top_left(2):bottom_right(2)) = this.z_tester_array{i_row,i_col}.getZCorrected();
                end
            end
            %combine the z statistics together and do the test
            combined_tester = ZTester(this.z0_image);
            combined_tester.doTest();
            %save the significant pixels and the p values
            this.combined_sig = combined_tester.sig_image;
            this.p_image = combined_tester.p_image;
        end
        
        %METHOD: SET SIZE
        %Set the size of the test for each z_tester
        %PARAMETERS:
            %size: size of the test
        function setSize(this, size)
            %for each column
            for i_col = 1:this.n_col
                %for each row
                for i_row = 1:this.n_row
                    %set the size of this z_tester
                    this.z_tester_array{i_row,i_col}.setSize(size);
                end
            end
        end
        
    end
    
end

