classdef BlockData < handle
    %BLOCKDATA
    %Class for obtaining images for the 140316 data
    
    %MEMBER VARIABLES
    properties
        width; %width of the image
        height; %height of the image
        area; %area of the image
        n_black; %number of black images
        n_white; %number of white images
        n_grey; %number of grey images
        n_sample; %number of sample images
        black_name; %'/' + name of black image file excluding number at the end
        white_name; %... name of white image ...
        grey_name; %... name of grey image ...
        sample_name; %... name of sample image
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        function this = BlockData()
            %assign member variables
            this.width = 1996;
            this.height = 1996;
            this.area = 1996^2;
            this.n_black = 20;
            this.n_white = 20;
            this.n_grey = 20;
            this.n_sample = 100;
            this.black_name = '/black_140316_';
            this.white_name = '/white_140316_';
            this.grey_name = '/grey_140316_';
            this.sample_name = '/block_';
        end
        
        %LOAD BLACK IMAGE
        %Return a black image
        %PARAMETERS
            %folder_location: string targetting location of the black image
            %index: index of image
        function slice = loadBlack(this,folder_location,index)
            slice = imread(strcat(folder_location,this.black_name,num2str(index),'.tif'));
        end
        
        %LOAD GREY IMAGE
        %Return a grey image
        %PARAMETERS
            %folder_location: string targetting location of the grey image
            %index: index of image
        function slice = loadGrey(this,folder_location,index)
            slice = imread(strcat(folder_location,this.grey_name,num2str(index),'.tif'));
        end
        
        %LOAD WHITE IMAGE
        %Return a white image
        %PARAMETERS
            %folder_location: string targetting location of the white image
            %index: index of image
        function slice = loadWhite(this,folder_location,index)
            slice = imread(strcat(folder_location,this.white_name,num2str(index),'.tif'));
        end
        
        %LOAD SAMPLE IMAGE
        %Return a sample image
        %PARAMETERS
            %folder_location: string targetting location of the sample image
            %index: index of image
        function slice = loadSample(this,folder_location,index)
            slice = imread(strcat(folder_location,this.sample_name,num2str(index),'.tif'));
        end
        
        %LOAD BLACK IMAGE STACK
        %Return stack of black images
        %PARAMETERS:
            %folder_location: string targetting location of the black image
            %range (optional): vector of indices of images requested, if
            %empty return the full range
        function stack = loadBlackStack(this,folder_location,range)
            %if range not provided, provide the full range
            if nargin == 2
                range = 1:this.n_black;
            end
            %declare stack of images
            stack = zeros(this.height,this.width,numel(range));
            %for each image, put it in the stack
            for index = 1:numel(range)
                stack(:,:,index) = this.loadBlack(folder_location,range(index));
            end
        end
        
        %LOAD GREY IMAGE STACK
        %Return stack of grey images
        %PARAMETERS:
            %folder_location: string targetting location of the grey image
            %range (optional): vector of indices of images requested, if
            %empty return the full range
        function stack = loadGreyStack(this,folder_location,range)
            %if range not provided, provide the full range
            if nargin == 2
                range = 1:this.n_grey;
            end
            %declare stack of images
            stack = zeros(this.height,this.width,numel(range));
            %for each image, put it in the stack
            for index = 1:numel(range)
                stack(:,:,index) = this.loadGrey(folder_location,range(index));
            end
        end
        
        %LOAD WHITE IMAGE STACK
        %Return stack of white images
        %PARAMETERS:
            %folder_location: string targetting location of the white image
            %range (optional): vector of indices of images requested, if
            %empty return the full range
        function stack = loadWhiteStack(this,folder_location,range)
            %if range not provided, provide the full range
            if nargin == 2
                range = 1:this.n_white;
            end
            %declare stack of images
            stack = zeros(this.height,this.width,numel(range));
            %for each image, put it in the stack
            for index = 1:numel(range)
                stack(:,:,index) = this.loadWhite(folder_location,range(index));
            end
        end
        
        %LOAD SAMPLE IMAGE STACK
        %Return stack of sample images
        %PARAMETERS:
            %folder_location: string targetting location of the sample image
            %range (optional): vector of indices of images requested, if
            %empty return the full range
        function stack = loadSampleStack(this,folder_location,range)
            %if range not provided, provide the full range
            if nargin == 2
                range = 1:this.n_sample;
            end
            %declare stack of images
            stack = zeros(this.height,this.width,numel(range));
            %for each image, put it in the stack
            for index = 1:numel(range)
                stack(:,:,index) = this.loadSample(folder_location,range(index));
            end
        end
        
    end
    
end

