classdef BlockData < handle
    %BLOCKDATA Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        width;
        height;
        area;
        n_black;
        n_white;
        n_grey;
        n_sample
    end
    
    methods
        
        function this = BlockData()
            this.width = 1996;
            this.height = 1996;
            this.area = 1996^2;
            this.n_black = 20;
            this.n_white = 20;
            this.n_grey = 20;
            this.n_sample = 100;
        end
        
        function slice = loadBlack(this,folder_location,index)
            slice = imread(strcat(folder_location,'/black_140316_',num2str(index),'.tif'));
        end
        
        function slice = loadGrey(this,folder_location,index)
            slice = imread(strcat(folder_location,'/grey_140316_',num2str(index),'.tif'));
        end
        
        function slice = loadWhite(this,folder_location,index)
            slice = imread(strcat(folder_location,'/white_140316_',num2str(index),'.tif'));
        end
        
        function slice = loadSample(this,folder_location,index)
            slice = imread(strcat(folder_location,'/block_',num2str(index),'.tif'));
        end
        
        function stack = loadBlackStack(this,folder_location,range)
            if nargin == 2
                range = 1:this.n_black;
            end
            stack = zeros(this.height,this.width,numel(range));
            for index = 1:numel(range)
                stack(:,:,index) = this.loadBlack(folder_location,range(index));
            end
        end
        
        function stack = loadGreyStack(this,folder_location,range)
            if nargin == 2
                range = 1:this.n_grey;
            end
            stack = zeros(this.height,this.width,numel(range));
            for index = 1:numel(range)
                stack(:,:,index) = this.loadGrey(folder_location,range(index));
            end
        end
        
        function stack = loadWhiteStack(this,folder_location,range)
            if nargin == 2
                range = 1:this.n_white;
            end
            stack = zeros(this.height,this.width,numel(range));
            for index = 1:numel(range)
                stack(:,:,index) = this.loadWhite(folder_location,range(index));
            end
        end
        
        function stack = loadSampleStack(this,folder_location,range)
            if nargin == 2
                range = 1:this.n_sample;
            end
            stack = zeros(this.height,this.width,numel(range));
            for index = 1:numel(range)
                stack(:,:,index) = this.loadSample(folder_location,range(index));
            end
        end
        
    end
    
end

