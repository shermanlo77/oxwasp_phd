%CLASS: IMAGE SCALE WITH SIGNIFICANT PIXELS
%Class for plotting an image, with certain pixels coloured in
%
%HOW TO USE:
    %pass the image through the constructor
    %pass a boolean image through the method addSigPixels
    %use the method plot() to plot the image with coloured in significant pixels
classdef ImagescSignificant < handle
    
    %MEMBER VARIABLES
    properties (SetAccess = private)
        image; %image
        clim; %2 column vector, limit of the image values
        dilate_size; %scalar, how much to dilate the boolean significant image
        sig_image; %image of booleans, true if want that pixel to be coloured in
        sig_color; %3 row vector, contain numbers between 0 and 1, define the colour of the significant pixels
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %image: image to be plotted
        function this = ImagescSignificant(image)
            %assign member variables default values
            this.image = image;
            this.clim = [0,0];
            this.clim(1) = min(min(image));
            this.clim(2) = max(max(image));
            this.dilate_size = 1;
            this.sig_color = [1,0,0];
        end
        
        %METHOD: SET DILATE SIZE
        %PARAMETERS:
            %dilate_size: size of the dilation of the sig_image
        function setDilateSize(this, dilate_size)
            this.dilate_size = dilate_size;
        end
        
        %METHOD: ADD SIGNIFICANT PIXELS
        %PARAMETERS:
            %sig_image: image of booleans, true if that pixel is significant
        function addSigPixels(this, sig_image)
            this.sig_image = sig_image;
        end
        
        %METHOD: PLOT
        %Plots this.image using imagesc
        %colours in significant pixels, indicated by this.sig_image
        function plot(this)
            %plot the image
            im = imagesc(this.image, this.clim);
            
            %if sig_image is not empty, ie it has a value
            if ~isempty(this.sig_image)
                
                %get the significant pixels
                sig_plot = this.sig_image;
                %if the dilation size is not zero
                if this.dilate_size ~= 1
                    %dilate the significant map
                    sig_plot = imdilate(sig_plot,strel('square',this.dilate_size));
                end
                %make the significant pixels transparent
                im.AlphaData = ~sig_plot;
                %set the background of the image ot sig_color
                im.Parent.Color = this.sig_color;
            end
            
            %plot the colour bar and remove the x and y ticks
            colorbar;
            im.Parent.XTick = [];
            im.Parent.YTick = [];
        end
        
    end
    
end

