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
        poster_enlarge;
        is_poster;
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
            this.poster_enlarge = 5;
            this.is_poster = false;
        end
        
        function turnOnPoster(this)
            this.is_poster = true;
        end
        
        function setCLim(this, clim)
          this.clim = clim;
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
        function im = plot(this)
            %get the colour map
            colour_map = colormap;
            %get the number of steps in this colour map
            n_colour_step = numel(colour_map(:,1));
            %get the dimensions of the image and the area
            [height,width] = size(this.image);
            area = height*width;
            
            %declare a height x width x 3 RGB matrix
            image_plot = zeros(height,width,3);
            
            %make a copy of the image
            %any pixels below and above clim, adjust them to equal the boundary
            imageTruncate = this.image;
            imageTruncate(imageTruncate>this.clim(2)) = this.clim(2);
            imageTruncate(imageTruncate<this.clim(1)) = this.clim(1);
            
            %for each colour
            for i = 1:3
                %interpolate the colourmap
                image_plot(:,:,i) = interp1(linspace(this.clim(1),this.clim(2),n_colour_step),colour_map(:,i),imageTruncate);
            end
            
            %if sig_image is not empty, ie it has a value
            if ~isempty(this.sig_image)
                
                %get the significant pixels
                sig_plot = this.sig_image;
                %if the dilation size is not zero
                if this.dilate_size ~= 1
                    %dilate the significant map
                    sig_plot = imdilate(sig_plot,strel('square',this.dilate_size));
                end
                %for each colour
                for i = 1:3
                    %change the colour of the significant pixels
                    image_plot(find(sig_plot)+(i-1)*area) = this.sig_color(i);
                end
            end
            
            if this.is_poster
                image_plot = imresize(image_plot,this.poster_enlarge);
            end
                
            %plot the image
            im = imshow(image_plot,'InitialMagnification','fit');   
            %plot the colour bar and set it
            colorbar;
            im.Parent.CLim = this.clim;

        end
        
    end
    
end

