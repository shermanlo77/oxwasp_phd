%CLASS: IMAGE SCALE WITH SIGNIFICANT PIXELS
%Class for plotting an image, with certain pixels coloured in
%
%HOW TO USE:
  %pass the image through the constructor
  %pass a boolean image through the method addSigPixels
  %use the method plot() to plot the image with coloured in significant pixels
classdef Imagesc < handle
  
  %MEMBER VARIABLES
  properties (SetAccess = private)
    image; %image
    clim; %2 column vector, limit of the image values
    dilateSize = 1; %scalar, how much to dilate the boolean significant image
    positiveImage; %image of booleans, true if want that pixel to be coloured in
    %3 row vector, contain numbers between 0 and 1, define the colour of the significant pixels
    positiveColour = [1,0,0];
    posterScale = 5;
    isPoster = false;
  end
  
  %METHODS
  methods (Access = public)
    
    %CONSTRUCTOR
    %PARAMETERS:
      %image: image to be plotted
    function this = Imagesc(image)
      %assign member variables
      this.image = image;
      this.clim = [0,0];
      this.clim(1) = min(min(image));
      this.clim(2) = max(max(image));
    end
    
    %METHOD: TURN ON POSTER MODE
    function turnOnPoster(this)
      this.isPoster = true;
    end
    
    %METHOD: SET C LIM
    function setCLim(this, clim)
      this.clim = clim;
    end
    
    %METHOD: SET DILATE SIZE
    %PARAMETERS:
      %dilateSize: size of the dilation of the positiveImage
    function setDilateSize(this, dilateSize)
      this.dilateSize = dilateSize;
    end
    
    %METHOD: ADD POSITIVE PIXELS
    %PARAMETERS:
      %positiveImage: image of booleans, true if that pixel is positive
    function addPositivePixels(this, positiveImage)
      this.positiveImage = positiveImage;
    end
    
    %METHOD: PLOT
    %Plots this.image using imagesc
    %colour in significant pixels, indicated by this.positiveImage
    function im = plot(this)
      %get the colour map
      colourMap = colormap;
      %get the number of steps in this colour map
      nColourStep = numel(colourMap(:,1));
      %get the dimensions of the image and the area
      [height,width] = size(this.image);
      area = height*width;
      
      %declare a height x width x 3 RGB matrix
      imagePlot = zeros(height,width,3);
      
      %make a copy of the image
      %any pixels below and above clim, adjust them to equal the boundary
      imageTruncate = this.image;
      imageTruncate(imageTruncate>this.clim(2)) = this.clim(2);
      imageTruncate(imageTruncate<this.clim(1)) = this.clim(1);
      
      %for each colour
      for i = 1:3
        %interpolate the colourmap
        imagePlot(:,:,i) = interp1(linspace(this.clim(1), this.clim(2), nColourStep), ...
            colourMap(:,i), imageTruncate);
      end
      
      %if positiveImage is not empty, ie it has a value
      if ~isempty(this.positiveImage)
        
        %get the significant pixels
        positivePlot = this.positiveImage;
        %if the dilation size is not zero
        if this.dilateSize ~= 1
          %dilate the significant map
          positivePlot = imdilate(positivePlot,strel('square',this.dilateSize));
        end
        %for each colour
        for i = 1:3
          %change the colour of the significant pixels
          imagePlot(find(positivePlot)+(i-1)*area) = this.positiveColour(i);
        end
      end
      
      %if this is a poster, resize it
      if this.isPoster
        imagePlot = imresize(imagePlot,this.posterScale);
      end
      
      %plot the image
      im = imshow(imagePlot, 'InitialMagnification', 'fit');
      %plot the colour bar and set it
      colorbar;
      im.Parent.CLim = this.clim;
      
    end
    
  end
  
end

