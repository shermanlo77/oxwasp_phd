%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: IMAGE SCALE WITH SIGNIFICANT PIXELS
%Class for plotting an image, with certain pixels coloured in
%
%HOW TO USE:
  %pass the image through the constructor
  %pass a boolean image through the method addPositivePixels(positiveImage)
  %use the method plot() to plot the image with coloured in positive pixels
classdef Imagesc < handle
  
  %MEMBER VARIABLES
  properties (SetAccess = private)
    image; %image
    clim; %2 column vector, limit of the image values
    dilateSize = 1; %scalar, how much to dilate the boolean positive image
    positiveImage; %image of booleans, true if want that pixel to be coloured in
    %3 row vector, contain numbers between 0 and 1, define the colour of the positive pixels
    positiveColour = [1,0,0];
    posterScale = 5;
    isPoster = false;
    isBw = false; %boolean, true if black white
  end
  
  %METHODS
  methods (Access = public)
    
    %CONSTRUCTOR
    %PARAMETERS:
      %image: image to be plotted (matrix)
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
    
    %METHOD: SET TO BLACK WHITE
    function setToBw(this)
      this.isBw = true;
      this.positiveColour = [1,1,1];
    end
    
    %METHOD: PLOT
    %Plots this.image using imagesc
    %colour in significant pixels, indicated by this.positiveImage
    function im = plot(this)
      %get the colour map
      if (this.isBw)
        colourMap = colormap('gray');
      else
        colourMap = colormap;
      end
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
    
    %METHOD: ADD SCALE
    %Add scale bar to the imagesc
    %Optimise to be used with LatexFigure.sub();
    %To be called after plot();
    %
    %PARAMETERS:
      %scan: scan object containing magnification and detector resolution
      %lengthCm: the length of the scale bar in cm
      %scaleColour: colour of the scale bar using MATLAB notation, eg 'k' or [1,0,0]
    function addScale(this, scan, lengthCm, scaleColour)
      [imageHeight, imageWidth] = size(this.image);
      xStart = imageWidth * 0.01;
      lengthOfCm = scan.magnification * lengthCm * 1E-2 / scan.resolution;
      line([xStart, xStart+lengthOfCm], [imageHeight, imageHeight]*0.9,...
          'LineWidth',3,'Color',scaleColour);
      text(xStart+lengthOfCm/2,imageHeight*0.95,strcat(num2str(lengthCm),'cm'), ...
          'HorizontalAlignment','center','Color',scaleColour);
    end
    
    %METHOD: REMOVE LABEL SPACE
    %Adjust inner position when plotting projections
    %To be called after calling plot()
    function removeLabelSpace(this)
      ax = gca;
      ax.Position = [-0.03, 0.02, 0.9, 0.9];
    end
    
  end
  
end

