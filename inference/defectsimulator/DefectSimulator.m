%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: DEFECT SIMULATOR
%Class for generating a Gaussian image. Subclass can cotaminate and/or defect the Gaussian image,
    %this class contains methods to cotaminate and/or defect the image
%
%HOW TO IMPLEMENT:
  %Use the contructor to pass a rng
  %Implement the method getDefectedImage which returns the desired image
%NOTES:
  %In this version, when defecting a pixel, the pixel is multiplied by altStd and added by altMean,
      %thus caution should be used when adding multiple defects as overlapping defects will cause
      %unexpected results
%HOW THE USER SHOULD USE SUB CLASSES:
  %Pass a rng in the constructor
  %Call image = getDefectedImage(size)
classdef DefectSimulator < handle
  
  %MEMBER VARIABLES
  properties (SetAccess = private)
    randStream; %rng
  end
  
  properties (SetAccess = protected)
    isContaminated; %boolean, indicate if this contaminates the image
  end
  
  %METHODS
  methods (Access = public)
    
    %CONSTRUCTOR
    %PARAMETERS:
      %randStream: rng
    function this = DefectSimulator(randStream)
      %assign member variables
      this.randStream = randStream;
    end
    
    %METHOD: GET DEFECTED IMAGE
    %Returns a Gaussian image
    %Subclasses can override this method and instead return an image with all N(0,1) pixels except
        %for the alt pixels this is then followed by adding or multiplying by smooth functions
    %PARAMETER:
      %size: 2 row vector [height, width]
    %RETURN:
      %image: a Gaussian image (subclasses can contaminate or defect it)
      %isNonNullImage: boolean map, true if that pixel is a defect
    function [image, isNonNullImage] = getDefectedImage(this, size)
      image = this.randStream.randn(size);
      isNonNullImage = false(size);
    end
    
  end
  
  methods (Access = protected)
    
    %METHOD: ADD SQUARE DEFECT
    %Replace a square with samples from the alt distribution
    %PARAMETERS:
      %image: image to be defected
      %isNonNullImage: boolean image, true for defect
      %coOd: coordinate of the middle of the square
      %defectSize: 2 vector defining the size of the square defect
      %mean: mean parameter of the alt distribution
      %std: std parameter of the alt distribution
    %RETURN:
      %image: the defected image
      %isNonNullImage: boolean image, true for defect
    function [image, isNonNullImage] = addSquareDefect(this, image, isNonNullImage, coOd, ...
        defectSize, mean, std)
      %get the range of columns and rows to fill with a defect
      rowIndex = this.getRange(coOd(1), defectSize(1));
      columnIndex = this.getRange(coOd(2), defectSize(2));
      [rowIndex, columnIndex] = this.checkBoundary(image, rowIndex, columnIndex);
      %set the square to have samples from the alt distribution
      image(rowIndex, columnIndex) = image(rowIndex, columnIndex) * std + mean;
      %set the square in this.altImage to be true
      isNonNullImage(rowIndex, columnIndex) = true;
    end
    
    %METHOD: ADD LINE DEFECT
    %Replace a verticle with samples from the alt distribution
    %PARAMETERS:
      %image: the image to be defected
      %isNonNullImage: boolean image, true for defect
      %x: x coordinate of the center of the line
      %thickness: the thickness of the line
      %mean: mean parameter of the alt distribution
      %std: std parameter of the alt distribution
    %RETURN:
      %image: the defected image
      %isNonNullImage: boolean image, true for defect
    function [image, isNonNullImage] = addLineDefect(this, image, isNonNullImage, x, thickness, ...
          mean, std)
      %get the column index which the defect is to be added
      defectColumn = this.getRange(x, thickness);
      defectColumn = this.checkColumnBoundary(image, defectColumn);
      %set the line to be samples from the alt distribution
      image(:,defectColumn) = image(:,defectColumn) * std + mean;
      %set the line in this.altImage to true
      isNonNullImage(:, defectColumn) = true;
    end
    
    %METHOD: ADD DUST
    %Randommly select pixel with probability p, these selected pixels are alt
    %PARAMETERS:
      %image: the image to be defected
      %isNonNullImage: boolean image, true for defect
      %p: probability pixel is alt
      %mean: mean parameter of the alt distribution
      %std: std parameter of the alt distribution
    %RETURN:
      %image: the defected image
      %isNonNullImage: boolean image, true for defect
    function [image, isNonNullImage] = addDust(this, image, isNonNullImage, p, mean, std)
      isDust = this.randStream.rand(size(image)) < p;
      isNonNullImage(isDust) = true;
      image(isDust) = image(isDust) * std + mean;
    end
    
    %METHOD: ADD PLANE
    %Add a gradient, value of 0 in the middle
    %PARAMETERS:
      %image: image to be defected
      %grad: 2 vector, the gradient of the plane
    %RETURN:
      %image: the defected image
    function image = addPlane(this, image, grad)
      %mesh grid of the image
      [height, width] = size(image);
      [xGrid, yGrid] = meshgrid(1:width, 1:height);
      %calculate the value of the plane for each x and y
      plane = grad(2) * (xGrid - width/2) + grad(1) * (yGrid - height/2);
      %add the plane to this.defect_image
      image = image + plane;
    end
    
    %METHOD: MULTIPLY
    %PARAMETERS:
      %image: image to be defected
      %multiplier: the image is multipled by this
    %RETURN:
      %image: the image multiplied by the multiplier
    function image = multiply(this, image, multiplier)
      image = image * multiplier;
    end
    
    %METHOD: ADD SINUSOID
    %Add a sinusoid
    %PARAMETERS:
      %image: image to be defected
      %amplitude: amplitude of the sinusoid
      %wavelength: 2 column vector defining the wavelength of [y,x] direction, can be negative
      %angularOffset: offset the sinusoid in radians,
          %for angularOffset = 0 the middle of the image = 0
    %RETURN:
      %image: the defected image
    function image = addSinusoid(this, image, amplitude, wavelength, angularOffset)
      %meshgrid of the the image
      [height, width] = size(image);
      [xGrid, yGrid] = meshgrid(1:width, 1:height);
      %shift the grid so that the middle is the origin
      xGrid = xGrid - width/2;
      yGrid = yGrid - height/2;
      %convert the wavelength to a frequency
      f = 1./wavelength;
      %work out the value of the sinusoid for each x and y
      sinusoid = amplitude * sin( 2*pi*(f(1)*yGrid + f(2)*xGrid) + angularOffset);
      %add the sinusoid to the image
      image = image + sinusoid;
    end
    
    %METHOD: GET RANGE
    %Get the list of index given the coordinates of the center and the length
    %PARAMETERS:
      %centreCood: centre of the range
      %length: length of the range
    %RETURN:
      %range: length number of integers, where centre_cood is in the middle
    function range = getRange(this, centreCood, length)
      %if the length is odd
      if mod(length,2)
        %get the range of integers
        %example: XXOXX where O is the centre
        range = (centreCood - (length-1)/2) : (centreCood + (length-1)/2);
        %else the length is even
      else
        %get the range of integers, including the middle and cutting the right hand side
        %example: XXOX where O is the centre
        range = (centreCood - length/2) : (centreCood + length/2 - 1);
      end
    end
    
    %METHOD: CHECK BOUNDARY
    %Given indices for the rows and columns, remove the ones which are outside the boundary
    %PARAMETERS:
      %image: image to check the boundary on
      %rowIndex: indices of rows
      %columnIndex: indicies of columns
    %RETURN:
      %rowIndex: row_index with boundary check
      %columnIndex: column_index with bounday check
    function [rowIndex, columnIndex] = checkBoundary(this, image, rowIndex, columnIndex)
      %check the boundary of the rows and columns and return it
      rowIndex = this.checkRowBoundary(image, rowIndex);
      columnIndex = this.checkColumnBoundary(image, columnIndex);
    end
    
    %METHOD: CHECK ROW BOUNDARY
    %Given indices for the rows, remove the ones which are outside the boundary
    %PARAMETERS:
      %image: image to check the boundary on
      %rowIndex: indices of rows
    %RETURN:
      %rowIndex: rowIndex with boundary check
    function rowIndex = checkRowBoundary(this, image, rowIndex)
      %remove the rows where it is equal and below 0 and bigger than the height
      indexRemove = (rowIndex <= 0) | (rowIndex > numel(image(:,1)));
      rowIndex(indexRemove) = [];
    end
    
    %METHOD: CHECK COLUMN BOUNDARY
    %Given indices for the columns, remove the ones which are outside the boundary
    %PARAMETERS:
      %image: image to check the boundary on
      %columnIndex: indicies of columns
    %RETURN:
      %columnIndex: columnIndex with bounday check
    function columnIndex = checkColumnBoundary(this, image, columnIndex)
      %remove the columns where it is equal and below 0 and bigger than the width
      index_remove = (columnIndex <= 0) | (columnIndex > numel(image(1,:)));
      columnIndex(index_remove) = [];
    end
    
  end
  
end

