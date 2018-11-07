%ABSTRACT CLASS: DEFECT SIMULATOR
%Class for adding smooth function and defecting pixels with samples from the alt distribution
%
%HOW TO IMPLEMENT:
  %Use the contructor to pass a rng
  %Implement the method getDefectedImage which returns N(0,1) image except for where there are
      %defects. Where there are defects, the pixels sample the alt distribution
%HOW THE USER SHOULD USE IT:
  %Pass a rng in the constructor
  %Call image = getDefectedImage(size)
classdef DefectSimulator < handle
  
  %MEMBER VARIABLES
  properties (SetAccess = private)
    randStream; %rng
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
    %Return an image with all N(0,1) pixels except for the alt pixels
    %This is then followed by adding or multiplying by smooth functions
    %In the superclass version, it returns a pure Gaussian image with no defects
    %PARAMETER:
      %size: 2 row vector [height, width]
    %RETURN:
      %image: a defected Gaussian image
      %isAltImage: boolean map, true if that pixel is a defect
    function [image, isAltImage] = getDefectedImage(this, size)
      image = this.randStream.randn(size);
      isAltImage = zeros(size);
    end
    
  end
  
  methods (Access = protected)
    
    %METHOD: ADD SQUARE DEFECT
    %Replace a square with samples from the alt distribution
    %PARAMETERS:
      %image: image to be defected
      %isAltImage: boolean image, true for defect
      %coOd: coordinate of the middle of the square
      %defectSize: 2 vector defining the size of the square defect
      %mean: mean parameter of the alt distribution
      %std: std parameter of the alt distribution
    %RETURN:
      %image: the defected image
      %isAltImage: boolean image, true for defect
    function [image, isAltImage] = addSquareDefect(this, image, isAltImage, coOd, defectSize, ...
        mean, std)
      %get the range of columns and rows to fill with a defect
      rowIndex = this.getRange(coOd(1), defectSize(1));
      columnIndex = this.getRange(coOd(2), defectSize(2));
      [rowIndex, columnIndex] = this.checkBoundary(image, rowIndex, columnIndex);
      %set the square to have samples from the alt distribution
      image(rowIndex, columnIndex) = ...
          this.randStream.randn(numel(rowIndex), numel(columnIndex)) * std + mean;
      %set the square in this.altImage to be true
      isAltImage(rowIndex, columnIndex) = true;
    end
    
    %METHOD: ADD LINE DEFECT
    %Replace a verticle with samples from the alt distribution
    %PARAMETERS:
      %image: the image to be defected
      %isAltImage: boolean image, true for defect
      %x: x coordinate of the center of the line
      %thickness: the thickness of the line
      %mean: mean parameter of the alt distribution
      %std: std parameter of the alt distribution
    %RETURN:
      %image: the defected image
      %isAltImage: boolean image, true for defect
    function [image, isAltImage] = addLineDefect(this, image, isAltImage, x, thickness, mean, std)
      %get the column index which the defect is to be added
      defectColumn = this.getRange(image, x, thickness);
      defectColumn = this.checkColumnBoundary(image, defectColumn);
      %set the line to be samples from the alt distribution
      image(:,defectColumn) = this.randStream.randn(numel(image(:,1)), 1) * std + mean;
      %set the line in this.altImage to true
      isAltImage(:, defectColumn) = true;
    end
    
    %METHOD: ADD DUST
    %Random select pixel with probability p, these selected pixels are alt
    %PARAMETERS:
      %image: the image to be defected
      %isAltImage: boolean image, true for defect
      %p: probability pixel is alt
      %mean: mean parameter of the alt distribution
      %std: std parameter of the alt distribution
    %RETURN:
      %image: the defected image
      %isAltImage: boolean image, true for defect
    function [image, isAltImage] = addDust(this, image, isAltImage, p, mean, std)
      %for all pixels
      for iPixel = 1:numel(image)
        %if this pixel is to be alt, assign alt sample
        if(this.randStream.rand < p)
          image(iPixel) = this.randStream.randn() * std + mean;
          isAltImage(iPixel) = true;
        end
      end
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
        %example: XXOX where I is the centre
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

