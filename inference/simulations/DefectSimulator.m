%CLASS: DEFECT SIMULATOR
%Class for adding smooth function and defects to scan images
%
%HOW TO USE:
%Use the contructor to define the size of the image
%Use methods to define the smooth function and/or add defects
%Use the method defectImage(image) to add the smooth function and defects to the image
classdef DefectSimulator < handle

    %MEMBER VARIABLES
    properties (SetAccess = private)
        height; %height of the image
        width; %width of the image
        defect_image; %image of combination of smooth function and defects
        sig_image; %boolean image, true for defect, false for no defect
        n_sig; %number of significant pixels (calculated after the method setMask() is called)
        n_null; %number of null pixels (calculated after the method setMask() is called)
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %size: ROW vector defining the size of the image
        function this = DefectSimulator(size)
            %assign member variables
            this.height = size(1);
            this.width = size(2);
            this.defect_image = zeros(size);
            this.sig_image = zeros(size);
        end
        
        %METHOD: ADD SQUARE DEFECT
        %Add a square (defect) to the image
        %PARAMETERS:
            %co_od: coordinate of the middle of the square
            %defect_size: column vector defining the size of the square defect
            %intensity: greyvalue of the square
        function addSquareDefect(this, co_od, defect_size, intensity)
            %get the range of columns and rows to fill with a defect
            row_index = this.getRange(co_od(1), defect_size(1));
            column_index = this.getRange(co_od(2), defect_size(2));
            [row_index, column_index] = this.checkBoundary(row_index, column_index);
            
            %add the square to this.defect_image
            this.defect_image(row_index, column_index) = this.defect_image(row_index, column_index) + intensity;
            %set the square in this.sig_image to be true
            this.sig_image(row_index, column_index) = true;
        end
        
        %METHOD: ADD SQUARE DEFECT GRID
        %Add a grid of squares
        %PARAMETERS:
            %n_defect: column vector setting the number of [rows;columns] of defects
            %defect_size: column vector defining the size of the square defect
            %intensity: greyvalue of the square
        function addSquareDefectGrid(this, n_defect, defect_size, intensity)
            %get the x and y coordinates of the center of squares
            y_cood = linspace(0,this.height,n_defect(1));
            x_cood = linspace(0,this.width,n_defect(2));
            %for each column
            for i_x = 1:n_defect(2)
                %for each row
                for i_y = 1:n_defect(1)
                    %add square defect
                    this.addSquareDefect(round([y_cood(i_y);x_cood(i_x)]),defect_size,intensity);
                end
            end
        end
        
        %METHOD: ADD LINE DEFECT
        %Add a vertical line
        %PARAMETERS:
            %x_cood: x coordinate of the center of the line
            %thickness: the thickness of the line
            %intensity: greyvalue of the line
        function addLineDefect(this, x_cood, thickness, intensity)
            %get the column index which the defect is to be added
            defect_column = this.getRange(x_cood, thickness);
            defect_column = this.checkColumnBoundary(defect_column);
            %add the line to this.defect_image
            this.defect_image(:,defect_column) = this.defect_image(:,defect_column) + intensity;
            %set the line in this.sig_image to true
            this.sig_image(:, defect_column) = true;
        end
        
        %METHOD: GET RANGE
        %Get the list of index given the coordinates of the center and the length
        %PARAMETERS:
            %centre_cood: centre of the range
            %length: length of the range
        %RETURN:
            %range: length number of integers, where centre_cood is in the middle
        function range = getRange(this, centre_cood, length)
            %if the length is odd
            if mod(length,2)
                %get the range of integers
                %example: XXOXX where O is the centre
                range = (centre_cood - (length-1)/2) : (centre_cood + (length-1)/2);
            %else the length is even
            else
                %get the range of integers, including the middle and cutting the right hand side
                %example: XXOX where I is the centre
                range = (centre_cood - length/2) : (centre_cood + length/2 - 1);
            end
        end
        
        %METHOD: CHECK BOUNDARY
        %Given indices for the rows and columns, remove the ones which are outside the boundary
        %PARAMETERS:
            %row_index: indices of rows
            %column_index: indicies of columns
        %RETURN:
            %row_index: row_index with boundary check
            %column_index: column_index with bounday check
        function [row_index, column_index] = checkBoundary(this, row_index, column_index)
            %check the boundary of the rows and columns and return it
            row_index = this.checkRowBoundary(row_index);
            column_index = this.checkColumnBoundary(column_index);
        end
        
        %METHOD: CHECK ROW BOUNDARY
        %Given indices for the rows, remove the ones which are outside the boundary
        %PARAMETERS:
            %row_index: indices of rows
        %RETURN:
            %row_index: row_index with boundary check
        function row_index = checkRowBoundary(this, row_index)
            %remove the rows where it is equal and below 0 and bigger than the height
            index_remove = (row_index <= 0) | (row_index > this.height);
            row_index(index_remove) = [];
        end
        
        %METHOD: CHECK COLUMN BOUNDARY
        %Given indices for the columns, remove the ones which are outside the boundary
        %PARAMETERS:
            %column_index: indicies of columns
        %RETURN:
            %column_index: column_index with bounday check
        function column_index = checkColumnBoundary(this, column_index)
            %remove the columns where it is equal and below 0 and bigger than the width
            index_remove = (column_index <= 0) | (column_index > this.width);
            column_index(index_remove) = [];
        end
        
        %METHOD: ADD PLANE
        %Add a gradient, value of 0 in the middle
        %PARAMETERS:
            %grad: the gradient of the plane
        function addPlane(this, grad)
            %mesh grid of the image
            [x_grid, y_grid] = meshgrid(1:this.width, 1:this.height);
            %calculate the value of the plane for each x and y
            plane = grad(2) * (x_grid - this.width/2) + grad(1) * (y_grid - this.height/2);
            %add the plane to this.defect_image
            this.defect_image = this.defect_image + plane;
        end
        
        %METHOD: ADD SINUSOID
        %Add a sinusoid
        %PARAMETERS:
            %amplitude: amplitude of the sinusoid
            %wavelength: 2 column vector defining the wavelength of [y,x] direction, can be negative
            %angular_offset: offset the sinusoid in radians, for angular_offset = 0 the middle of the image = 0
        function addSinusoid(this, amplitude, wavelength, angular_offset)
            %meshgrid of the the image
            [x_grid, y_grid] = meshgrid(1:this.width, 1:this.height);
            %shift the grid so that the middle is the origin
            x_grid = x_grid - this.width/2;
            y_grid = y_grid - this.height/2;
            %convert the wavelength to a frequency
            f = 1./wavelength;
            %work out the value of the sinusoid for each x and y
            sinusoid = amplitude * sin( 2*pi*(f(1)*y_grid + f(2)*x_grid) + angular_offset);
            %add the sinusoid to the image
            this.defect_image = this.defect_image + sinusoid;
        end
        
        %METHOD: DEFECT IMAGE
        %Add this.defect_image to the image
        %PARAMETER:
            %image: image for the defect simulator to add to
        %RETURN:
            %image: image + this.defect_image
        function image = defectImage(this, image)
            image = image + this.defect_image;
        end
        
        %METHOD: SET MASK
        %Set the mask of the sig_image, pixels outside the mask are set to false
        %Also calculates the number of sig and non-sig pixels
        %PARMAETERS:
            %mask: boolean image, true if that pixel is a mask
        function setMask(this, mask)
            this.sig_image(~mask) = false;
            this.n_sig = sum(sum(this.sig_image));
            this.n_null = sum(sum( (~this.sig_image) & mask));
        end
        
    end
    
end

