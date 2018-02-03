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
            %get the coordinates of the 2 corners of the square
            top_left = co_od - floor(defect_size/2);
            bottom_right = co_od +ceil(defect_size/2);
            
            %correct boundaries for the coordinates
            top_left(top_left < 1) = 1;
            if bottom_right(1) > this.height
                bottom_right(1) = this.height;
            end
            if bottom_right(2) > this.width
                bottom_right(2) = this.width;
            end
            
            %add the square to this.defect_image
            this.defect_image(top_left(1):bottom_right(1),top_left(2):bottom_right(2)) = this.defect_image(top_left(1):bottom_right(1),top_left(2):bottom_right(2)) + intensity;
            %set the square in this.sig_image to be true
            this.sig_image(top_left(1):bottom_right(1),top_left(2):bottom_right(2)) = true;
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
        
    end
    
end

