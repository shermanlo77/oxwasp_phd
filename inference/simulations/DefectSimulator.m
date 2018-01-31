classdef DefectSimulator < handle

    properties
        height;
        width;
        defect_image;
        sig_image;
    end
    
    methods
        
        function this = DefectSimulator(size)
            this.height = size(1);
            this.width = size(2);
            this.defect_image = zeros(size);
            this.sig_image = zeros(size);
        end
        
        function addSquareDefect(this, co_od, defect_size, intensity)
            top_left = co_od - floor(defect_size/2);
            bottom_right = co_od +ceil(defect_size/2);
            
            top_left(top_left < 1) = 1;
            if bottom_right(1) > this.height
                bottom_right(1) = this.height;
            end
            if bottom_right(2) > this.width
                bottom_right(2) = this.width;
            end
            this.defect_image(top_left(1):bottom_right(1),top_left(2):bottom_right(2)) = this.defect_image(top_left(1):bottom_right(1),top_left(2):bottom_right(2)) + intensity;
            this.sig_image(top_left(1):bottom_right(1),top_left(2):bottom_right(2)) = true;
        end
        
        function addSquareDefectGrid(this, n_defect, defect_size, intensity)
            y_cood = linspace(0,this.height,n_defect(1));
            x_cood = linspace(0,this.width,n_defect(2));
            for i_x = 1:n_defect(2)
                for i_y = 1:n_defect(1)
                    this.addSquareDefect(round([y_cood(i_y);x_cood(i_x)]),defect_size,intensity);
                end
            end
        end
        
        function addPlane(this, grad)
            [x_grid, y_grid] = meshgrid(1:this.width, 1:this.height);
            plane = grad(2) * (x_grid - this.width/2) + grad(1) * (y_grid - this.height/2);
            this.defect_image = this.defect_image + plane;
        end
        
        function addSinusoid(this, amplitude, wavelength, angular_offset)
            [x_grid, y_grid] = meshgrid(1:this.width, 1:this.height);
            x_grid = x_grid - this.width/2;
            y_grid = y_grid - this.height/2;
            
            f = 1./wavelength;
            
            sinusoid = amplitude * sin( 2*pi*(f(1)*y_grid + f(2)*x_grid) + angular_offset);
            this.defect_image = this.defect_image + sinusoid;
        end
        
        function image = defectImage(this, image)
            image = image + this.defect_image;
        end
        
    end
    
end

