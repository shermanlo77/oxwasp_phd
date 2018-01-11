classdef DefectSimulator < handle

    properties
        height;
        width;
        image;
        sig_image;
    end
    
    methods
        
        function this = DefectSimulator(image)
            this.image = image;
            [this.height, this.width] = size(this.image);
            this.sig_image = zeros(this.height, this.width);
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
            
            
            this.image(top_left(1):bottom_right(1),top_left(2):bottom_right(2)) = this.image(top_left(1):bottom_right(1),top_left(2):bottom_right(2)) + intensity;
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
        
        function addPlane(this, grad, intercept)
            [x_grid, y_grid] = meshgrid(1:this.width, 1:this.height);
            plane = grad(2) * (x_grid - this.width/2) + grad(1) * (y_grid - this.height/2) + intercept;
            this.image = this.image + plane;
        end
        
    end
    
end

