classdef KernelRegression_Lookup < KernelRegression
    
    properties
        x_range;
        y_lookup;
        regressor;
    end
    
    methods
        
        %CONSTRUCTOR
        function this = KernelRegression_Lookup(kernel, scale_parameter)
            this@KernelRegression(kernel, scale_parameter);
            this.regressor = KernelRegression(kernel, scale_parameter);
        end
        
        function train(this, x, y)
            this.regressor.train(x,y);
            this.x_range = floor(min(x)) : ceil(max(x));
            this.y_lookup = zeros(size(this.x_range));
            for i_x = 1:numel(this.x_range)
                this.y_lookup(i_x) = this.regressor.predict(this.x_range(i_x));
            end
        end
        
        function y = predict(this, x)
            y = zeros(size(x));
            for i_x = 1:numel(x)
                x_int = round(x(i_x));
                if x_int < this.x_range(1)
                    x_int = this.x_range(1);
                elseif x_int > this.x_range(end)
                    x_int = this.x_range(end);
                end
                y(i_x) = this.y_lookup(x_int - this.x_range(1) + 1);
            end
        end
        
    end
    
end

