classdef TricubeKernel < handle
    
    properties
    end
    
    methods (Access = public)
        
        function this = TricubeKernel() 
        end
        
        function k = evaluateKernel(this, x)
            k = zeros(size(x));
            k(abs(x)>=1) = 0;
            k(abs(x)<1) = (1-abs(x(abs(x)<1)).^3).^3;
        end
    end
    
end

