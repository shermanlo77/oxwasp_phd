classdef EpanechnikovKernel < Kernel
    
    properties
    end
    
    methods (Access = public)
        
        function this = EpanechnikovKernel()
        end
        
        function k = evaluate(this, x)
            k = zeros(size(x));
            k(abs(x)>=1) = 0;
            k(abs(x)<1) = 0.75 * (1-x(abs(x)<1).^2);
        end
        
    end
    
end

