classdef GaussianKernel < handle

    properties
    end
    
    methods (Access = public)
        
        function this = GaussianKernel()
        end
        
        function k = evaluate(this, x)
            k = normpdf(x);
        end
        
    end
    
end

