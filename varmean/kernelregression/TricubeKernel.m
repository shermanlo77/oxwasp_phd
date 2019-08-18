%MIT License
%Copyright (c) 2019 Sherman Lo

classdef TricubeKernel < Kernel
    
    properties
    end
    
    methods (Access = public)
        
        function this = TricubeKernel()
            this@Kernel('Tricube');
        end
        
        function k = evaluate(this, x)
            k = zeros(size(x));
            k(abs(x)>=1) = 0;
            k(abs(x)<1) = (1-abs(x(abs(x)<1)).^3).^3;
        end
    end
    
end

