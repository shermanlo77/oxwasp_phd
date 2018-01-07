classdef Kernel < handle
    
    properties
    end
    
    methods (Abstract)
        k = evaluate(this, x);
    end
    
end

