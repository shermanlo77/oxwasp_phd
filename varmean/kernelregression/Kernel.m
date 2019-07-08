classdef Kernel < handle
    
    %MEMBER VARIABLES
    properties (SetAccess = private)
        name; %name of the kernel
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        function this = Kernel(name)
            this.name = name;
        end
        
    end
    
    %ABSTRACT METHODS
    methods (Abstract)
        k = evaluate(this, x);
    end
    
end

