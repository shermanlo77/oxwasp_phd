classdef MeanVar_GLM_canonical < MeanVar_GLM
    %MEANVAR_GLM_CANONICAL
    %canonical link, linear feature
    
    properties
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %PARAMETERS: see upser class MeanVar_GLM
        function this = MeanVar_GLM_canonical(shape_parameter,polynomial_order)
            %call superclass
            this@MeanVar_GLM(shape_parameter,polynomial_order,LinkFunction_Canonical());
        end
        
    end
    
end

