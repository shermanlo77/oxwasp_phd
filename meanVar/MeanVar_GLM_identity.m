classdef MeanVar_GLM_identity < MeanVar_GLM
    %MEANVAR_GLM_IDENTITY
    %identity link, linear feature
    
    properties
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %PARAMETERS: see upser class MeanVar_GLM
        function this = MeanVar_GLM_identity(shape_parameter,polynomial_order)
            %call superclass
            this@MeanVar_GLM(shape_parameter,polynomial_order,LinkFunction_Identity());
        end
        
    end
    
end

