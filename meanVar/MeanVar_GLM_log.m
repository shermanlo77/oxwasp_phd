classdef MeanVar_GLM_log < MeanVar_GLM
    %MEANVAR_GLM_IDENTITY
    %identity link, linear feature
    
    properties
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %PARAMETERS: see upser class MeanVar_GLM
        function this = MeanVar_GLM_log(shape_parameter,polynomial_order)
            %call superclass
            this@MeanVar_GLM(shape_parameter,polynomial_order,LinkFunction_Log());
        end
        
    end
    
end

