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
            this@MeanVar_GLM(shape_parameter,polynomial_order,[-1;0]);
        end
        
        %GET LINK FUNCTION DIFFERENTATED
        %PARAMETERS:
            %mu: column vector of means
        %RETURN:
            %g_dash: colum vector of g'(mu)
        function g_dash = getLinkDiff(this,mu)
            g_dash = ones(numel(mu),1);
        end
        
        %GET MEAN (LINK FUNCTION)
        function mu = getMean(this,eta)
            mu = eta;
        end
    end
    
end

