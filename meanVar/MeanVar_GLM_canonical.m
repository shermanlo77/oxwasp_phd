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
            this@MeanVar_GLM(shape_parameter,polynomial_order,[-1;0]);
        end
        
        %GET LINK FUNCTION DIFFERENTATED
        %PARAMETERS:
            %mu: column vector of means
        %RETURN:
            %g_dash: colum vector of g'(mu)
        function g_dash = getLinkDiff(this,mu)
            g_dash = this.shape_parameter./(mu.^2);
        end
        
        %GET MEAN (LINK FUNCTION)
        function mu = getMean(this,eta)
            mu = -this.shape_parameter./eta;
        end
        
        %GET NAME
        %Return name for this glm
        function name = getName(this)
            name = cell2mat({'inverse, order ',num2str(this.polynomial_order)});
        end
    end
    
end

