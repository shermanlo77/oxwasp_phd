classdef MeanVar_GLM_canonical < MeanVar_GLM
    %MEANVAR_GLM_CANONICAL
    %canonical link, linear feature
    
    properties
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        function this = MeanVar_GLM_canonical(shape_parameter)
            %call superclass
            this@MeanVar_GLM(shape_parameter);
        end
        
        %GET DESIGN MATRIX
        %PARAMETERS:
            %grey_values: column vector of greyvalues
        %RETURN:
            %X: n x 2 design matrix
        function X = getDesignMatrix(this,grey_values)
            X = [ones(numel(grey_values),1),1./grey_values];
        end
        
        %GET LINK FUNCTION DIFFERENTATED
        %PARAMETERS:
            %mu: column vector of means
        %RETURN:
            %g_dash: colum vector of g'(mu)
        function g_dash = getLinkDiff(this,mu)
            g_dash = this.shape_parameter./(mu.^2);
        end
        
        %GET NATURAL PARAMETER from systematic component
        function theta = getNaturalParameter(this,eta)
            theta = eta;
        end
    end
    
end

