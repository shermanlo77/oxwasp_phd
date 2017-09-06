classdef LinkFunction_Canonical < LinkFunction

    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = LinkFunction_Canonical()
            this@LinkFunction('canonical',-1);
        end
        
        %GET LINK FUNCTION DIFFERENTATED
        %PARAMETERS:
            %mu: column vector of means
        %RETURN:
            %g_dash: colum vector of g'(mu)
        function g_dash = getLinkDiff(this,mu,shape_parameter)
            g_dash = shape_parameter./(mu.^2);
        end
        
        %GET MEAN (LINK FUNCTION)
        function mu = getMean(this,eta,shape_parameter)
            mu = -shape_parameter./eta;
        end
        
    end
    
end

