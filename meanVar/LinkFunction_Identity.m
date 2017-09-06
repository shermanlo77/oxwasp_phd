classdef LinkFunction_Identity < LinkFunction

    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = LinkFunction_Identity()
            this@LinkFunction('identity',1);
        end
        
        %GET LINK FUNCTION DIFFERENTATED
        %PARAMETERS:
            %mu: column vector of means
        %RETURN:
            %g_dash: colum vector of g'(mu)
        function g_dash = getLinkDiff(this,mu,shape_parameter)
            g_dash = ones(numel(mu),1);
        end
        
        %GET MEAN (LINK FUNCTION)
        function mu = getMean(this,eta,shape_parameter)
            mu = eta;
        end
        
    end
    
end

