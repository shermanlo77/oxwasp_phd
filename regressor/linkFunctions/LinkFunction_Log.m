classdef LinkFunction_Log < LinkFunction

    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = LinkFunction_Log()
            this@LinkFunction('log',0);
        end
        
        %GET LINK FUNCTION DIFFERENTATED
        %PARAMETERS:
            %mu: column vector of means
        %RETURN:
            %g_dash: colum vector of g'(mu)
        function g_dash = getLinkDiff(this,mu)
            g_dash = 1./mu;
        end
        
        %GET MEAN (LINK FUNCTION)
        function mu = getMean(this,eta)
            mu = exp(eta);
        end
        
    end
    
end

