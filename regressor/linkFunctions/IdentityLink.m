classdef IdentityLink < Link

    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = IdentityLink()
            this@Link('identity',1);
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

