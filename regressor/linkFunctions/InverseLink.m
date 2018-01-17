classdef InverseLink < Link

    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = InverseLink()
            this@Link('inverse',-1);
        end
        
        %GET LINK FUNCTION DIFFERENTATED
        %PARAMETERS:
            %mu: column vector of means
        %RETURN:
            %g_dash: colum vector of g'(mu)
        function g_dash = getLinkDiff(this,mu)
            g_dash = 100./(mu.^2);
        end
        
        %GET MEAN (LINK FUNCTION)
        function mu = getMean(this,eta)
            mu = -100./eta;
        end
        
    end
    
end
