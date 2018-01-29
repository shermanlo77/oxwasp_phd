classdef LocalQuadraticRegression < LocalLinearRegression
    
    properties
    end
    
    methods
        
        function this = LocalQuadraticRegression(kernel, lambda)
            this@LocalLinearRegression(kernel, lambda);
        end
        
        
        function regression = getLocalRegression(this)
            regression = QuadraticRegression();
        end
    end
    
end

