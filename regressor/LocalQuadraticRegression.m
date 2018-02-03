%CLASS: LOCAL QUADRATIC REGRESSION
%A class used for local quadratic regression
%
%See superclass LocalLinearRegression
classdef LocalQuadraticRegression < LocalLinearRegression
    
    properties
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %kernel: kernel object
            %lambda: parameter for the kernel
        function this = LocalQuadraticRegression(kernel, lambda)
            %call superclass
            this@LocalLinearRegression(kernel, lambda);
        end
        
    end
    
    %PROTECTED METHOD
    methods (Access = protected)        
        
        %OVERRIDE: GET LOCAL REGRESSION
        %Returns a regression object for the local regression
        %RETURN:
            %regression: regression object for local regression
        function regression = getLocalRegression(this)
            regression = QuadraticRegression();
        end
    end
    
end

