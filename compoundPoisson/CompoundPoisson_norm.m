%COMPOUND POISSON NORMAL APPROXIMATION
%SEE SUPER CLASS COMPOUND POISSON
    %pdf functions overriden to use normal approximation
classdef CompoundPoisson_norm < CompoundPoisson
    
    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = CompoundPoisson_norm()
            %call superclass
            this@CompoundPoisson();
            this.can_support_zero_mass = false;
            this.name = 'cp_norm';
        end
        
        %OVERRIDE: GET DENSITY
        %PARAMETER:
            %x: vector of compound Poisson variables
        %RETURN:
            %pdf: vector of densities, for each element in x
        function pdf = getPdf(this,x)
            pdf = normpdf(x,this.mu,this.sigma);
        end
        
        %OVERRIDE: GET LOG DENSITY
        %PARAMETER:
            %x: scalar, compound poisson variable
        %RETURN:
            %ln_pdf: log density
        function ln_pdf = getlnpdf(this,x)
            ln_pdf = log(this.getPdf(x));
        end
        
        %OVERRIDE: GET INVERSE CDF
        function x = getInvCdf(this,p_array, ~,~,~)
            x = norminv(p_array,this.mu,this.sigma);
        end
        
    end
    
end

