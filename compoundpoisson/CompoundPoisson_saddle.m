%COMPOUND POISSON SADDLEPOINT APPROXIMATION
%SEE SUPER CLASS COMPOUND POISSON
    %overrides pdf methods to use saddlepoint approximation
classdef CompoundPoisson_saddle < CompoundPoisson
    
    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = CompoundPoisson_saddle()
            %call superclass
            this@CompoundPoisson();
            this.can_support_zero_mass = false;
            this.name = 'cp_saddle';
        end
        
        %OVERRIDE: GET DENSITY
        %For a range of the domain and given parameters, return the
        %saddlepoint density. The normalisation constant was worked out
        %using the trapezium rule, with n_point - 1 strips
        %PARAMETERS:
            %x: row vector, equally spaced out compound poisson random variables
        %RETURN:
            %f: row vector of size n_points containing the saddle density for each point
            %x: row vector of size n_points, linspace(x_min,x_max,n_point)
        function pdf = getPdf(this,x)
            %work out the log terms
            log_terms = sum([-(this.alpha+2)/(2*(this.alpha+1))*log(x);
                -x*this.beta;
                (x*this.beta/this.alpha).^(this.alpha/(this.alpha+1))*(this.lambda)^(1/(this.alpha+1))*(this.alpha+1)
                ]);
            
            %k is some constant to control over and under flow
            k = max(log_terms);
            %k = this.lambda;
            
            %work out the saddle density
            pdf = exp(log_terms - k);

            %work out the height of the trapziums
            h = (max(x)-min(x))/(numel(x)-1);
            %integrate the function
            area = 0.5*h*(pdf(1)+pdf(end)+2*sum(pdf(2:(end-1))));
            
            %normalise the saddle density
            pdf = pdf/area;
        end
        
        %OVERRIDE: GET LOG DENSITY
        %PARAMETER:
            %x: scalar, compound poisson variable
        %RETURN:
            %ln_pdf: log density
        function ln_pdf = getlnpdf(this,x)
            ln_pdf = log(this.getPdf(x));
        end
        
    end
    
end

