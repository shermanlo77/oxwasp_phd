%NORMALITY TESTER CHI SQUARED
%Subclass of NormalityTester
    %implements the chi squared goodness of fit Normality test
classdef NormalityTester_chiSquared < NormalityTester
    
    %MEMBER VARIABLES
    properties
        %expected frequencies in each bin
        n_in_bin
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %PARAMETERS:
            %n_in_bin: expected frequencies in each bin
        function this = NormalityTester_chiSquared(n_in_bin)
            %assign member variables
            this.n_in_bin = n_in_bin;
        end %constructor
        
        %METHOD: GET P VALUE
        %Given a vector of grey values, conduct a normality hypothesis test
        %PARAMETERS:
            %grey_value_array: vector of grey values
        %RETURN:
            %p: a single p value
        function p = getPValue(this, x)
            %conduct chi squared goodness of fit test
            %parameters:
                %x: data
                %this.n_in_bin: expected frequencies in each bin
                %true: the data is not normalised
            p = chi2gof_norm(x, this.n_in_bin, true);
        end %getPValue
        
    end %methods
    
end %class

