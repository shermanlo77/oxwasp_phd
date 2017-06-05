%NORMALITY TESTER KOLMOGOROV SMIRNOV
%Subclass of NormalityTester
    %implements the Kolmogorov Smirnov Normality test
classdef NormalityTester_kolmogorovSmirnov < NormalityTester
    
    %MEMBER VARIABLES
    properties
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        function this = NormalityTester_kolmogorovSmirnov()
        end
        
        %METHOD: GET P VALUE
        %Given a vector of grey values, conduct a normality hypothesis test
        %PARAMETERS:
            %grey_value_array: vector of grey values
        %RETURN:
            %p: a single p value
        function p = getPValue(this, x)
            %try to do the ks test
            try
                %normalise the vector x
                x_std = std(x); %get the standard deviation
                %if the standard deviation is 1, don't rescale the data
                if x_std == 0
                    x_std = 1;
                end
                %do the ks test on the normalised data
                %it is normalised to have mean 0 and std 1
                [~,p] = kstest((x-mean(x))/x_std);
            %if the ks test fails, set the p value to be nan
            catch
                p = nan;
            end                
        end %getPValue
        
    end %methods
    
end %class

