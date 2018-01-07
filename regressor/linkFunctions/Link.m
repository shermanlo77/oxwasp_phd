%ABSTRACT CLASS: LINK FUNCTION
%Implements a method for a given link function
%Implements:
    %getLinkDiff: get link function differentated given mean
    %getMean: get mean given eta
classdef Link < handle

    %MEMBER VARIABLES
    properties (SetAccess = private)
        name; %name of implemented class
        initial_intercept; %initial value of the intercept
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %name: name of the implemented subclass
            %initial_intercept: initial value of the intercept
        function this = LinkFunction(name, initial_intercept)
            %assign member variables
            this.name = name;
            this.initial_intercept = initial_intercept;
        end
        
    end
    
    methods (Abstract)
        
        %GET LINK FUNCTION DIFFERENTATED
        %PARAMETERS:
            %mu: column vector of means
        %RETURN:
            %g_dash: colum vector of g'(mu)
        g_dash = getLinkDiff(this,mu)
        
        %GET MEAN (LINK FUNCTION)
        mu = getMean(this,eta)
    end
    
end

