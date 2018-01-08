classdef Regressor < handle
    %VARIANCEMODELLING Abstract superclass for modelling the variance
    %
    %Classifier for predicting variance given some feature (e.g. mean).
    %The classifier is trained using the method train and then can predict using the method
    %predict. To assess performance, the mean squared prediction error can
    %be obtained in the method prediction_mse.
    
    %MEMBER VARIABLES
    properties (SetAccess = protected)
        parameter; %parameter to estimate to fit onto the data
        n_train; %size of the training set
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %Do nothing
        function this = Regressor()
        end
        
        %PREDICTION MEAN SQUARED ERROR
        %Return the mean squared prediction error
        %PARAMETERS:
            %y: greyvalue variance (column vector)
            %x: greyvalue mean (column vector)
            %x and y are the same size
        %RETURN:
            %mse: scalar mean squared error
        function mse = getPredictionMSE(this,x,y)
            %given greyvalue mean, predict greyvalue variance
            y_predict = this.predict(x);
            %work out the mean squared error
            mse = sum((y-y_predict).^2)/numel(y);
        end
        
        function [mse_vector] = getPredictionMSSE(this, x, y)
            mse_vector = [nan; this.getPredictionMSE(x,y)];
        end

    end
    
    %ABSTRACT METHODS
    methods (Abstract, Access = public)
        
        %TRAIN CLASSIFIER
        train(this,sample_mean,sample_var);
        
        %PREDICT VARIANCE
        %RETURN:
            %variance_prediction: predicted greyvalue variance (column vector)
            %up_error: 84% percentile
            %down_error: 16% percentile
        [variance_prediction, up_error, down_error] = predict(this,sample_mean);

        has_errorbar = hasErrorbar(this)
        
    end
    
end

