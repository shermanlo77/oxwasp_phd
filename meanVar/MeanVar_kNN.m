classdef MeanVar_kNN < VarianceModelling
    %MEANVAR_KNN Regress the mean and variance relationship using kNN
    
    %MEMBER VARIABLES
    properties
        training_mean; %array of means in the training set
        training_var; %array of variances in the test set
    end
    
    methods
        
        %CONSTRUCTOR
        %PARAMETERS:
            %k: number of neighbours to look for
        function this = MeanVar_kNN(k)
            %assign member variable
            this.parameter = k;
        end
        
        %TRAIN CLASSIFIER
        %PARAMETERS:
            %training_mean: vector of means
            %training_var: vector of variances
        function train(this,training_mean,training_var)
            %assign member variables
            this.training_mean = training_mean;
            this.training_var = training_var;
            this.n_train = numel(training_mean);
        end
        
        %PREDICT VARIANCE
        %RETURN:
            %variance_prediction: predicted greyvalue variance (column vector)
            %up_error: 84% percentile
            %down_error: 16% percentile
        function [variance_prediction, up_error, down_error] = predict(this,sample_mean)
            %error bars not available for kNN
            up_error = nan;
            down_error = nan;
            
            %get the number of predictions to be made
            n = numel(sample_mean);
            
            %kNN requires a k*n matrix, check the size of it
            %if this matrix is of sensible size
            if n*this.parameter <= 1E8
                %do k nearest neighbour
                %get the index of the k nearest neighbours
                nn_index = (knnsearch(this.training_mean,sample_mean,'K',this.parameter))';
                %take the mean over the k nearest neighbours
                variance_prediction = (mean(this.training_var(nn_index),1))';
            %else the matrix is too big
            else
                %declare a vector for storing the predictions
                variance_prediction = zeros(n,1);
                %n_request is the number of prediction which can be done right now
                n_request = round(1E8/this.parameter);
                %do kNN for n_request data (recursive)
                variance_prediction(1:n_request) = this.predict(sample_mean(1:n_request));
                %do kNN for the rest of the data (recursive)
                variance_prediction((n_request+1):end) = this.predict(sample_mean((n_request+1):end));
            end
        end
        
    end
    
end
