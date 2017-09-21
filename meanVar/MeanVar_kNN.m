classdef MeanVar_kNN < VarianceModel
    %MEANVAR_KNN Regress the mean and variance relationship using kNN
    
    %MEMBER VARIABLES
    properties
        %the minimum greyvalue in the training set
        mean_lookup_start;
        %maximum greyvalue in training set = mean_lookup_start + n_lookup
        n_lookup;
        %array of n_lookup+1 knn variances, one for each greyvalue in the range mean_lookup_start up to mean_lookup_start+n_lookup 
        variance_lookup;
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
            this.mean_lookup_start = floor(min(training_mean)); %minimum greyvalue
            this.n_lookup = ceil(max(training_mean)) - this.mean_lookup_start; %range of greyvalue
            %variance_lookup is a vector of knn variance prediction for the range of greyvalues
            this.variance_lookup = this.getKNNPrediction(training_mean, training_var, this.mean_lookup_start + (0:this.n_lookup)');
        end
        
        %PREDICT VARIANCE
        %PARAMETERS:
            %sample_mean: column vector of mean greyvalues
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
            
            %declare array for variance prediction
            variance_prediction = zeros(n,1);
            
            %for each prediction (or for each given mean)
            for i_n = 1:n
                %get the lookup index
                lookup = round(sample_mean(i_n)) - this.mean_lookup_start;
                %if the lookup index is bigger than the allocated range, set it to the limit
                if lookup < 0
                    lookup = 0;
                elseif lookup > this.n_lookup
                    lookup = this.n_lookup;
                end
                %save the lookup index to the array variance_prediction
                variance_prediction(i_n) = lookup;
            end
            %look up the variance prediction using the worked out lookup index
            variance_prediction = this.variance_lookup(variance_prediction+1);
        end
        
        function test_response = getKNNPrediction(this, training_feature, training_response, test_feature)
            %get the number of predictions to be made
            n = numel(test_feature);
            
            %kNN requires a k*n matrix, check the size of it
            %if this matrix is of sensible size
            if n*this.parameter <= 1E8
                %do k nearest neighbour
                %get the index of the k nearest neighbours
                nn_index = (knnsearch(training_feature,test_feature,'K',this.parameter))';
                %take the mean over the k nearest neighbours
                test_response = (mean(training_response(nn_index),1))';
            %else the matrix is too big
            else
                %declare a vector for storing the predictions
                test_response = zeros(n,1);
                %n_request is the number of prediction which can be done right now
                n_request = round(1E8/this.parameter);
                %do kNN for n_request data (recursive)
                test_response(1:n_request) = this.getKNNPrediction(training_feature, training_response, test_feature(1:n_request));
                %do kNN for the rest of the data (recursive)
                test_response((n_request+1):end) = this.getKNNPrediction(training_feature, training_response, test_feature((n_request+1):end));
            end
        end
        
        %GET NAME
        %Return name for this glm
        function name = getName(this)
            name = cell2mat({num2str(this.parameter),'-NN'});
        end
        
    end
    
end
