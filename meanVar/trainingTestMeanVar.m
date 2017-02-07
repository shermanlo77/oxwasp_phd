function [mse_training_array, mse_test_array] = trainingTestMeanVar(data, variance_model_array, n_train, n_repeat)
    %TRAINING/TEST MEAN VAR Gets the training and test MSE when fitting and predicting the mean and variance relationship
    %PARAMETERS:
        %data: data object
        %variance_model_array: cell array of variance model objects, each with a different parameter
        %n_train: number of images to be used in estimating the mean and variance of the training set
        %n_repeat: number of times to repeat the experiment, this is done by shuffling the training/test set
    %RETURN:
        %mse_training_array: n_repeat x n_parameter matrix of the training mse
        %mse_test_array: n_repeat x n_parameter matrix of the test mse

    %get the pixels which do not belong to the 3d printed sample
    threshold = reshape(data.getThreshold_topHalf(),[],1); %reshape it to a vector

    %get the number of variance models in variance_model_array
    n_parameter = numel(variance_model_array);
    
    %array to store the mse for each polynomial and each spilt
        %dim1: for each repeat
        %dim2: for each parameter value
    mse_training_array = zeros(n_repeat,n_parameter); %training mse
    mse_test_array = zeros(n_repeat,n_parameter); %test mse

    %for each parameter
    for i_parameter = 1:n_parameter
        
        %display the progress
        disp(strcat('i_parameter=',num2str(i_parameter)));
        
        %get the variance model with the i_parameter
        model = variance_model_array{i_parameter};

        %for n_repeat times
        for i_repeat = 1:n_repeat
            
            %display the iteration number
            disp(strcat('iteration',num2str(i_repeat)));

            %get random index of the training and test data
            index_suffle = randperm(data.n_sample);
            training_index = index_suffle(1:n_train);
            test_index = index_suffle((n_train+1):data.n_sample);

            %get variance mean data of the training set
            [sample_mean,sample_var] = data.getSampleMeanVar_topHalf(training_index);
            %segment the mean var data
            sample_mean(threshold) = [];
            sample_var(threshold) = [];

            %train the classifier
            model.train(sample_mean,sample_var);
            %get the training mse
            mse_training_array(i_repeat,i_parameter) = model.getPredictionMSE(sample_mean,sample_var);

            %get the variance mean data of the test set
            [sample_mean,sample_var] = data.getSampleMeanVar_topHalf(test_index);
            %segment the mean var data
            sample_mean(threshold) = [];
            sample_var(threshold) = [];
            %get the test mse
            mse_test_array(i_repeat,i_parameter) = model.getPredictionMSE(sample_mean,sample_var);

        end

    end

end
