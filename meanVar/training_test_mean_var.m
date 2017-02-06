function [mse_training_array, mse_test_array] = training_test_mean_var(data, variance_model_handle, n_train, parameter_array, n_repeat)
    %TRAINING/TEST MEAN VAR Gets the training and test MSE when fitting and predicting the mean and variance relationship
    %PARAMETERS:
        %data: data object
        %variance_model_handle: function handle to instantise a new variance model
        %n_train: number of images to be used in estimating the mean and variance of the training set
        %parameter_array: [1,2] x n_parameter matrix of parameters
            %each row represent a different parameter (e.g. shape parameter and polynomial order)
            %each column for different values of the parameters to investigate
        %n_repeat: number of times to repeat the experiment, this is done by shuffling the training/test set
    %RETURN:
        %mse_training_array: n_repeat x n_parameter matrix of the training mse
        %mse_test_array: n_repeat x n_parameter matrix of the test mse

    %get the pixels which do not belong to the 3d printed sample
    threshold = reshape(data.getThreshold_topHalf(),[],1); %reshape it to a vector

    %get the size of parameter_array
    [parameter_dim, n_parameter] = size(parameter_array);

    %array to store the mse for each polynomial and each spilt
        %dim1: for each repeat
        %dim2: for each parameter value
    mse_training_array = zeros(n_repeat,n_parameter); %training mse
    mse_test_array = zeros(n_repeat,n_parameter); %test mse

    %for each parameter
    for i_parameter = 1:n_parameter
        
        %display the progress
        disp(strcat('i_parameter=',num2str(i_parameter)));
        
        %for each dimension (of number) of parameters, feed it into the constructor of the variance model
        if parameter_dim == 1
            model = feval(variance_model_handle,parameter_array(i_parameter));
        elseif parameter_dim == 2
            model = feval(variance_model_handle,parameter_array(1,i_parameter),parameter_array(2,i_parameter));
        end

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
            model.train(sample_mean,sample_var,100);
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
