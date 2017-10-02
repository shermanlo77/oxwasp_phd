%EXPERIMENT FOR VARIANCE AND BIAS
%Taking expectations over folds to work out RSS, MSE, VAR, BIAS
%
%The images are spilt half training, half test, mean and variance were estimated for each of these sets
%The training set is used to train the variance-mean model
%The model is then used to predict the variance given the mean for the test set
%The mean prediction and the test variance are updated for every fold
%RSS are updated for every fold
%Variance of the variance prediciton are updated for every fold
%
%Plots the statistics RSS, MSE, VAR, BIAS, BIAS^2, sigma^2 for every pixel as heatmap, boxplot
classdef Experiment_ImageVarBias2 < Experiment_GLMVarMean
    
    %MEMBER VARIABLES
    properties
        
        %variables with 2 dimensions, storing a statistics for every pixel
        y_array; %test variances
        y_hat_array; %test variance predictions
        rss_array; %residual squares
        var_array; %variance of the variance prediction
        sigma_array; %variance of the variance
        
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %PARAMETER
            %experiment_name: name of the experiment (passed onto superclass)
        function this = Experiment_ImageVarBias2(experiment_name)
            %call superclass
            this@Experiment_GLMVarMean(experiment_name);
        end
        
        %OVERRIDE: setUpExperiment
        %PARAMETERS:
            %n_repeat: number of folds
            %rand_stream: random stream
        function setUpExperiment(this,n_repeat, rand_stream)
            %pass parameters to superclass
            this.setUpExperiment@Experiment_GLMVarMean(n_repeat, rand_stream);
        end
        
        %OVERRIDE ASSIGN ARRAY
        %assign the member variables
        function assignArray(this)
            n_pixel = sum(this.mean_variance_estimator.segmentation);
            this.y_array = zeros(n_pixel, this.getNGlm(), this.getNShadingCorrector());
            this.y_hat_array = zeros(n_pixel, this.getNGlm(), this.getNShadingCorrector());
            this.rss_array = zeros(n_pixel, this.getNGlm(), this.getNShadingCorrector());
            this.var_array = zeros(n_pixel, this.getNGlm(), this.getNShadingCorrector());
            this.sigma_array = zeros(n_pixel, this.getNGlm(), this.getNShadingCorrector());
        end
        
        %DO ONE ITERATION OF EXPERIMENT
        function doIteration(this)
            %get the model
            model = this.getGlm(this.i_glm);

            %get random index of the training and test data
            index_suffle = randperm(this.n_sample);
            training_index = index_suffle(1:this.n_train);
            test_index = index_suffle((this.n_train+1):end);

            %get variance mean data of the training set
            [sample_mean,sample_var] = this.getMeanVar(training_index);
            %train the classifier
            model.train(sample_mean,sample_var);
            
            %get the variance mean data of the test set
            [~,y] = this.getMeanVar(test_index);
            y_hat = model.predict(sample_mean);
            
            %update the means of y, y_hat and rss
            this.y_array(:,this.i_glm, this.i_shad) = this.meanUpdate(this.y_array(:,this.i_glm, this.i_shad), y);
            this.y_hat_array(:,this.i_glm, this.i_shad) = this.meanUpdate(this.y_hat_array(:,this.i_glm, this.i_shad), y_hat);
            this.rss_array(:,this.i_glm, this.i_shad) = this.meanUpdate(this.rss_array(:,this.i_glm, this.i_shad), (y_hat - y).^2);
            
            %update the variances of y_hat and y
            this.var_array(:,this.i_glm, this.i_shad) = this.varianceUpdate(this.var_array(:,this.i_glm, this.i_shad), this.y_hat_array(:,this.i_glm, this.i_shad), y_hat);
            this.sigma_array(:,this.i_glm, this.i_shad) = this.varianceUpdate(this.sigma_array(:,this.i_glm, this.i_shad), this.y_array(:,this.i_glm, this.i_shad), y);
            
        end
        
        %PRINT RESULTS
        %Box plot the training and test MSSE
        function printResults(this)
            this.plotBoxPlot(this.rss_array,'RSS');
            this.plotBoxPlot(this.var_array,'Variance');
            this.plotBoxPlot(this.y_hat_array - this.y_array,'bias');
            this.plotBoxPlot((this.y_hat_array - this.y_array).^2,'bias^2');
            this.plotBoxPlot(this.var_array + (this.y_hat_array - this.y_array).^2,'MSE');
        end
        
        %MEAN UPDATE
        %Update the current mean given a new value
        %PARAMETERS:
            %old_statistic: current mean, can be 2 dimensional
            %new_value: new value, same dimension as old_statistic
        function new_statistic = meanUpdate(this, old_statistic, new_value)
            %if this is the first iteration
            if this.i_repeat == 1
                %the mean is the value
                new_statistic = new_value;
            %else more than 1 iteration has been done
            else
                %update the mean
                new_statistic = ((this.i_repeat - 1)/this.i_repeat)*old_statistic + new_value/this.i_repeat;
            end
        end
        
        %VARIANCE UPDATE
        %Update the current variance given a new value and updated mean
        %PARAMETERS:
            %old_statistic: current mean, can be 2 dimensional
            %new_value: new value, same dimension as old_statistic
        function new_statistic = varianceUpdate(this, old_statistic, new_mean, new_value)
            %if this is the 1st iteration, let new_statistic store the 1st value
            %the 1st value will then be passed onto the next iteration when the variance will be first calculated
            if this.i_repeat == 1
                new_statistic = new_value;
            %else if this is the 2nd iteration
            %work out the variance for the first time, old_statistic is the 1st value
            elseif this.i_repeat == 2
                new_statistic = ((old_statistic - new_mean).^2 + (new_value - new_mean).^2)/2;
            %else the variance needs updating, update it
            else
                new_statistic = ((this.i_repeat - 1)/this.i_repeat)*old_statistic + ((new_value - new_mean).^2)/(this.i_repeat - 1);
            end
        end

    end
    
    methods (Abstract)
        
        %returns scan object
        scan = getScan(this);
        
        %returns number of glm to investigate
        n_glm = getNGlm(this);
        
        %returns glm model given index
        %index can range from 1 to getNGlm()
        model = getGlm(this, index);
        
        %returns number of shading correctors to investigate
        n_shad = getNShadingCorrector(this);
        
        %returns shading corrector given index
        %index can range from 1 to getNShadingCorrector
        %RETURNS:
            %shading_corrector: ShadingCorrector object
            %reference_index: row vector containing integers
                %pointing to which reference scans to be used for shading correction training
        [shading_corrector, reference_index] = getShadingCorrector(this, index);
        
    end

end
