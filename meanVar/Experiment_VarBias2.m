%EXPERIMENT VAR BIAS 2
%Experiment for estimating the rss, sigma^2, mse, variance, bias^2 and bias for a given greyvalue
%This is done by spilting randomly the images into the training set and test set of equal size
%The variance model is trained using the training set
%The trained model is used to predict the variances (y hat) for a range of grey values
%The variances for a range of greyvalues (y) are exracted from the test set
%This is repeated multiple times by reassigning the training set and test set to get a range of y and y_hat
%The statistics were calculated using the collections of y and y_hat for a range of grey values 
classdef Experiment_VarBias2 < Experiment
    
    %MEMBER VARIABLES
    properties
        
        %random stream object
        rand_stream;
        
        n_bootstrap; %number of bootstrap samples (1 boostrap is done by reassigning the training set and test set)
        shape_parameter; %shape_parameter of the gamma glm
        
        %row vector of greyvalue mean
        x_plot;
        %number of greyvalue means in x_plot
        n_plot;
        
        %matrix of test set variances
            %dim 1: for each x in x_plot
            %dim 2: for each fold
        y_array;
        
        %matrix of predicted variances
            %dim 1: for each x in x_plot
            %dim 2: for each fold
            %dim 3: for each model
        y_predict;
        
        %temporary variable
        %stores the grey values of each masked pixel, for each image
        mean_variance_estimator;
        
        %number of images to be used in training
        n_train;
        %total number of images in dataset
        n_image;

    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %PARAMETERS:
        	%experiment_name: name of the experiment
        function this = Experiment_VarBias2(experiment_name)
        	%call superclass, passing experiment_name
            this@Experiment(experiment_name);
        end
        
        %SET UP EXPERIMENT
        %PARAMETERS:
        	%n_boostrap: number of folds for the experiment (reassigment the training set and test set)
        function setUpExperiment(this, n_bootstrap, n_plot, rand_stream)
            %assign member variables
            this.n_bootstrap = n_bootstrap;
            this.n_plot = n_plot;
            this.rand_stream = rand_stream;
        end
        
        %DO EXPERIMENT
        %Do the experiment
        function doExperiment(this)
            %save the grey values and store it
            this.saveGreyvalueArray();
            %set the random stream
            RandStream.setGlobalStream(this.rand_stream);

            %for n_boostrap times
            for i_bootstrap = 1:this.n_bootstrap
            	%get random permutation
                index = randperm(this.n_image);
                %assign image index for the training set
                train_index = index(1:this.n_train);
                %assign image index for the test set
                test_index = index((this.n_train+1):end);
                
                %get the training mean and variance
                [sample_mean,sample_var] = getMeanVar(this, train_index);
                
                %for each model
                for i_model = 1:this.getNModel()
                   %get the model
                   model = this.getModel(i_model);
                   %train the model using the training mean and variance
                   model.train(sample_mean, sample_var);
                   %predict the variance given this.x_plot
                   this.y_predict(:,i_bootstrap,i_model) = model.predict(this.x_plot);                   
                end
                
                %get the test mean and variance
                [sample_mean,sample_var] = getMeanVar(this, test_index);
                %for each x in this.n_plot
                for i_x = 1:this.n_plot
                	%get the point of the variance where its mean is cloest to x
                    [~,var_index] = min(abs(sample_mean - this.x_plot(i_x)));
                    %save that variance in this,y_array
                    this.y_array(i_x,i_bootstrap) = sample_var(var_index);
                end
                
                %print progress bar
                this.printProgress(i_bootstrap/this.n_bootstrap);
                
            end
        end
        
        %DELETE GREYVALUE ARRAY
        function deleteVariables(this)
        	%delete the saved greyvalues
            this.mean_variance_estimator = [];
        end
        
        %PRINT RESULTS
        function printResults(this)
            
            %get array of glm names
            model_name_array = cell(this.getNModel(),1);
            for i = 1:this.getNModel()
                model = this.getModel(i);
                model_name_array{i} = model.getFileName();
            end
            
            %declare an array of mean y's
            %f has dimensions
            	%dim 1: for each x
            	%dim 2: for each bootstrap or fold
            f = repmat(mean(this.y_array,2),1,this.n_bootstrap);
            
            %work out the variance of y given each x
            noise_plot = var(this.y_array,[],2);

            %for each model
            for i_model = 1:this.getNModel()
                
                %work out the RSS for each x, mean of (y - y_hat)^2
                rss_plot = mean( (this.y_array - this.y_predict(:,:,i_model)).^2,2);
                %work out the MSE for each x, mean of (y_predict - f)^2
                mse_plot = mean( (this.y_predict(:,:,i_model) - f).^2,2);
                %work out the bias for each x, mean of y_predict - mean of y
                bias_plot = mean(this.y_predict(:,:,i_model),2) - mean(this.y_array,2);
                %work out the variance for each x, variance of y_predict
                var_plot = var(this.y_predict(:,:,i_model),[],2);
                
                %plot the rss and noise for each x
                fig = figure_latexSub();
                plot(this.x_plot,rss_plot);
                hold on;
                plot(this.x_plot,noise_plot);
                legend('RSS','\sigma^2','Location','northwest');
                xlabel('mean greyvalue (arb. unit)');
                ylabel('statistic (arb. unit^2)');
                saveas(fig,fullfile('reports','figures',strcat('noise',model_name_array{i_model},'.eps')),'epsc');

                %plot the mse, variance and bias^2 for each x
                fig = figure_latexSub();
                plot(this.x_plot,mse_plot);
                hold on;
                plot(this.x_plot,var_plot);
                plot(this.x_plot,bias_plot.^2);
                legend('MSE','VAR','BIAS^2','Location','northwest');
                xlabel('mean greyvalue (arb. unit)');
                ylabel('statistic (arb. unit^2)');
                saveas(fig,fullfile('reports','figures',strcat('mse',model_name_array{i_model},'.eps')),'epsc');

                %plot the bias for each x
                fig = figure_latexSub();
                plot(this.x_plot,bias_plot);
                xlabel('mean greyvalue (arb. unit)');
                ylabel('bias (arb. unit)');
                saveas(fig,fullfile('reports','figures',strcat('bias',model_name_array{i_model},'.eps')),'epsc');
               
            end
            
            
        end
        
        %GET MEAN VARIANCE
        %Get mean and variance vector using the images indicated by the parameter image_index
        %PARAMETERS:
            %image_index: vector of integers, points to which images to use for mean and variance estimation
        %RETURNS:
            %sample_mean: mean vector
            %sample_var: variance vector
        function [sample_mean,sample_var] = getMeanVar(this, image_index)
            %work out the mean and variance
            [sample_mean,sample_var] = this.mean_variance_estimator.getMeanVar(image_index);
        end
        
        %SAVE GREY VALUE ARRAY
        %Set up member variables
        function saveGreyvalueArray(this)
            
            scan = this.getScan();
            this.mean_variance_estimator = MeanVarianceEstimator(scan);
            this.mean_variance_estimator.saveGreyvalueArray(scan);
            
            this.n_train = round(scan.n_sample/2);
            this.y_array = zeros(this.n_plot, this.n_bootstrap);
            this.y_predict = zeros(this.n_plot, this.n_bootstrap, this.getNModel());
            this.shape_parameter = (this.n_train-1)/2;
            this.n_image = scan.n_sample;
            
            [sample_mean,~] = this.getMeanVar(1:this.n_image);
            this.x_plot = (linspace(min(sample_mean),max(sample_mean),this.n_plot))';
                
        end %saveGreyvalueArray
        
        %GET N MODEL
        %Return the number of models in this experiment
        function n_model = getNModel(this)
            n_model = 6;
        end
        
        %GET MODEL
        %Return a model given an index
        %PARAMETER:
        	%index: interger pointing to which model to return
        function model = getModel(this, index)
            switch index
                case 1
                    model = MeanVar_GLM(this.shape_parameter,1,LinkFunction_Identity());
                case 2
                    model = MeanVar_GLM(this.shape_parameter,-1,LinkFunction_Canonical());
                case 3
                    model = MeanVar_GLM(this.shape_parameter,-2,LinkFunction_Canonical());
                case 4
                    model = MeanVar_GLM(this.shape_parameter,1,LinkFunction_Log());
                case 5
                    model = MeanVar_GLM(this.shape_parameter,-1,LinkFunction_Log());
                case 6
                    model = MeanVar_kNN(1E3);
            end
        end
        
    end
    
    %ABSTRACT METHODS
    methods (Abstract)

    	%GET SCAN
    		%Returns a scan object, containing images of a dataset
        scan = getScan(this);
    end
    
end

