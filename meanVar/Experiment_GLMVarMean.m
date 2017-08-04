classdef Experiment_GLMVarMean < Experiment
    %EXPERIMENT_GLMVARMEAN Assess the performance of GLM fit on mean var data
    %   The images are spilt into 2 parts, training and test. GLM is used
    %   to model the variance and mean relationship, with variance as the
    %   response. The response is gamma randomlly distributed with known
    %   shape parameter.   
    %
    %   The images were segmented to only consider pixels from the ROI.
    %
    %   The training set is used to train the glm, which is then used to
    %   predict the variance of the test set. The training and mean
    %   standarised residuals are plotted, that is the residual divided by
    %   the std of gamma.
    
    %MEMBER VARIABLES
    properties
        
        i_repeat; %number of iterations done
        n_repeat; %number of itereations to complete the experiment
        n_sample; %number of images in a scan
        n_train; %number of images in the training set (half of n_sample)
        shape_parameter; %shape parameter of gamma
        
        %array of training and test mse
            %dim 1: for each repeat
            %dim 2: for each glm
        training_mse_array;
        test_mse_array;
        
        %segmentation boolean vectpr
        segmentation;
        %random stream
        rand_stream;
           
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %PARAMETERS:
            %experiment_name
        function this = Experiment_GLMVarMean(experiment_name)
            %call superclass
            this@Experiment(experiment_name);
        end

        %SET UP EXPERIMENT
        %PARAMETERS:
            %n_repeat: number of times to repeat the experiment
            %rand_uint32_seed: uint32 to set the random seed seed
        function setUpExperiment(this, n_repeat, rand_uint32_seed)
            
            %get the scan object
            scan = this.getScan();
            
            %assign member variables
            this.i_repeat = 1;
            this.n_repeat = n_repeat;
            this.n_sample = scan.n_sample;
            this.n_train = round(this.n_sample/2);
            this.shape_parameter = (this.n_train-1)/2;
            this.training_mse_array = zeros(this.n_repeat,this.getNGlm());
            this.test_mse_array = zeros(this.n_repeat,this.getNGlm());
            this.saveSegmentation(scan.getSegmentation());
            this.rand_stream = RandStream('mt19937ar','Seed',rand_uint32_seed);
        end
        
        
        %DO EXPERIMENT
        function doExperiment(this)
            
            %do n_repeat times
            while (this.i_repeat <= this.n_repeat)
            
                %use its random stream
                RandStream.setGlobalStream(this.rand_stream);
                %for each glm
                for i_glm = 1:this.getNGlm()
                    %get the training and test mse
                    [mse_training, mse_test] = this.trainingTestMeanVar(this.getGlm(this.shape_parameter, i_glm));
                    %save the training and test mse in the array
                    this.training_mse_array(this.i_repeat,i_glm) = mse_training;
                    this.test_mse_array(this.i_repeat,i_glm) = mse_test;
                end
                
                %print the progress
                this.printProgress(this.i_repeat / this.n_repeat);
                %increment i_repeat
                this.i_repeat = this.i_repeat + 1;
                %save the state of this experiment
                this.saveState();
                
            end
        end
        
        
        %TRAINING/TEST MEAN VAR
        %Gets the training and test MSE when fitting and predicting the mean and variance relationship
        %PARAMETERS:
            %model: variance model object
        %RETURN:
            %mse_training (scalar)
            %mse_test (scalar)
        function [mse_training, mse_test] = trainingTestMeanVar(this, model)

                %get random index of the training and test data
                index_suffle = randperm(this.n_sample);
                training_index = index_suffle(1:this.n_train);
                test_index = index_suffle((this.n_train+1):end);

                %get variance mean data of the training set
                [sample_mean,sample_var] = this.getMeanVar(training_index);

                %train the classifier
                model.train(sample_mean,sample_var);
                %get the training mse
                mse_training = model.getPredictionMSSE(sample_mean,sample_var);

                %get the variance mean data of the test set
                [sample_mean,sample_var] = this.getMeanVar(test_index);
                
                %get the test mse
                mse_test = model.getPredictionMSSE(sample_mean,sample_var);

        end
        
        %PRINT RESULTS
        %Box plot the training and test MSSE
        function printResults(this)
            
            figure;
            boxplot(this.training_mse_array,'boxstyle','filled','medianstyle','target','outliersize',4,'symbol','o');
            
            figure;
            boxplot(this.test_mse_array,'boxstyle','filled','medianstyle','target','outliersize',4,'symbol','o');
            
        end
        
        %PLOT FULL FIT
        %Plot the variance and mean histogram, along with the fitted glm
        %Using all n_sample images, for all GLM
        function plotFullFit(this)
            
            %get scan
            scan = this.getScan();
            %get the sample mean and variance using all images
            [sample_mean,sample_var] = this.getMeanVar(1:this.n_sample);
            
            %indicate sample means and variances which are not outliers
            mean_not_outlier = removeOutliers_iqr(sample_mean);
            var_not_outlier = removeOutliers_iqr(sample_var);
            %get boolean vector, true for values which are not outliers for both mean and variance
            not_outlier = mean_not_outlier & var_not_outlier;
            %remove outliers in the vector sample_mean and sample_var
            sample_mean = sample_mean(not_outlier);
            sample_var = sample_var(not_outlier);

            %shape parameter is number of (images - 1)/2, this comes from the chi
            %squared distribution
            full_shape_parameter = (scan.n_sample-1)/2;

            %for each glm
            for i_glm = 1:this.getNGlm()

                %get the glm
                model = this.getGlm(full_shape_parameter, i_glm);

                %train the glm
                model.train(sample_mean,sample_var);

                %plot the frequency density
                figure;
                ax = hist3Heatmap(sample_mean,sample_var,[this.getNBin(),this.getNBin()],false);
                hold on;

                %get a range of greyvalues to plot the fit
                x_plot = linspace(ax.XLim(1),ax.XLim(2),100);
                %get the variance prediction along with the error bars
                [variance_prediction, up_error, down_error] = model.predict(x_plot');

                %plot the fit/prediction
                plot(x_plot,variance_prediction,'r');
                %plot the error bars
                plot(x_plot,up_error,'r--');
                plot(x_plot,down_error,'r--');
            end
        end %plotFullFit
        
        %SAVE SEGMENTATION
        %Given segmentation from a scan object, save it as a vector
        function saveSegmentation(this, segmentation)
            this.segmentation = reshape(segmentation,[],1);
        end
        
        %GET MEAN VARIANCE
        %Get mean and variance vector using the images indicated by the parameter data_index
        %The mean and variance are already segmented
        %PARAMETERS:
            %data_index: vector of integers, points to which images to use for mean and variance estimation
        %RETURNS:
            %sample_mean: mean vector
            %sample_var: variance vector
        function [sample_mean,sample_var] = getMeanVar(this, data_index)
            %get the scan object
            scan = this.getScan();
            %get the sample mean and variance
            [sample_mean,sample_var] = scan.getSampleMeanVar_vector(data_index);
            %segment the mean var data
            sample_mean = sample_mean(this.segmentation);
            sample_var = sample_var(this.segmentation);
        end
        
    end
    
    methods (Abstract)
        
        %returns scan object
        scan = getScan(this);
        
        %returns number of glm to investigate
        n_glm = getNGlm(this);
        
        %returns glm model given index
        %index can range from 1 to getNGlm()
        model = getGlm(this, shape_parameter, index);
        
        %returns the number of bins to be used in histogram plotting
        n_bin = getNBin(this);
        
    end
    
end

