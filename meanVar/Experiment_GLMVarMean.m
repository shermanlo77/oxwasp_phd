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
        
        %array of training and test error
            %dim 1: for each repeat
            %dim 2: for each glm
            %dim 3: for each shading corrector
        training_error_array;
        test_error_array;
        
        %array of greyvalues (to be deleted when experiment is done)
            %dim 1: for each segmented pixel
            %dim 2: for each image
            %dim 3: for each shading correction
        greyvalue_array;
        
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
            this.training_error_array = zeros(this.n_repeat,this.getNGlm(),this.getNShadingCorrector());
            this.test_error_array = zeros(this.n_repeat,this.getNGlm(),this.getNShadingCorrector());
            this.rand_stream = RandStream('mt19937ar','Seed',rand_uint32_seed);
            
            this.saveGreyvalueArray();
        end
        
        
        %DO EXPERIMENT
        function doExperiment(this)
            
            %do n_repeat times
            while (this.i_repeat <= this.n_repeat)
            
                %use its random stream
                RandStream.setGlobalStream(this.rand_stream);
                %for each glm
                for i_shad = 1:this.getNShadingCorrector()
                    for i_glm = 1:this.getNGlm()
                        %get the training and test mse
                        [error_training, error_test] = this.trainingTestMeanVar(this.getGlm(this.shape_parameter, i_glm), i_shad);
                        %save the training and test mse in the array
                        this.training_error_array(this.i_repeat,i_glm,i_shad) = error_training;
                        this.test_error_array(this.i_repeat,i_glm,i_shad) = error_test;
                    end
                end
                
                %print the progress
                this.printProgress(this.i_repeat / this.n_repeat);
                %increment i_repeat
                this.i_repeat = this.i_repeat + 1;
                %save the state of this experiment
                
            end
        end
        
        %DELETE VARIABLES
        %Delete variables when the experiment is completed
        function deleteVariables(this)
            this.greyvalue_array = [];
        end
        
        %TRAINING/TEST MEAN VAR
        %Gets the training and test MSE when fitting and predicting the mean and variance relationship
        %PARAMETERS:
            %model: variance model object
        %RETURN:
            %mse_training (scalar)
            %mse_test (scalar)
        function [error_training, error_test] = trainingTestMeanVar(this, model, shading_index)

                %get random index of the training and test data
                index_suffle = randperm(this.n_sample);
                training_index = index_suffle(1:this.n_train);
                test_index = index_suffle((this.n_train+1):end);

                %get variance mean data of the training set
                [sample_mean,sample_var] = this.getMeanVar(training_index, shading_index);

                %train the classifier
                model.train(sample_mean,sample_var);
                %get the training mse
                error_training = model.getPredictionMSSE(sample_mean,sample_var);

                %get the variance mean data of the test set
                [sample_mean,sample_var] = this.getMeanVar(test_index, shading_index);
                
                %get the test mse
                error_test = model.getPredictionMSSE(sample_mean,sample_var);

        end
        
        %PRINT RESULTS
        %Box plot the training and test MSSE
        function printResults(this)
            
            for i_shad = 1:this.getNShadingCorrector()
                figure;
                boxplot(this.training_error_array(:,:,i_shad),'boxstyle','filled','medianstyle','target','outliersize',4,'symbol','o');

                figure;
                boxplot(this.test_error_array(:,:,i_shad),'boxstyle','filled','medianstyle','target','outliersize',4,'symbol','o');
            end
            
        end
        
        %PLOT FULL FIT
        %Plot the variance and mean histogram, along with the fitted glm
        %Using all n_sample images, for all GLM
        function plotFullFit(this)
            
            this.saveGreyvalueArray();

            %shape parameter is number of (images - 1)/2, this comes from the chi
            %squared distribution
            scan = this.getScan();
            full_shape_parameter = (scan.n_sample-1)/2;

            %for each shading corrector
            for i_shad = 1:this.getNShadingCorrector()
                %for each glm
                for i_glm = 1:this.getNGlm()
                    
                    %get the glm
                    model = this.getGlm(full_shape_parameter, i_glm);

                    %get the sample mean and variance
                    [sample_mean,sample_var] = this.getMeanVar(1:numel(this.greyvalue_array(1,:,1)),i_shad);
                    
                    %train the glm
                    model.train(sample_mean,sample_var);
                    
                    %indicate sample means and variances which are not outliers
                    mean_not_outlier = removeOutliers_iqr(sample_mean);
                    var_not_outlier = removeOutliers_iqr(sample_var);
                    %get boolean vector, true for values which are not outliers for both mean and variance
                    not_outlier = mean_not_outlier & var_not_outlier;
                    %remove outliers in the vector sample_mean and sample_var
                    sample_mean = sample_mean(not_outlier);
                    sample_var = sample_var(not_outlier);

                    %plot the frequency density
                    figure;
                    ax = hist3Heatmap(sample_mean,sample_var,[this.getNBin(),this.getNBin()],true);
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
            end
            
            this.deleteVariables();
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
            %image_index: vector of integers, points to which images to use for mean and variance estimation
            %shading_index: which shading correction to use, integer
        %RETURNS:
            %sample_mean: mean vector
            %sample_var: variance vector
        function [sample_mean,sample_var] = getMeanVar(this, image_index, shading_index)
            selected_greyvalue_array = this.greyvalue_array(:,image_index,shading_index);
            sample_mean = mean(selected_greyvalue_array,2);
            sample_var = var(selected_greyvalue_array,[],2);
        end
        
        %SAVE GREY VALUE ARRAY
        %Set up the member variable greyvalue_array
        function saveGreyvalueArray(this)
            
            %get the segmentation
            segmentation = this.getSegmentation();
            %get the number of segmented pixels
            n_pixel = sum(sum(segmentation));
            
            %get the scan object
            scan = this.getScan();
            
            %declare the array greyvalue array
            this.greyvalue_array = zeros(n_pixel, scan.n_sample, this.getNShadingCorrector()); 
            
            %for each shading corrector
            for i_shad = 1:this.getNShadingCorrector()
                
                %get the scan object
                scan = this.getScan();
                %get the shading corrector
                [shading_corrector, reference_index] = this.getShadingCorrector(i_shad);
                %add the shading corrector
                scan.addShadingCorrector(shading_corrector, reference_index);
                
                %load the images and reshape it to be a design matrix
                image_stack = scan.loadImageStack();
                image_stack = reshape(image_stack,scan.area,scan.n_sample);

                %segment the design matrix
                image_stack = image_stack(segmentation,:);

                %add the greyvalues to the array
                this.greyvalue_array(:,:,i_shad) = image_stack;
            end
        end %saveGreyvalueArray
        
        %GET SEGMENTATION
        %Return the binary segmentation image
        %true for pixels to be considered
        function segmentation = getSegmentation(this)
            scan = this.getScan();
            segmentation = scan.getSegmentation();
        end
        
        %IMPLEMENTED: GET N BIN
        function n_bin = getNBin(this)
            n_bin = 100;
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

