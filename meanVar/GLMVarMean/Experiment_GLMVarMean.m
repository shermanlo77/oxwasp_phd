%EXPERIMENT_GLMVARMEAN Assess the performance of Gamma GLM fit on mean var data
%   The images are spilt into 2 sets, training and test. GLM is used
%   to model the variance and mean relationship, with variance as the
%   response. The response is gamma randomly distributed with known
%   shape parameter.   
%
%   The images were segmented to only consider pixels from the ROI.
%
%   The training set is used to train the glm, which is then used to
%   predict the variance of the test set. Various residuals are plotted
classdef Experiment_GLMVarMean < Experiment
    
    %MEMBER VARIABLES
    properties (SetAccess = protected)
        
        i_repeat; %number of folds done
        i_glm; %number of glm done
        i_shad; %number of shading corrections done
        i_iteration; %number of loops done
        n_iteration; %total number of loops done (for progress bar)
        n_repeat; %number of itereations to complete the experiment
        n_sample; %number of images in a scan
        n_train; %number of images in the training set (half of n_sample)
        
        shape_parameter; %shape parameter of gamma
        
        %array of training and test error
            %dim 1: for each repeat
            %dim 2: for each glm
            %dim 3: for each shading corrector
        training_msse_array;
        test_msse_array;
        training_mse_array;
        test_mse_array;
        
        %temporary variable, to be deleted after the experiment
        %stores the grey values of each masked pixel, for each image
        mean_variance_estimator;
        
        %random stream
        rand_stream;
        
        %cell array of shading corrector and glm names
        shading_corrector_array;
        glm_name_array;
        
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %experiment_name
        function this = Experiment_GLMVarMean(experiment_name)
            %call superclass
            this@Experiment(experiment_name);
        end
        
        %PRINT RESULTS
        %Box plot the training and test MSSE
        function printResults(this)
            this.plotBoxPlot(this.training_msse_array,'training MSSE');
            this.plotBoxPlot(this.test_msse_array,'test MSSE');
            this.plotBoxPlot(this.training_mse_array,'training MSE');
            this.plotBoxPlot(this.test_mse_array,'test MSE');
        end
        
    end
    
    %PROTECTED METHODS
    methods (Access = protected)

        %SET UP EXPERIMENT
        %PARAMETERS:
            %n_repeat: number of times to repeat the experiment
            %rand_steam: random stream
        function setup(this, n_repeat, rand_stream)
            
            %get the scan object
            scan = this.getScan();
            
            %assign member variables
            this.i_repeat = 1;
            this.i_glm = 1;
            this.i_shad = 1;
            this.i_iteration = 0;
            
            this.n_repeat = n_repeat;
            this.n_sample = scan.n_sample;
            this.n_train = round(this.n_sample/2);
            
            this.n_iteration = this.n_repeat * this.getNShadingCorrector() * this.getNGlm();
            
            this.shape_parameter = (this.n_train-1)/2;
            
            this.mean_variance_estimator = MeanVarianceEstimator(scan);
            
            this.assignArray();
            this.rand_stream = rand_stream;
        end
        
        %ASSIGN ARRAY
        %Declare arrays for storing the training and test MSE and MSSE
        function assignArray(this)
            this.training_msse_array = zeros(this.n_repeat,this.getNGlm(),this.getNShadingCorrector());
            this.test_msse_array = zeros(this.n_repeat,this.getNGlm(),this.getNShadingCorrector());
            
            this.training_mse_array = zeros(this.n_repeat,this.getNGlm(),this.getNShadingCorrector());
            this.test_mse_array = zeros(this.n_repeat,this.getNGlm(),this.getNShadingCorrector());
        end
        
        
        %DO EXPERIMENT
        function doExperiment(this)
            %set random stream
            RandStream.setGlobalStream(this.rand_stream);
            
            %for each shading correction
            while (this.i_shad <= this.getNShadingCorrector())
                %save the shading corrected greyvalues
                this.saveGreyvalueArray();
                %for each glm
                while (this.i_glm <= this.getNGlm())
                    %for each fold
                    while (this.i_repeat <= this.n_repeat)
                        %save the mse and msse, training and test
                        this.doIteration();
                        this.i_repeat = this.i_repeat + 1;
                        
                        this.i_iteration = this.i_iteration + 1;
                        this.printProgress(this.i_iteration / this.n_iteration);
                    end
                    this.i_repeat = 1;
                    this.i_glm = this.i_glm + 1;
                end
                this.i_glm = 1;
                this.i_shad = this.i_shad + 1;
            end
            
            %get array of shading corrector names
            this.shading_corrector_array = cell(this.getNShadingCorrector(),1);
            for i = 1:this.getNShadingCorrector()
                [shading_corrector,reference_index] = this.getShadingCorrector(i);
                shading_corrector.n_image = numel(reference_index);
                this.shading_corrector_array{i} = shading_corrector.getName();
            end
            
            %get array of glm names
            this.glm_name_array = cell(this.getNGlm(),1);
            for i = 1:this.getNGlm()
                model = this.getGlm(i);
                this.glm_name_array{i} = model.getName();
            end
            
            %delete variables
            this.deleteVariables();
            
        end
        
        %DELETE GREYVALUE ARRAY
        function deleteVariables(this)
            this.mean_variance_estimator = [];
        end
        
        %DO ONE ITERATION OF EXPERIMENT
        function doIteration(this)
            %get the training and test mse
            [training_error, test_error] = this.trainingTestMeanVar();
            %save the training and test mse in the array
            this.training_msse_array(this.i_repeat,this.i_glm,this.i_shad) = training_error(1);
            this.training_mse_array(this.i_repeat,this.i_glm,this.i_shad) = training_error(2);
            this.test_msse_array(this.i_repeat,this.i_glm,this.i_shad) = test_error(1);
            this.test_mse_array(this.i_repeat,this.i_glm,this.i_shad) = test_error(2);
        end
        
        %TRAINING/TEST MEAN VAR
        %Gets the training and test MSE when fitting and predicting the mean and variance relationship
        %PARAMETERS:
            %glm_index: integer, pointing to which glm to use
            %shading_index: integer, pointing to which shading corrector to use
        %RETURN:
            %training_error: two vector [msse; mse] 
            %test_error: two vector [msse; mse]
        function [training_error, test_error] = trainingTestMeanVar(this)
            
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
            %get the training mse
            training_error = model.getPredictionMSSE(sample_mean,sample_var);
            
            %get the variance mean data of the test set
            [sample_mean,sample_var] = this.getMeanVar(test_index);

            %get the test mse
            test_error = model.getPredictionMSSE(sample_mean,sample_var);

        end
        
        %PLOT BOX PLOT
        %Plots statisitcs for each glm and shading corrector
        %PARAMETERS:
            %stat_array: array of statistics
                %dim 1: for each repeat
                %dim 2: for each glm
                %dim 3: for each shading corrector
            %stat_name: name of the statistic
        function plotBoxPlot(this,stat_array, stat_name)
            %produce figure
            figure;
            ax = gca;
            for i = 1:this.getNShadingCorrector()
                plot(0,0);
                hold on;
            end
            %get the colours for each hold on
            colour_order = ax.ColorOrder;
            
            %for each shading correction
            for i = 1:this.getNShadingCorrector()
                %get the position of the box plot for this current shading correction
                position = (1:this.getNGlm())-0.25+0.5*(i)/(this.getNShadingCorrector()+1);
                %box plot the errors
                boxplot = Boxplots(stat_array(:,:,i),false);
                boxplot.setPosition(position);
                boxplot.setColour(colour_order(i,:));
                boxplot.plot();
            end
            %retick the x axis
            ax.XTick = 1:this.getNGlm();
            %label each glm with its name
            ax.XTickLabelRotation = 45;
            ax.XTickLabel = this.glm_name_array;
            ax.YLim = [max([min(min(min(stat_array))),ax.YLim(1)]),min([max(max(max(stat_array))),ax.YLim(end)])];
            %label the axis and legend
            ylabel(stat_name);
            legend(this.shading_corrector_array);
        end
        
        %PLOT FULL FIT
        %Plot the variance and mean histogram, along with the fitted glm
        %Using all n_sample images, for all GLM
        function plotFullFit(this)

            %shape parameter is number of (images - 1)/2, this comes from the chi
            %squared distribution
            scan = this.getScan();
            %save the gamma shape parameter
            this.shape_parameter = (scan.n_sample-1)/2;
            %instantise a mean variance estimator
            this.mean_variance_estimator = MeanVarianceEstimator(scan);
            
            %for each shading corrector
            this.i_shad = 1;
            while (this.i_shad <= this.getNShadingCorrector())
                
                %for this shading corrector, save the greyvalues
                this.saveGreyvalueArray();
                
                %for each glm
                for i = 1:this.getNGlm()
                    
                    %get the glm
                    model = this.getGlm(i);

                    %get the sample mean and variance
                    [sample_mean,sample_var] = this.getMeanVar(1:this.n_sample);
                    
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
                    ax = hist3Heatmap(sample_mean,sample_var,[this.getNBin(),this.getNBin()],false);
                    colorbar;
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
                    %label the axis
                    xlabel('mean (arb. unit)');
                    ylabel('variance (arb. unit^2)');
                end
                this.i_shad = this.i_shad + 1;
            end
            %delete the storage of greyvalues
            this.deleteVariables();
        end %plotFullFit
        
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
        %Set up the member variable mean_variance_estimator
        function saveGreyvalueArray(this)
            %get the scan object
            scan = this.getScan();
            %get the shading corrector
            [shading_corrector, reference_index] = this.getShadingCorrector(this.i_shad);
            %add the shading corrector
            scan.addShadingCorrector(shading_corrector, reference_index);
            %save the greyvalues
            this.mean_variance_estimator.saveGreyvalueArray(scan);
        end %saveGreyvalueArray
        
        %IMPLEMENTED: GET N BIN
        function n_bin = getNBin(this)
            n_bin = 100;
        end
        
    end
    
    methods (Abstract, Access = protected)
        
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

