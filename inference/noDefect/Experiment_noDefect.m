%ABSTRACT CLASS: EXPERIMENT NO DEFECT
%Experiment for investigating the FDR rate when doing inference on an image with a smooth function added to it
%Different thresholds and strength of the function are investigated
classdef Experiment_noDefect < Experiment
    
    %MEMBER VARIABLES
    properties (SetAccess = private)
        rng; %random number generator
        parameter_array; %array of parameters for the function
        sigma_array; %array of test threshold
        n_repeat; %number of times to repeat the experiment
        %array of fdr for each sigma, each parammeter and each repeat
            %dim of each element of the cell array:
                %dim 1: for each n_repeat
                %dim 2: for each parameter
                %dim 3: for each sigma
        fdr_array;
        plot_index; %2 column vector, pointer to which sigma and parameter to plot respectively
        convolution_plot; %resulting convolution to plot
        aRTist_plot; %image of aRTist, pointed by plot_index
        
        %progress bar member variables
        i_iteration;
        n_iteration;
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %name: name of experiment
        function this = Experiment_noDefect(name)
            %call superclass
            this@Experiment(name);
        end
        
        %IMPLEMENTED: PRINT RESULTS
        %PARAMETERS:
            %parameter_name: name of the parameter of axis label purposes
        function printResults(this, parameter_name)
            
            %get the maximum fdr
            fdr_max = max(max(max(this.fdr_array)));
            
            %for each sigma
            for i_sigma = 1:numel(this.sigma_array)
                %boxplot the false positive rate vs parameter
                figure;
                box_plot = Boxplots(this.fdr_array(:,:,i_sigma),true);
                box_plot.setPosition(this.parameter_array);
                box_plot.plot();
                ylabel('false positive rate');
                xlabel(parameter_name);
                ylim([0,fdr_max]);
            end
            
            %meshgrid to plot mean FPR vs parameter vs sigma
            [sigma_grid, parameter_grid] = meshgrid(this.sigma_array, this.parameter_array);
            figure;
            surf(parameter_grid,sigma_grid,squeeze(mean(this.fdr_array)));
            xlabel(parameter_name);
            ylabel('sigma threshold');
            zlabel('mean false positive rate');
            
            %plot aRTist
            figure;
            image_plot = ImagescSignificant(this.aRTist_plot);
            image_plot.addSigPixels(this.convolution_plot.sig_image);
            image_plot.plot();
            
            %print emperical null mean
            figure;
            image_plot = ImagescSignificant(this.convolution_plot.mean_null);
            image_plot.plot();
            
            %print -ln p value
            figure;
            image_plot = ImagescSignificant(-log10(this.convolution_plot.p_image));
            image_plot.plot();
            
        end
    end
    
    %PROTECTED METHODS
    methods (Access = protected)
        
        %IMPLEMENTED: SETUP EXPERIMENT
        %PARAMETERS:
            %rng: random number generator
            %parameter_array: array of parameters for the function
        function setup(this, rng, parameter_array)
            this.rng = rng;
            this.parameter_array = parameter_array;
            this.sigma_array = [2,3,4,5];
            this.n_repeat = 20;
            this.fdr_array = zeros(this.n_repeat, numel(this.parameter_array), numel(this.sigma_array));
            this.plot_index = [1;numel(this.parameter_array)];
            this.i_iteration = 0;
            this.n_iteration = numel(this.parameter_array) * this.n_repeat;
        end
        
        %IMPLEMENTED: DO EXPERIMENT
        function doExperiment(this)
            
            %get the data
            data = this.getData();
            %get the segmentation image
            segmentation = data.getSegmentation();
            %get the number of segmented images
            n_pixel = sum(sum(segmentation));
            %get the number of training images
            n_train = round(data.n_sample/2);
            
            %for each parameter
            for i_parameter = 1:numel(this.parameter_array)
                
                %get the defect simulator for this parameter
                defect_simulator = this.getDefectSimulator(this.parameter_array(i_parameter));
                
                %for this.n_repeat times
                for i_repeat = 1:this.n_repeat
                    
                    %get random permutation for each image
                    index = this.rng.randperm(data.n_sample);
                    %assign each permulation for mean-var training, 1 test image and aRTist
                    meanvar_index = index(1:(n_train-1));
                    test_index = index(n_train-1);
                    artist_index = index((n_train+1):end);
                    
                    %simulate aRTist
                    aRTist = mean(data.loadImageStack(artist_index),3);
                    %add function to aRTist
                    aRTist = defect_simulator.defectImage(aRTist);
                    
                    %get the training images
                    training_stack = data.loadImageStack(meanvar_index);
                    
                    %segment the image
                    training_stack = reshape(training_stack,data.area,n_train-1);
                    training_stack = training_stack(reshape(segmentation,[],1),:);
                    %get the segmented mean and variance greyvalue
                    training_mean = mean(training_stack,2);
                    training_var = var(training_stack,[],2);
                    %train glm using the training set mean and variance
                    model = GlmGamma(1,IdentityLink());
                    model.setShapeParameter((n_train-2)/2);
                    model.train(training_mean,training_var);
                    
                    %get the test images
                    test = data.loadImageStack(test_index);
                    
                    %predict variance given test image
                    var_predict = reshape(model.predict(reshape(test,[],1)), data.height, data.width);
                    
                    %get the z statistics and mask the z image
                    z_image = (test - aRTist)./sqrt(var_predict);
                    z_image(~segmentation) = nan;
                    
                    %get the emperical null
                    convolution = EmpericalConvolution(z_image,20, 20, [200,200]);
                    convolution.setUseVarUniform(true);
                    convolution.estimateNull(1000);
                    convolution.setMask(segmentation);
                    
                    %print progress bar
                    this.i_iteration = this.i_iteration + 1;
                    this.printProgress(this.i_iteration / this.n_iteration);
                    
                    %for each sigma to investigate
                    for i_sigma = 1:numel(this.sigma_array)
                        %set the threshold of the test and do the test
                        convolution.setSigma(this.sigma_array(i_sigma));
                        convolution.doTest();
                        %all positives are false, save the FDR
                        this.fdr_array(i_repeat,i_parameter,i_sigma) = sum(sum(convolution.sig_image))/n_pixel;
                        
                        %if this is the first repeat and this particular sigma and parameter is to be plotted
                        if ( (i_repeat == 1) && all([i_sigma;i_parameter] == this.plot_index) )
                            %save the convolution
                            this.convolution_plot = convolution;
                            %save the aRTist image
                            this.aRTist_plot = aRTist;
                        end
                        
                    end
                end
            end
        end
        
        %METHOD: GET DATA
        %Return an object containing images
        function data = getData(this)
            data = AbsBlock_July16_30deg();
            data.addDefaultShadingCorrector();
        end
        
    end
    
    %ABSTRACT METHODS
    methods (Abstract, Access = protected)
        %ABSTRACT METHOD: GET DEFECT SIMULATOR
        %Return defect simulator which adds a smooth function given a parameter
        defect_simulator = getDefectSimulator(this, parameter);
    end
    
end

