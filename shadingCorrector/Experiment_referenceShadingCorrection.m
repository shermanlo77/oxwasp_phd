classdef Experiment_referenceShadingCorrection < Experiment
    %EXPERIMENT_REFERNECESHADINGCORRECTION (ABSTRACT)
    %Estimates the between and within pixel variance of the reference images pre/post shading correction
        %In a repeat of the experiment, one image from each reference scan is used to train the shading correction
        %The remaining images are used to estimate the between/within variance, one for each reference scan
        %Between/within variance is plotted vs power
    %Methods to be implemented:
        %loadData(this)
            %returns a Scan object containing the reference scans
        %doExperimentForAllShadingCorrections(this)
            %calls shadingCorrection_ANOVA for different shading correctors
    %Other things to be implemented:
        %rand_stream needs to be instantised in setUpExperiment()
    
    %MEMBER VARIABLES
    properties
        
        i_repeat; %number of iterations done
        i_shading_corrector; %number of shading corrections investigated - 1 in the current iteration
        n_repeat; %number of times to repeat the experiment
        n_shading_corrector; %number of shading correctors to investigate per iteration of the experiment
        n_reference; %number of reference scans
        n_sample; %number of images per scan
        rand_stream; %random stream
        
        %array of between and within variances for each iteration and shading correction
            %dim 1: for each iteration
            %dim 2: for each reference or power
            %dim 3: for each shading correction
        var_b_array;
        var_w_array;
        
        %cell of shading corrected b/g/w images
            %dim 1: for each reference or power
            %dim 2: for each shading corrector
        image_array;
        
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %PARAMETERS: name of the experiment
        function this = Experiment_referenceShadingCorrection(name)
            %superclass
            this@Experiment(name);
        end
        
        %SET UP EXPERIMENT
        %PARAMETERS:
            %n_repeat: number of times to repeat the experiment
            %n_shading_corrector: number of shading corrections to be investigated
        function setUpExperiment(this, n_repeat, n_shading_corrector)
            %load the data
            bgw_data = this.loadData();
            
            %assign member variables
            this.i_repeat = 1;
            this.i_shading_corrector = 1;
            this.n_repeat = n_repeat;
            this.n_shading_corrector = n_shading_corrector;
            this.n_reference = bgw_data.getNReference();
            this.n_sample = bgw_data.reference_scan_array(1).n_sample;
            this.var_b_array = zeros(this.n_repeat, this.n_reference, this.n_shading_corrector);
            this.var_w_array = zeros(this.n_repeat, this.n_reference, this.n_shading_corrector);
            this.image_array = cell(this.n_reference, this.n_shading_corrector);
        end
        
        %DO EXPERIMENT (one iteration)
        function doExperiment(this)
            
            %for this.n_repeat times
            while (this.i_repeat <= this.n_repeat)
            
                %use its random stream
                RandStream.setGlobalStream(this.rand_stream);
                
                %do experiment for all shading correctors to be investigated
                this.doExperimentForAllShadingCorrections();
                
                %reset the member variable i_shading_corrector to 1
                this.i_shading_corrector = 1;
                %print the progress
                this.printProgress(this.i_repeat / this.n_repeat);
                %increment i_repeat
                this.i_repeat = this.i_repeat + 1;
                
            end
        end
        
        %PRINT RESULTS
        %Plots the between/within variance for each shading correction
            %Between/within variance vs power as a box plot
        function printResults(this)
            %get the array of powers
            bgw_data = this.loadData();
            power_array = bgw_data.getPowerArray();
            
            %shift the between/within variance box plots by this amount
            shift = max(power_array)/250;
            
            %for each shading correction
            for i_shad = 1:this.n_shading_corrector
                
                figure;
                
                %plot blue and red invisible lines
                %this is to set the legend
                plot(0,0,'b');
                hold on;
                plot(0,0,'r');
                
                %invisible plot the between and within variance vs power
                %this is to get the default properties of the x axis
                %box plot changes the properties of the x axis
                plot(power_array,mean(this.var_b_array(:,:,i_shad)),'LineStyle','none');
                plot(power_array,mean(this.var_w_array(:,:,i_shad)),'LineStyle','none');
                %get the x axis properties
                ax = gca;
                x_tick = ax.XTick;
                x_tick_label = ax.XTickLabel;
                
                %box plot between/within variance vs power
                %shift the box plots horizontally a little bit from the actual x position
                boxplot(this.var_b_array(:,:,i_shad),'position', power_array-shift,'boxstyle','filled','medianstyle','target','outliersize',4,'symbol','o','Colors','b');
                boxplot(this.var_w_array(:,:,i_shad),'position', power_array+shift,'boxstyle','filled','medianstyle','target','outliersize',4,'symbol','o','Colors','r');
                
                %set the x axis properties to be the default before box plot changed it
                ax = gca;
                ax.XTick = x_tick;
                ax.XTickLabel = x_tick_label;
                
                %set y scale to be log
                ax.YScale = 'log';
                
                %label the axis, set the limits and plot the legent
                ylim_min = min([min(min(min(this.var_b_array))),min(min(min(this.var_w_array)))]);
                ylim_max = max([max(max(max(this.var_b_array))),max(max(max(this.var_w_array)))]);
                ylim([ylim_min,ylim_max]);
                xlabel('Power (W)');
                ylabel('variane (arb. unit^2)');
                legend('between','within');
            end
        end
        
        %SHADING CORRECTION ANOVA
        %Estimates the between and within variance and saves it to the array this.var_b_array and this.var_w_array
        %A random image from each reference image is used for training the shading correction
        %The rest of the images are used in the variance prediction
        %The memver variable this.i_shading_corrector is incremented
        %PARAMETERS:
            %shading_corrector: shading corrector to be used
            %reference_index: row vector containg integers, indicates which reference scans to be used in shading correction training
        function shadingCorrection_ANOVA(this, shading_corrector, reference_index)
            
            %get the data
            bgw_data = this.loadData();
            %declare array of integers for image pointing to each reference scan
                %dim 1: for each image
                %dim 2: for each reference scan
            image_index = zeros(this.n_sample, numel(reference_index));
            
            %for each reference scan
            for i = 1:this.n_reference
                %get a random permutation of integers and save it to image_index
                image_index(:,i) = randperm(this.n_sample)';
            end

            %add the shading correction to the data
            %use 1 image per reference scan for shading correction training
            %the 1st row of image indecies are used to shading correction
            bgw_data.addShadingCorrector(shading_corrector,reference_index,image_index(1,reference_index));            
            %turn on remove dead pixels
            bgw_data.turnOnRemoveDeadPixels();
            
            %remove the image indices which were used for shading correction
            image_index(1,:) = [];

            %for each reference scan
            for i_ref = 1:this.n_reference
                
                %load shading corrected images, pointed by the pointers in image_index
                ref_stack = bgw_data.reference_scan_array(i_ref).loadImageStack(image_index(:,i_ref));

                %if this is the first run, save the first shading corrected image
                if this.i_repeat == 1
                    this.image_array{i_ref, this.i_shading_corrector} = ref_stack(:,:,1);
                end

                %estimated the between and within variance of the stack of images
                [var_b, var_w] = Experiment_referenceShadingCorrection.var_between_within(ref_stack);
                
                %save the within pixel variance
                this.var_b_array(this.i_repeat, i_ref, this.i_shading_corrector) = var_b;
                %save the between pixel variance
                this.var_w_array(this.i_repeat, i_ref, this.i_shading_corrector) = var_w;

            end
            
            %increment the shading corrector counter
            this.i_shading_corrector = this.i_shading_corrector + 1;
        end
        
    end
    
    %ABSTRACT METHODS
    methods (Abstract)
        
        %loadData(this)
            %returns a Scan object containing the reference scans
        loadData(this);
        
        %doExperimentForAllShadingCorrections(this)
            %calls shadingCorrection_ANOVA for different shading correctors
        doExperimentForAllShadingCorrections(this);
    end
    
    %STATIC METHODS
    methods (Static)
        
        %FUNCTION: BETWEEN WITHIN VARIANCE
        %Estimates the between and within pixel variance, given a stack of images
        %PARAMETERS:
            %image_stack: stack of images
                %dim 1: for each row
                %dim 2: for each column
                %dim 3: for each image
        %RETURN:
            %var_b: between pixel variance
            %var_w: within pixel variance
        function [var_b, var_w] = var_between_within(image_stack)

            %get the dimensions of the image stack
            [height, width, n] = size(image_stack);
            %work out the area
            area = height*width;

            %get the mean for each pixel
            mean_image = mean(image_stack,3);
            %get the global mean
            mean_all = mean(reshape(mean_image,[],1));

            %estimate the within pixel variance
            var_w = sum(sum(sum( ( image_stack - repmat(mean_image,1,1,n) ).^2 ))) / (area*n - area);
            %estimate the between pixel variance
            var_b = n * sum(sum((mean_image - mean_all).^2))/(area-1);

        end
        
    end
    
end
