classdef Experiment_bgwShadingANOVA < Experiment
    %EXPERIMENT_BGWSHADINGANOVA Experiment for using ANOVA to assess the
    %performance of shading correction
        %Use n_train b/g/w images to train the shading corrector, then uses the
        %reminder of b/g/w images to do shading correction on. The variance
        %between and within pixel is recorded. Repeated n_repeat times
    
    %MEMBER VARIABLES
    properties
        
        i_repeat; %number of iterations done
        n_repeat; %number of times to repeat the experiment
               
        block_data;  %object containing the data
        n_train; %the size of the training set
        
        %random stream
        rand_stream;
        
        %cell of arrays, one array for each shading corrector
        %each array is 3 dimensional, containing variances
            %dim 1: for each repeat (n_repeat length)
            %dim 2: within pixel, between pixel (length 2)
            %dim 3: b/g/w (length 2 or 3)
        std_array;
        
        %cell of shading corrected b/g/w images
            %dim 1: b/g/w
            %dim 2: for each shading corrector
        bgw_shading_array;
        
        %dim 2 pointer for bgw_shading_array
        shading_correction_pointer;
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        function this = Experiment_bgwShadingANOVA()
            %superclass
            this@Experiment('bgwShadingANOVA');
        end
        
        %DECLARE RESULT ARRAY
        %PARAMETERS:
            %n_repeat: number of times to repeat the experiment
        function setUpExperiment(this)
            %assign member variables
            this.i_repeat = 1;
            this.n_repeat = 100;
            this.block_data = BGW_140316();
            this.n_train = 1;
            this.rand_stream = RandStream('mt19937ar','Seed',uint32(227482200));
            this.std_array = cell(4,1);
            this.bgw_shading_array = cell(3,4);
            for i = 1:4
                this.std_array{i} = zeros(this.n_repeat,2,3);
            end
            this.shading_correction_pointer = 1;
        end
        
        %DO EXPERIMENT (one iteration)
        function doExperiment(this)
            
            %for this.n_repeat times
            while (this.i_repeat <= this.n_repeat)
            
                %use its random stream
                RandStream.setGlobalStream(this.rand_stream);
                %for the 4 different types of shading correction, get the
                %within and between pixel variance and save it to the member
                %variable std_array
                this.shadingCorrection_ANOVA(ShadingCorrector_null(), false);
                this.shadingCorrection_ANOVA(ShadingCorrector(), false);
                this.shadingCorrection_ANOVA(ShadingCorrector(), true);
                this.shadingCorrection_ANOVA(ShadingCorrector_polynomial([2,2,2]), true);
                this.shading_correction_pointer = 1;
                
                %print the progress
                this.printProgress(this.i_repeat / this.n_repeat);
                %increment i_repeat
                this.i_repeat = this.i_repeat + 1;
                %save the state of this experiment
                this.saveState();
                
            end
        end
        
        %PRINT RESULTS
        %Prints the shading corrected b/g/w images for the 4 different
        %types of shading correction
        %Prints the table of between and within pixel variances
        function printResults(this)
            %define names of the colours and shading corrections
            colour_array = {'Black','Grey','White'};
            shading_array = {'no_shad','bw','bgw','polynomial'};
            %for each shading correction
            for i_shad = 1:4
                %set up the figure
                fig = figure;
                fig.Position(3) = 1000;
                fig.Position(4) = 200;
                %for each reference image
                for i_ref = 1:3
                    %plot the shading corrected reference image
                    subplot(1,3,i_ref,imagesc_truncate(this.bgw_shading_array{i_ref,i_shad}));
                    colorbar(subplot(1,3,i_ref)); %include colour bar
                    axis(gca,'off'); %turn axis off
                    title(colour_array{i_ref}); %label the reference image
                end
                %save the figure
                saveas(fig,strcat('reports/figures/shadingCorrection/shadingCorrection_',shading_array{i_shad},'.png'));
            end
            
            %for each shading correction
            for i_shad = 1:4
                %variance shall be stated in log scale
                figure;
                %get the min and maximum variance, offset it by 0.1
                min_v = min(min(min(log10(this.std_array{i_shad}))))-0.1;
                max_v = max(max(max(log10(this.std_array{i_shad}))))+0.1;
                %for each reference image
                for i_ref = 1:3
                    %box plot the within and between variance
                    subplot(1,3,i_ref);
                    boxplot(log10(this.std_array{i_shad}(:,:,i_ref)),{'W','B'});
                    %on the left hand side, label the axis
                    if i_ref == 1
                        ylabel('variance (log_{10})');
                    end
                    %label the colour
                    xlabel(colour_array{i_ref});
                    %set the y limit using the minimum and maximum variance
                    ylim([min_v,max_v]);
                end
            end
        end
        
        %SHADING CORRECTION ANOVA
        function shadingCorrection_ANOVA(this, shading_corrector, want_grey)
            %PARAMETERS:
                %data_object: object which loads the data
                %n_train: number of images to be used for training the shading corrector
                %shading_corrector_class: function handle which will be used for instantiating a new shading corrector
                %want_grey: boolean, true to use grey images for training the shading corrector
                %parameters: nan or vector of parameters for smoothing in shading correction
                %n_repeat: number of times to repeat the experiment

            %get the training and test black images index
            index = randperm(this.block_data.n_black);
            black_train = index(1:this.n_train);
            black_test = index((this.n_train+1):end);

            %get the training and test white images index
            index = randperm(this.block_data.n_white);
            white_train = index(1:this.n_train);
            white_test = index((this.n_train+1):end);

            %get the training and test grey images index
            index = randperm(this.block_data.n_grey);
            grey_train = index(1:this.n_train);
            grey_test = index((this.n_train+1):end);

            %declare array of images, reference stack is an array of mean b/g/w images
            reference_index = zeros(this.n_train, 2+want_grey);
            %load mean b/w images
            reference_index(:,1) = black_train';
            reference_index(:,2) = white_train';
            %load mean grey images if requested
            if want_grey
                reference_index(:,3) = grey_train';
            end

            %add the shading corrector to the data
            this.block_data.addShadingCorrector(shading_corrector,reference_index);

            %turn on remove dead pixels
            this.block_data.turnOnRemoveDeadPixels();

            %test_stack_array is a collection of array of b/g/w images
            test_stack_array = cell(1,3); %one array for each colour

            %load the test b/g/w images as an array and save it to test_stack_array
            test_stack_array{1} = this.block_data.loadBlackStack(black_test);
            test_stack_array{2} = this.block_data.loadGreyStack(grey_test);
            test_stack_array{3} = this.block_data.loadWhiteStack(white_test);

            %for each colour b/g/w test images
            for i_ref = 1:3

                %get the mean shading corrected image
                mean_image = mean(test_stack_array{i_ref},3);
                %if this is the first run, save the mean shading corrected image
                if this.i_repeat == 1
                    this.bgw_shading_array{i_ref, this.shading_correction_pointer} = mean_image;
                end

                %get the mean of all greyvalues in the mean image
                mean_all = mean(reshape(mean_image,[],1));

                %get the number of test images
                n_test = size(test_stack_array{i_ref},3);

                %save the within pixel variance
                this.std_array{this.shading_correction_pointer}(this.i_repeat,1,i_ref) = sum(sum(sum( ( test_stack_array{i_ref} - repmat(mean_image,1,1,n_test) ).^2 ))) / (this.block_data.area*n_test - this.block_data.area);
                %save the between pixel variance
                this.std_array{this.shading_correction_pointer}(this.i_repeat,2,i_ref) = n_test * sum(sum((mean_image - mean_all).^2))/(this.block_data.area-1);

            end

            %increment the shading correction pointer
            this.shading_correction_pointer = this.shading_correction_pointer + 1;
            
        end
        
    end   
    
end

