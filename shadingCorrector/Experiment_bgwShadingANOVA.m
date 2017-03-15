classdef Experiment_bgwShadingANOVA < Experiment
    %EXPERIMENT_BGWSHADINGANOVA Experiment for using ANOVA to assess the
    %performance of shading correction
        %Use n_train b/g/w images to train the shading corrector, then uses the
        %reminder of b/g/w images to do shading correction on. The variance
        %between and within pixel is recorded. Repeated n_repeat times
    
    %MEMBER VARIABLES
    properties
        
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
        
        %object containing the data
        block_data;
        
        %the size of the training set
        n_train;
        
        %random stream
        rand_stream;
        
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        function this = Experiment_bgwShadingANOVA()
            %superclass
            this@Experiment('bgwShadingANOVA');
            %assign member variables
            this.n_train = 10;
            this.block_data = BlockData_140316('../data/140316');
            this.rand_stream = RandStream('mt19937ar','Seed',uint32(227482200));
        end
        
        %DECLARE RESULT ARRAY
        %PARAMETERS:
            %n_repeat: number of times to repeat the experiment
        function declareResultArray(this,n_repeat)
            %assign member variables
            this.std_array = cell(4,1);
            this.bgw_shading_array = cell(3,4);
            for i = 1:4
                this.std_array{i} = zeros(n_repeat,2,3);
            end
            this.shading_correction_pointer = 1;
        end
        
        %DO EXPERIMENT (one iteration)
        function doExperiment(this)
            %use its random stream
            RandStream.setGlobalStream(this.rand_stream);
            %for the 4 different types of shading correction, get the
            %within and between pixel variance and save it to the member
            %variable std_array
            this.shadingCorrection_ANOVA(@ShadingCorrector_null, false, nan);
            this.shadingCorrection_ANOVA(@ShadingCorrector, false, nan);
            this.shadingCorrection_ANOVA(@ShadingCorrector, true, nan);
            this.shadingCorrection_ANOVA(@ShadingCorrector_polynomial, true, [2,2,2]);
            this.shading_correction_pointer = 1;
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
                %set up a string cell to store the results of within and between pixel variance
                table_string = cell(4,3);
                %label the heading row
                table_string{1,1} = '';
                table_string{1,2} = 'Within pixel variance';
                table_string{1,3} = 'Between pixel variance';
                %for each reference image
                for i_ref = 1:3
                    %put the name of the reference image in the first column
                    table_string{i_ref+1,1} = colour_array{i_ref};
                    %put the within pixel variance in the 2nd column
                    table_string{i_ref+1,2} = quoteQuartileError(this.std_array{i_shad}(:,1,i_ref), 100);
                    %put the between pixel variance in the 3rd column
                    table_string{i_ref+1,3} = quoteQuartileError(this.std_array{i_shad}(:,2,i_ref), 100);
                end
                %print the table in latex format
                printStringArrayToLatexTable(table_string, strcat('reports/tables/',this.experiment_name,'_',shading_array{i_shad},'.tex_table'));
            end
        end
        
        %SHADING CORRECTION ANOVA
        function shadingCorrection_ANOVA(this, shading_corrector_class, want_grey, parameters)
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

            %turn off shading correction when loading the b/g/w images
            this.block_data.turnOffShadingCorrection();

            %declare array of images, reference stack is an array of mean b/g/w images
            reference_stack = zeros(this.block_data.height, this.block_data.width, 2+want_grey);
            %load mean b/w images
            reference_stack(:,:,1) = mean(this.block_data.loadBlackStack(black_train),3);
            reference_stack(:,:,2) = mean(this.block_data.loadWhiteStack(white_train),3);
            %load mean grey images if requested
            if want_grey
                reference_stack(:,:,3) = mean(this.block_data.loadGreyStack(grey_train),3);
            end

            %instantise shading corrector using provided reference stack
            shading_corrector = feval(shading_corrector_class,reference_stack);

            %if parameters are provided, add it to the shading corrector
            %then add the shading corrector to the data
            if ~isnan(parameters)
                this.block_data.addManualShadingCorrector(shading_corrector,parameters);
            else
                this.block_data.addManualShadingCorrector(shading_corrector);
            end

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
    
    methods(Static)
        
        %GLOBAL: Call this to start experiment automatically
        function main()
            %repeat the experiment this many times
            n_repeat = 20;
            %set up the experiment
            Experiment.setUpExperiment(@Experiment_bgwShadingANOVA,n_repeat);
            %run the experiment, it will save results to reports folder
            Experiment.runExperiments('bgwShadingANOVA',n_repeat);
        end
        
    end
    
end

