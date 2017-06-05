%EXPERIMENT BGW NORMALITY TEST
    %Experiment for testing the Normality of each pixel greyvalue in the bgw images
    %The test is done for:
        %using the chi squared test and the kolmogorovSmirnov test
        %no shading correction, bw, bgw and polynomial shading correction
        %on the black, grey and white images
    %The 1st image is used for training the shading correction, the rest for the hypothesis test
classdef Experiment_bgwNormalityTest < Experiment
    
    %MEMBER VARIABLES
    properties
        
        %cell array of p values
        %each entry contains an image of p values
            %dim 1: for no, bw, bgw, polynonial shading correction
            %dim 2: for black, white, grey images
            %dim 3: chi squared test, ks test
        p_array;
        
        test_array; %cell array of test objects [chi squared, ks]
        i_test; %iterator for which test to use 
        i_shade; %iterator for which shading correction to use
        i_colour; %iterator for which colour (b/g/w) to do the test on
        
        progress_counter; %iterator, counts up to 24 for every image analysed
    
    end %properties
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        function this = Experiment_bgwNormalityTest()
            %call superclass with the experiment name
            this@Experiment('bgwNormalityTest');
        end %constructor
        
        %METHOD: SET UP EXPERIMENT
        function setUpExperiment(this)
            %declare member variables
            this.p_array = cell(4,3,2);
            this.test_array = cell(2,1);
            this.test_array{1} = NormalityTester_chiSquared(5);
            this.test_array{2} = NormalityTester_kolmogorovSmirnov();
            this.i_test = 1;
            this.i_shade = 1;
            this.i_colour = 1;
            this.progress_counter = 0;
        end %set up experiment
        
        %METHOD: DO EXPERIMENT
        %for each test, for each shading correction, for each colour...
        %conduct the normality test and save it to p_array
        function doExperiment(this)
            
            %while the experiment is not completed
            while ~this.is_complete
    
                %get the method for hypothesis testing
                tester = this.test_array{this.i_test};

                %for each shading correction
                while this.i_shade <= 4

                    %get the data
                    bgw_data = BGW_140316();

                    %apply the corresponding shading correction
                    %use the first image for training shading correction
                    switch this.i_shade
                        case 2
                            bgw_data.addShadingCorrector(ShadingCorrector(),[1,1]);
                        case 3
                            bgw_data.addShadingCorrector(ShadingCorrector(),[1,1,1]);
                        case 4
                            bgw_data.addShadingCorrector(ShadingCorrector_polynomial([2,2,2]),[1,1,1]);
                    end %switch

                    %for each colour
                    while this.i_colour <= 3

                        %conduct the  hypothesis test (using the 2nd to 20th image)
                        tester.doTest(bgw_data,this.i_colour,2:20);
                        %save the p value to the member variable p_array
                        this.p_array{this.i_shade,this.i_colour,this.i_test} = tester.p_value;
                        
                        %update iterator variables
                        this.i_colour = this.i_colour + 1;
                        
                        %save the experiment and print the progress bar
                        this.progress_counter = this.progress_counter + 1;
                        this.saveState(); %save the experiment
                        this.printProgress(this.progress_counter / 24);
                    end %while i_colour
                    this.i_colour = 1;
                    
                    this.i_shade = this.i_shade + 1;
                    
                end %while i_shade
                this.i_shade = 1;
                
                this.i_test = this.i_test + 1;
                
                %if both methods for hypothesis tests are done
                %set is_complete to be true
                if this.i_test == 3
                    this.is_complete = true;
                end
            
            end %while ~is_complete
            
        end %doExperiment
        
        
        %METHOD: PRINT RESULTS
        %Print the p value images from the experiment
        %Figure 1 for chi squared, figure 2 for ks
        %Each figure contain subplots with 4 rows and 3 columns
            %dim 1: for no, bw, bgw, polynonial shading correction
            %dim 2: for black, white, grey images
        function printResults(this)
            %for each test
            for j_test = 1:2
                figure; %set a figure
                %for each shading correction
                for j_shade = 1:4
                    %for each colour
                    for j_colour = 1:3
                        p_array_local = this.p_array{j_shade,j_colour,j_test};
                        [height, width] = size(p_array_local);
                        
                        %plot p value image
                        subplot(4,3,(j_shade-1)*3 + j_colour);
                        imagesc(p_array_local,[0,1]);
                        colorbar;
                        axis(gca,'off');
                        
                        %highlight significant pixels
                        hold on;
                        %find the significant pixels using FDR
                        [x_sig, y_sig] = find( reshape(significantFDR(reshape(p_array_local,1,[]), 0.05), height, width) );
                        %scatter plot the significant pixels
                        scatter(x_sig, y_sig, 'r');
                        hold off;
                        
                    end
                end
            end 
        end %printResults
        
    end %methods
    
end %class

