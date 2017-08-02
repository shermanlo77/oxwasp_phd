%EXPERIMENT REFRENCE SHADING CORRECTION MAR 16
%See superclass Experiment_referenceShadingCorrection
classdef Experiment_referenceShadingCorrection_Mar16 < Experiment_referenceShadingCorrection

    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = Experiment_referenceShadingCorrection_Mar16()
            %super class, pass on experiment name in parameter
            this@Experiment_referenceShadingCorrection('referenceShadingCorrection_Mar16');
        end
        
        %OVERRIDE METHOD: SET UP EXPERIMENT
        %Calls superclass and instantise random stream
        function setUpExperiment(this)
            this.setUpExperiment@Experiment_referenceShadingCorrection(100, 4);
            this.rand_stream = RandStream('mt19937ar','Seed',uint32(227482200));
        end
        
        %IMPLEMENT METHOD: LOAD DATA
        %Return scan object containing reference scans
        function bgw_data = loadData(this)
            bgw_data = Bgw_Mar16();
        end
        
        %IMPLEMENTED METHOD: doExperimentForAllShadingCorrections
        %calls shadingCorrection_ANOVA for different shading correctors
        function doExperimentForAllShadingCorrections(this)
            %no shading correction
            this.shadingCorrection_ANOVA(ShadingCorrector_null(), 1:this.n_reference);
            %bw shading correction
            this.shadingCorrection_ANOVA(ShadingCorrector(), [1,this.n_reference]);
            %linear shading correction
            this.shadingCorrection_ANOVA(ShadingCorrector(), 1:this.n_reference);
            %linear polynomial shading correction
            this.shadingCorrection_ANOVA(ShadingCorrector_polynomial([2,2,2]), 1:this.n_reference);
        end
        
        %OVERRIDE METHOD: PRINT RESULTS
        %Image plot the shading corrected image, one for each reference
        %calls superclass version method to plot variance vs power
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
                    subplot(1,3,i_ref,imagesc_truncate(this.image_array{i_ref,i_shad}));
                    colorbar(subplot(1,3,i_ref)); %include colour bar
                    axis(gca,'off'); %turn axis off
                    title(colour_array{i_ref}); %label the reference image
                end
                %save the figure
                saveas(fig,strcat('reports/figures/shadingCorrection/shadingCorrection_',shading_array{i_shad},'.png'));
            end
            
            %super class print results
            this.printResults@Experiment_referenceShadingCorrection();
        end
        
    end
    
end

