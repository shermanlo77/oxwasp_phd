%EXPERIMENT REFRENCE SHADING CORRECTION MAR 16
%See superclass Experiment_referenceShadingCorrection
classdef Experiment_referenceShadingCorrection_Mar16 < Experiment_referenceShadingCorrection

    properties (SetAccess = protected)
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = Experiment_referenceShadingCorrection_Mar16()
            %super class, pass on experiment name in parameter
            this@Experiment_referenceShadingCorrection('referenceShadingCorrection_Mar16');
        end
        
        %OVERRIDE METHOD: PRINT RESULTS
        %Image plot the shading corrected image, one for each reference
        %calls superclass version method to plot variance vs power
        function printResults(this)
            %define names of the colours and shading corrections
            colour_array = {'Black','Grey','White'};
            shading_array = {'no_shad','bw','bgw','polynomial'};
            %for each shading correction
            for i_shad = 1:this.n_shading_corrector
                %set up the figure
                fig = figure;
                fig.Position(3) = 1000;
                fig.Position(4) = 200;
                %for each reference image
                for i_ref = 1:this.n_reference
                    %plot the shading corrected reference image
                    subplot(1,this.n_reference,i_ref,imagesc_truncate(this.image_array{i_ref,i_shad}));
                    colorbar(subplot(1,this.n_reference,i_ref)); %include colour bar
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
    
    %PROTECTED METHODS
    methods (Access = protected)
        
        %OVERRIDE METHOD: SET UP EXPERIMENT
        %Calls superclass and instantise random stream
        function setup(this)
            %super class, 100 repeats, 4 shading correctors, define random stream
            this.setup@Experiment_referenceShadingCorrection(100, 4, RandStream('mt19937ar','Seed',uint32(227482200)));
        end
        
        %IMPLEMENT METHOD: LOAD DATA
        %Return scan object containing reference scans
        function bgw_data = loadData(this)
            bgw_data = Bgw_Mar16();
        end
        
        %IMPLEMENT METHOD: GET SHADING CORRECTOR
            %returns a newly instantised shading corrector with the reference_index
            %parameter: integer, ranging from 1 to this.getNShadingCorrector
        function [shading_corrector, reference_index] = getShadingCorrector(this, index)
            switch index
                %no shading correction
                case 1
                    shading_corrector = ShadingCorrector_null();
                    reference_index = [];
                %bw shading correction
                case 2
                    shading_corrector = ShadingCorrector();
                    reference_index = [1,this.reference_white];
                %linear shading correction
                case 3
                    shading_corrector = ShadingCorrector();
                    reference_index = 1:this.reference_white;
                %polynomial shading correction
                case 4
                    shading_corrector = ShadingCorrector_polynomial([2,2,2]);
                    reference_index = 1:this.reference_white;
            end
        end
          
    end
    
end

