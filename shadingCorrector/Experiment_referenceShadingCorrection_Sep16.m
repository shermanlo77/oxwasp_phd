%EXPERIMENT REFRENCE SHADING CORRECTION SEP 16
%See superclass Experiment_referenceShadingCorrection
classdef Experiment_referenceShadingCorrection_Sep16 < Experiment_referenceShadingCorrection
    
    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = Experiment_referenceShadingCorrection_Sep16()
            %super class, pass on experiment name in parameter
            this@Experiment_referenceShadingCorrection('referenceShadingCorrection_Sep16');
        end
        
        %OVERRIDE METHOD: SET UP EXPERIMENT
        %Calls superclass and instantise random stream
        function setUpExperiment(this)
            %super class, 100 repeats, 3 shading correctors
            this.setUpExperiment@Experiment_referenceShadingCorrection(100, 3);
            %instantise random stream
            this.rand_stream = RandStream('mt19937ar','Seed',uint32(3056080743));
        end
        
        %IMPLEMENT METHOD: LOAD DATA
        %Return scan object containing reference scans
        function bgw_data = loadData(this)
            bgw_data = AbsBlock_Sep16([],[]);
        end
        
        %IMPLEMENTED METHOD: doExperimentForAllShadingCorrections
        %calls shadingCorrection_ANOVA for different shading correctors
        function doExperimentForAllShadingCorrections(this)
            %no shading correction
            this.shadingCorrection_ANOVA(ShadingCorrector_null(), []);
            %bw shading correction
            this.shadingCorrection_ANOVA(ShadingCorrector(), [1,this.reference_white]);
            %linear shading correction
            this.shadingCorrection_ANOVA(ShadingCorrector(), 1:this.reference_white);
        end
        
    end
    
end

