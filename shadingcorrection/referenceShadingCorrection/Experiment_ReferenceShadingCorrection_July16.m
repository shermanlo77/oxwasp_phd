%EXPERIMENT REFRENCE SHADING CORRECTION JULY 16
%See superclass Experiment_referenceShadingCorrection
classdef Experiment_ReferenceShadingCorrection_July16 < Experiment_ReferenceShadingCorrection
    
    properties
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = Experiment_ReferenceShadingCorrection_July16()
            %super class, pass on experiment name in parameter
            this@Experiment_ReferenceShadingCorrection('ReferenceShadingCorrection_July16');
        end
        
    end
    
    %PROTECTED METHODS
    methods (Access = protected)
        
        %OVERRIDE METHOD: SET UP EXPERIMENT
        %Calls superclass and instantise random stream
        function setup(this)
            %super class, 100 repeats, 3 shading correctors, define random stream
            this.setup@Experiment_ReferenceShadingCorrection(100, 3, RandStream('mt19937ar','Seed',uint32(2604655634)));
        end
        
        %IMPLEMENT METHOD: LOAD DATA
        %Return scan object containing reference scans
        function bgw_data = loadData(this)
            bgw_data = AbsBlock_July16([],[]);
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
            end
        end
        
    end
    
end

