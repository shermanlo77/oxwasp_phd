classdef Experiment_GlmMse_Sep16_30deg < Experiment_GlmMse_Sep16
    
    properties
    end
    
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = Experiment_GlmMse_Sep16_30deg()
            %call superclass with experiment name
            this@Experiment_GlmMse_Sep16('GlmMse_Sep16_30deg');
        end
        
    end
    
    methods (Access = protected)
        
        %OVERRIDE: SET UP EXPERIMENT
        function setup(this)
            %call superclass with 100 repeats and a random stream
            this.setup@Experiment_GlmMse_Sep16(RandStream('mt19937ar','Seed',uint32(1562360917)));
        end
        
        %IMPLEMENTED: GET SCAN
        function scan = getScan(this)
            scan = AbsBlock_Sep16_30deg();
        end
        
    end
    
end

