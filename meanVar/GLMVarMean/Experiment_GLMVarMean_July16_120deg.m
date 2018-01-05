classdef Experiment_GLMVarMean_July16_120deg < Experiment_GLMVarMean_July16
    
    properties
    end
    
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = Experiment_GLMVarMean_July16_120deg()
            %call superclass with experiment name
            this@Experiment_GLMVarMean_July16('GLMVarMean_July16_120deg');
        end
        
    end
    
    methods (Access = protected)
        
        %OVERRIDE: SET UP EXPERIMENT
        function setup(this)
            %call superclass with 100 repeats and a random stream
            this.setup@Experiment_GLMVarMean_July16(RandStream('mt19937ar','Seed',uint32(1871182699)));
        end
        
        %IMPLEMENTED: GET SCAN
        function scan = getScan(this)
            scan = AbsBlock_July16_120deg();
        end

    end
    
end

