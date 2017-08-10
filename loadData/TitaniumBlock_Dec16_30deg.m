classdef TitaniumBlock_Dec16_30deg < TitaniumBlock_Dec16

    properties
    end
    
    methods
        
        function this = TitaniumBlock_Dec16_30deg()
            this@TitaniumBlock_Dec16('data/titaniumBlock_SnFilter_Dec16/scans/phantom_30deg/', '30deg_');
            this.addARTistFile('data/titaniumBlock_SnFilter_Dec16/sim/phantom/30deg.tif');
        end
        
    end
    
end

