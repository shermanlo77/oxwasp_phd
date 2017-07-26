classdef TitaniumBlock_Dec16_30deg < Scan

    properties
    end
    
    methods
        
        function this = TitaniumBlock_Dec16_30deg()
            this@Scan('data/titaniumBlock_SnFilter_Dec16/scans/phantom_30deg/', '30deg_', 2000, 2000, 20);
            
            this.addARTistFile('data/titaniumBlock_SnFilter_Dec16/sim/phantom/30deg.tif');
            
            this.reference_scan_array = ReferenceArrayGetter.getReferenceScanArray_Dec16();
        end
        
    end
    
end

