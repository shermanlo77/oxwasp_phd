classdef TitaniumBlock_Dec16_120deg < Scan

    properties
    end
    
    methods
        
        function this = TitaniumBlock_Dec16_120deg()
            this@Scan('data/titaniumBlock_SnFilter_Dec16/scans/phantom_120deg/', '120deg_', 2000, 2000, 20);
            
            this.addARTistFile('data/titaniumBlock_SnFilter_Dec16/sim/phantom/120deg.tif');
            
            this.reference_scan_array = ReferenceArrayGetter.getReferenceScanArray_Dec16();
        end
        
    end
    
end

