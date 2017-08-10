classdef TitaniumBlock_Dec16 < Scan
    
    properties
    end
    
    methods
        
        function this = TitaniumBlock_Dec16(folder_location, file_name)
            this@Scan(folder_location, file_name, 2000, 2000, 20, 190, 19.95, 1415);
            
            reference_scan_array(4) = Scan();
            this.reference_scan_array = reference_scan_array;
            this.reference_scan_array(1) = Scan('data/titaniumBlock_SnFilter_Dec16/scans/blank_0W/', '0w_', this.width, this.height, 20, 0, 0, this.time_exposure);
            this.reference_scan_array(2) = Scan('data/titaniumBlock_SnFilter_Dec16/scans/blank_10W/', '10w_', this.width, this.height, 20, this.voltage, 10.07, this.time_exposure);
            this.reference_scan_array(3) = Scan('data/titaniumBlock_SnFilter_Dec16/scans/blank_20W/', '20w_', this.width, this.height, 20, this.voltage, 19.95, this.time_exposure);
            this.reference_scan_array(4) = Scan('data/titaniumBlock_SnFilter_Dec16/scans/blank_26W/', '26w_', this.width, this.height, 20, this.voltage, 26.03, this.time_exposure);

            this.reference_scan_array(2).addARTistFile('data/titaniumBlock_SnFilter_Dec16/sim/blank/10w.tif');
            this.reference_scan_array(3).addARTistFile('data/titaniumBlock_SnFilter_Dec16/sim/blank/20w.tif');
            this.reference_scan_array(4).addARTistFile('data/titaniumBlock_SnFilter_Dec16/sim/blank/26w.tif');
            
            this.reference_white = 3;
            
        end
        
    end
    
end

