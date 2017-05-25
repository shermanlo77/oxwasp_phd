classdef BGW_140316 < Scan
    
    properties
        n_white;
        n_black;
        n_grey;
        
    end
    
    methods
        
        function this = BGW_140316()
            this@Scan([], [], 1996, 1996, 100);
            %assign member variables
            this.panel_counter = PanelCounter_Brass();
            
            this.n_white = 20;
            this.n_black = 20;
            this.n_grey = 20;
            
            this.min_greyvalue = 0;
            
            reference_image_array(3) = Scan();
            this.reference_image_array = reference_image_array;
            this.reference_image_array(1) = Scan('data/140316_bgw/black/', 'black_140316_', this.width, this.height, this.n_black);
            this.reference_image_array(2) = Scan('data/140316_bgw/white/', 'white_140316_', this.width, this.height, this.n_white);
            this.reference_image_array(3) = Scan('data/140316_bgw/grey/', 'grey_140316_', this.width, this.height, this.n_grey);
        end
    end
    
end

