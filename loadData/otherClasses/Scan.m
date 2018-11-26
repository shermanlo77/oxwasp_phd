%SCAN Class for handling x-ray images
classdef Scan < matlab.mixin.Heterogeneous & handle

    %MEMBER VARIABLES
    properties
        width; %width of the image
        height; %height of the image
        area; %area of the image
        n_sample; %number of images
        file_name; %... name of sample image
        folder_location; %location of the dataset
        aRTist_file; %location of the aRTist simulation
        reference_scan_array; %array of reference scan objects (in ascending powers)
        reference_white; %integer pointing to the reference for white in the phantom
        
        voltage; %in units of kV
        power; %in units of W
        time_exposure; %in units of ms
        
        panel_counter %panel counter object
        shading_corrector; %shading corrector object
        want_shading_correction; %boolean, true to do shading correction, default false, automatically turns to true if a shading corrector is added
        want_remove_dead_pixels; %boolean, true to remove dead pixels, default false
        
        min_greyvalue; %minimum possible greyvalue
    end
    
    methods
        
        %CONSTRUCTOR
        %PARAMETERS:
            %folder_location: location of the images
            %file_name: name of the image files
            %width: width of the image
            %height: height of the image
            %n_sample: number of images
            %voltage: in units of kV
            %power: in units of W
            %time_exposure: in units of ms
        function this = Scan(folder_location, file_name, width, height, n_sample, voltage, power, time_exposure)
            %assign member variable if parameters are provided
            if nargin > 0
                this.folder_location = folder_location;
                this.file_name = file_name;
                this.width = width;
                this.height = height;
                this.area = width * height;
                this.n_sample = n_sample;
                
                this.voltage = voltage;
                this.power = power;
                this.time_exposure = time_exposure;
                
                this.want_shading_correction = false;
                this.want_remove_dead_pixels = false;
                this.min_greyvalue = 0;
            end
        end
        
        %LOAD IMAGE
        %Return a sample image
        %PARAMETERS
            %index: index of image (scalar)
        function slice = loadImage(this,index)
            slice = imread(strcat(this.folder_location,this.file_name,num2str(index),'.tif'));
            slice = this.shadingCorrect(double(slice));
        end
        
        %LOAD IMAGE STACK
        %Return stack of sample images
        %PARAMETERS:
            %range (optional): vector of indices of images requested
                %if empty return the full range
        function stack = loadImageStack(this,range)
            %if range not provided, provide the full range
            if nargin == 1
                range = 1:this.n_sample;
            end
            %declare stack of images
            stack = zeros(this.height,this.width,numel(range));
            %for each image, put it in the stack
            for index = 1:numel(range)
                stack(:,:,index) = this.loadImage(range(index));
            end
        end
        
        %SHADING CORRECTION
        %if want_shading_correction, does shading correction on the
        %provided image and returns it
        %if want_remove_dead_pixels, remove dead pixels
        function slice = shadingCorrect(this,slice)
            if this.want_shading_correction
                slice = this.shading_corrector.shadingCorrect(slice);
            end
            if this.want_remove_dead_pixels
                slice = removeDeadPixels(slice);
            end
        end
        
        %ADD DEFAULT SHADING CORRECTOR
        %Add bw shading corrector, using 0 power and the white power
        function addDefaultShadingCorrector(this)
            this.addShadingCorrector(ShadingCorrector(),[1,this.reference_white]);
        end
        
        %ADD LINEAR SHADING CORRECTION
        %Add linear shading corrector, using 0 W and all the powers till the white power
        function addLinearShadingCorrector(this)
            this.addShadingCorrector(ShadingCorrector(),1:this.reference_white);
        end
        
        %ADD SHADING CORRECTOR
        %Assign a shading corrector to the member variable and calibrate it for shading correction
        %The reference images used are determined by the parameter reference_index
        %PARAMETERS:
            %shading_corrector: ShadingCorrector object
            %reference_index: matrix of integers, representing image index (optional), zeros are ignored
                %dim 1: #
                %dim 2: for each reference_scan_array
                %for each column, (eg 1st column for black images)
                %the mean of black images specified by dim 1 are used for shading corrector
                %
                %if not provided, use the mean of all images, black, grey and white
        function addShadingCorrector(this,shading_corrector,reference_index,image_index)
            
            %turn off shading correction to obtain the reference images
            this.turnOffShadingCorrection();
            
            %count the number of reference scans
            n_reference = numel(reference_index);
            
            shading_corrector.initalise(n_reference, this.height, this.width);
            
            %for each reference scan
            for i = 1:n_reference
                %if reference_index is not provided, take the mean of all images
                if nargin == 3
                    shading_corrector.addScan(this.reference_scan_array(reference_index(i)));
                %else take the mean of all images specified in the ith column of reference_index
                else
                    shading_corrector.addScan(this.reference_scan_array(reference_index(i)),image_index(:,i));
                end
            end
            
            %add the shading corrector to the member variable
            this.addManualShadingCorrector(shading_corrector);
            
        end
        
        %ADD MANUAL SHADING CORRECTOR
        %Assign a provided shading corrector to the member variable and calibrate it
        %for shading correction
        %PARAMETERS:
            %shading_corrector: shading_corrector object
        function addManualShadingCorrector(this,shading_corrector)
            %assign the provided shading corrector to the member variable
            this.shading_corrector = shading_corrector;

            %get the minimum possible greyvalue to the shading corrector
            this.shading_corrector.min_greyvalue = this.min_greyvalue;
            
            %if the shading corrector can smooth the reference images panel by panel
                %add the panel counter
            if this.shading_corrector.can_smooth
                this.shading_corrector.addPanelCounter(this.panel_counter);
            end
            
            %calibrate the shading corrector to do shading correction
            this.shading_corrector.calibrate();
            %set shading correction to be on
            this.turnOnShadingCorrection();
            
            %add the shading corrector to each reference scan in reference_scan_array
            for i = 1:numel(this.reference_scan_array)
                this.reference_scan_array(i).shading_corrector = this.shading_corrector;
            end
        end
        
        %TURN ON SHADING CORRECTION
        %Set the member variable want_shading_correction to be true
        function turnOnShadingCorrection(this)
            this.want_shading_correction = true;
            for i = 1:numel(this.reference_scan_array)
                this.reference_scan_array(i).turnOnShadingCorrection();
            end
        end
        
        %TURN OFF SHADING CORRECTION
        %Set the memebr variable want_shading_correction to be false
        function turnOffShadingCorrection(this)
            this.want_shading_correction = false;
            for i = 1:numel(this.reference_scan_array)
                this.reference_scan_array(i).turnOffShadingCorrection();
            end
        end
        
        %TURN ON REMOVE DEAD PIXELS
        function turnOnRemoveDeadPixels(this)
            this.want_remove_dead_pixels = true;
            for i = 1:numel(this.reference_scan_array)
                this.reference_scan_array(i).turnOnRemoveDeadPixels();
            end
        end
        
        %TURN OFF REMOVE DEAD PIXELS
        function turnOffRemoveDeadPixels(this)
            this.want_remove_dead_pixels = false;
            for i = 1:numel(this.reference_scan_array)
                this.reference_scan_array(i).turnOffRemoveDeadPixels();
            end
        end
        
        %TURN ON SET EXTREME TO NAN
        %Set the shading corrector to set extreme greyvalues to be NaN
        function turnOnSetExtremeToNan(this)
            this.shading_corrector.turnOnSetExtremeToNan();   
        end
        
        %TURN OFF SET EXTREME TO NAN
        %Set the shading corrector to keep extreme greyvalues
        function turnOffSetExtremeToNan(this)
            this.shading_corrector.turnOffSetExtremeToNan();
        end
        
        %ADD ARTIST FILE
        function addARTistFile(this,aRTist_file)
            this.aRTist_file = aRTist_file;
        end
        
        %GET ARTIST IMAGE
        function slice = getARTistImage(this)
            slice = double(imread(this.aRTist_file));
        end
        
        %GET N REFERENCE
        function n_reference = getNReference(this)
            n_reference = numel(this.reference_scan_array);
        end
        
        %GET POWER ARRAY
        %Return array of powers for each reference scan
        function power_array = getPowerArray(this)
            %get the number of reference scans
            n_reference = numel(this.reference_scan_array);
            %declare array of powers
            power_array = zeros(1,n_reference);
            %for each reference scan, get the power and save it
            for i = 1:n_reference
                power_array(i) = this.reference_scan_array(i).power;
            end
        end
        
        %GET SEGMENTATION
        %Returns a binary image, true values represent ROI
        function segmentation = getSegmentation(this)
          
          %get the roi
          roiPath = this.getRoiPath;
          opener = ij.io.Opener();
          roi = opener.openRoi(roiPath);
          
          %get the coordinates of the roi
          %note: matlab starts at 1, java starts at 0
          roiRectangle = roi.getBounds();
          x = roiRectangle.x + 1;
          y = roiRectangle.y + 1;
          
          %get the mask of the roi, this returns an imageProcessor which represent the roi with a
              %non-zero value, this image is also cropped
          mask = roi.getMask();
          %copy the values from java to matlab
          roiMask = zeros(mask.getWidth(), mask.getHeight());
          roiMask(1:end) = mask.getPixels();
          roiMask = logical(-roiMask');
          
          %copy the pixels from the mask to the segmentation matrix at the roi coordinates
          segmentation = false(this.height, this.width);
          segmentation(y:(y+mask.getHeight()-1), x:(x+mask.getWidth()-1)) = roiMask;
        end
        
        %METHOD: GET ROI PATH
        %Returns the path of the region of interst file 
        function roiPath = getRoiPath(this)
          roiPath = strcat(this.folder_location,'segmentation.roi');
        end
        
        %GET SHADING CORRECTED ARTIST IMAGE
        %Returns the aRTist image, shading corrected
        %Uses aRTist simulations of the references except for the black image
        %PARAMETERS:
            %shading_corrector: newly instantised shading corrector
            %reference_index: integer vector, pointing to which reference images to use
        %RETURN:
            %slice: shading corrected aRTist image
        function slice = getShadingCorrectedARTistImage(this, shading_corrector, reference_index)
            %get the folder location and file name of the artist image
            [artist_location,artist_name,~] = fileparts(this.aRTist_file);
            artist_location = strcat(artist_location,'/');
            %instantise a Scan object containing the aRTist image
            aRTist = Scan_Single(artist_location, artist_name, this.width, this.height, this.voltage, this.power, this.time_exposure);
            %instantise an array of Scan objects, storing aRTist reference images
            %store the array in the aRTist member variable reference_scan_array
            artist_reference_array(this.getNReference()) = Scan();
            aRTist.reference_scan_array = artist_reference_array;
            
            reference_scan = this.reference_scan_array(1);
            greyvalue = mean(reshape(reference_scan.loadImageStack(),[],1));
            aRTist.reference_scan_array(1) = Scan_SingleFlat(this.width, this.height, this.voltage, reference_scan.power, this.time_exposure, greyvalue);
            
            %for each reference scan, except for black
            for i = 2:this.getNReference()
                %get the reference scan
                reference_scan = this.reference_scan_array(i);
                %get the file location and file name of the aRTist reference image
                [artist_location,artist_name,~] = fileparts(reference_scan.aRTist_file);
                artist_location = strcat(artist_location,'/');
                %instantise a Scan object for that aRTist reference image
                aRTist.reference_scan_array(i) = Scan_Single(artist_location, artist_name, this.width, this.height, this.voltage, reference_scan.power, this.time_exposure);
            end
            
            aRTist.reference_white = this.reference_white;
            %add shading correction and get the shading corrected aRTist image
            aRTist.addShadingCorrector(shading_corrector,reference_index);
            slice = aRTist.loadImage();
        end
        
    end
    
end

