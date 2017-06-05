%CLASS: NORMALITY TESTER (ABSTRACT)
    %Given multiple images, test pixel by pixel if the greyvalue are Normally distributed
    %Returns an image of each pixel p value
    %The test is done via the method doTest
    %
    %The method getPValue(this, grey_value_array) needs to be implemented
classdef NormalityTester < handle
    
    %MEMBER VARIABLES
    properties
        p_value; %p value of each pixel
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        function this = NormalityTester()
        end
        
        %METHOD: DO HYPOTHESIS TEST
        %Given images, conduct Normality test and modify the member variable p_value
        %PARAMETERS:
            %data_object: a Scan object containing b/g/w images
            %image_id: 0 for loadSample, 1,2,3 for loadReference (1 for black, 2 for white, 3 for grey)
            %index: image indecies in the stack to be used in the hypothesis test
                %e.g. [1,2,3,4,5] to use the 1st 5 images
        function doTest(this, data_object, image_id, index)
            
            %if image_id is 0, then load sample images
            if image_id == 0
                image_stack = data_object.loadSampleStack(index);
            %else load reference images
            else
                image_stack = data_object.loadReferenceStack(image_id,index);
            end
            
            %get the size of the images and the number of images
            [height, width, n_image] = size(image_stack);
            
            %declare a variable for containing p values for each pixel
            p_value_local = zeros(height, width);
            
            %for each column
            parfor x = 1:width
                %for each row
                for y = 1:height
                    %conduct the hypothesis test on this pixel
                    p_value_local(y,x) = this.getPValue(reshape(image_stack(y,x,:),[],1));
                end
            end
            
            %save the p value to the member variable
            this.p_value = p_value_local;
        end %doTest
        
    end %methods
    
    %ABSTRACT METHODS
    methods (Abstract)
        
        %METHOD: GET P VALUE
        %Given a vector of grey values, conduct a normality hypothesis test
        %PARAMETERS:
            %grey_value_array: vector of grey values
        %RETURN:
            %p: a single p value
        getPValue(this, grey_value_array)
        
    end %abstract methods
    
end %class

