function [std_array] = shadingCorrection_ANOVA(data_object, n_train, shading_corrector_class, want_grey, parameters, n_repeat)
%SHADINGCORRECTION_ANOVA Does shading correction on the b/g/w images, uses
%ANOVA to analyse the variance
    %Use n_train b/g/w images to train the shading corrector, then uses the
    %reminder of b/g/w images to do shading correction on. The variance
    %between and within pixel is recorded. Repeated n_repeat times
%PARAMETERS:
    %data_object: object which loads the data
    %n_train: number of images to be used for training the shading corrector
    %shading_corrector_class: function handle which will be used for instantiating a new shading corrector
    %want_grey: boolean, true to use grey images for training the shading corrector
    %parameters: nan or vector of parameters for smoothing in shading correction
    %n_repeat: number of times to repeat the experiment
%RETURN:
    %std_array: n_repeat x 2 x 3 matrix containing variances, 1st column
    %for within pixel variance, 2nd column for between pixel variance, one
    %for each colour (b/g/w)
    %plots:
        %mean shading corrected image
        %box plot of std_array

    %declare array for storing variances
    std_array = zeros(n_repeat,2,3);

    %for n_repeat times
    for i_repeat = 1:n_repeat
        
        %get the training and test black images index
        index = randperm(data_object.n_black);
        black_train = index(1:n_train);
        black_test = index((n_train+1):end);

        %get the training and test white images index
        index = randperm(data_object.n_white);
        white_train = index(1:n_train);
        white_test = index((n_train+1):end);

        %get the training and test grey images index
        index = randperm(data_object.n_grey);
        grey_train = index(1:n_train);
        grey_test = index((n_train+1):end);
 
        %turn off shading correction when loading the b/g/w images
        data_object.turnOffShadingCorrection();

        %declare array of images, reference stack is an array of mean b/g/w images
        reference_stack = zeros(data_object.height,data_object.width,2+want_grey);
        %load mean b/w images
        reference_stack(:,:,1) = mean(data_object.loadBlackStack(black_train),3);
        reference_stack(:,:,2) = mean(data_object.loadWhiteStack(white_train),3);
        %load mean grey images if requested
        if want_grey
            reference_stack(:,:,3) = mean(data_object.loadGreyStack(grey_train),3);
        end

        %instantise shading corrector using provided reference stack
        shading_corrector = feval(shading_corrector_class,reference_stack);

        %if parameters are provided, add it to the shading corrector
        %then add the shading corrector to the data
        if ~isnan(parameters)
            data_object.addManualShadingCorrector(shading_corrector,parameters);
        else
            data_object.addManualShadingCorrector(shading_corrector);
        end

        %test_stack_array if a collection of array of b/g/w images
        test_stack_array = cell(1,3); %one array for each colour

        %load the test b/g/w images as an array and save it to test_stack_array
        test_stack_array{1} = data_object.loadBlackStack(black_test);
        test_stack_array{2} = data_object.loadWhiteStack(white_test);
        test_stack_array{3} = data_object.loadGreyStack(grey_test);

        %for each colour b/g/w test images
        for i_ref = 1:3

            %get the mean shading corrected image
            mean_image = mean(test_stack_array{i_ref},3);
            %if this is the first run, plot the mean shading corrected image
            if i_repeat == 1
                figure;
                imagesc_truncate(mean_image);
            end

            %remove dead pixels from the mean image
            mean_image = removeDeadPixels(mean_image);
            %get the mean of all greyvalues in the mean image
            mean_all = mean(reshape(mean_image,[],1));
            
            %get the number of test images
            n_test = size(test_stack_array{i_ref},3);
            %get the number of pixels in each image
            n_pixel = data_object.area;

            %for each test image
            for i_image = 1:n_test
                %remove dead pixels
                test_stack_array{i_ref}(:,:,i_image) = removeDeadPixels(test_stack_array{i_ref}(:,:,i_image));
            end

            %save the within pixel variance
            std_array(i_repeat,1,i_ref) = sum(sum(sum( ( test_stack_array{i_ref} - repmat(mean_image,1,1,n_test) ).^2 ))) / (n_pixel*n_test - data_object.area);
            %save the between pixel variance
            std_array(i_repeat,2,i_ref) = sum(sum((mean_image - mean_all).^2))/(data_object.area-1);

        end

    end
    
    %for each colour b/g/w
    for i_ref = 1:3
        %box plot the within and between pixel variance
        figure;
        boxplot(std_array(:,:,i_ref),{'within pixel','between pixel'});
        ylabel('variance');
    end

end

