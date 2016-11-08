%FUNCTION: LOAD THE DATA
%PARAMETER:
    %folder_location: string containing the location of the file
function [stack,length,area,n] = load_stack(folder_location)

    %check if the parameter is a string, if not display error message
    if ~ischar(folder_location)
        error('Error in load_stack(folder_location), folder_location is not a string');
    end

    length = 1996; %length of the images
    area = length^2; %area of the images
    n = 100; %number of images
    
    %array of matrices
        %dim1: rows
        %dim2: columns
        %dim3: for each image
    stack = zeros(length,length,n);
    
    %try...
    try
        %for each image, save the pixel grey values
        for i = 1:n
            slice = imread(strcat(folder_location,'/block_',num2str(i),'.tif'));
            stack(:,:,i) = slice;
        end
    %else there is something wrong opening the file, display error message
    catch
        error('Error in load_stack(folder_location), cannot find %s/block_###.tif',folder_location);
    end

end

