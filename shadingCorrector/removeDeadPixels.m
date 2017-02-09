function [new_image] = removeDeadPixels(image)

    %REMOVE DEAD PIXELS
    %Got a given image, replace nan with the median of its 8 neighbour
    %pixels (removing nan), or stay nan if all of its 8 neighbours are nan.
    %This is repeated until the image has no nan values
    %PARAMETERS:
        %image: matrix of greyvalues
    %RETURN:
        %image: image with nan removed

    %get the size of the image
    [size_dim_1, size_dim_2] = size(image);

    %get the pixels which are nan
    nan_index = isnan(image);
    %get the number of nan
    n_nan = sum(sum(nan_index));
    
    %while the image has nan
    while n_nan > 0
        
        %make a copy of the image with nan
        new_image = image;
    
        %get the y and x coordinate of the nan
        [nan_index_y, nan_index_x] = find(nan_index);

        %for each nan value
        for i = 1:n_nan
            
            %declare an array of neighbour pixel's greyvalues
            neighbour_greyvalue_array = nan(8,1);
            %declare a point for that array
            neighbour_pointer = 1;
            
            %for each y displacement
            for i_y = -1:1
                %for each x displacement
                for i_x = -1:1
                    
                    %if the displacement is not zero
                    if ~( (i_y==0) && (i_x==0) )
                    
                        %work out the y coordinate of the neighbour
                        y_cood = nan_index_y(i) + i_y;
                        %work out the x coordinate of the neighbour
                        x_cood = nan_index_x(i) + i_x;

                        %if the x and y coordinate are within the boundary of the image
                        if( all([y_cood > 0, y_cood <= size_dim_1, x_cood > 0, x_cood <= size_dim_2]) )
                            %save that neighbou pixel greyvalue in the array
                            neighbour_greyvalue_array(neighbour_pointer) = image(y_cood, x_cood);
                            %increment the pointer
                            neighbour_pointer = neighbour_pointer + 1;
                        end
                        
                    end
                end
            end

            %on the new image, replace the nan pixel with the median, ignoring nan
            new_image(nan_index_y(i),nan_index_x(i)) = nanmedian(neighbour_greyvalue_array);

        end
        
        %replace the old image with the new one
        image = new_image;
        
        %get the pixels which are nan
        nan_index = isnan(image);
        %get the number of nan
        n_nan = sum(sum(nan_index));
        
    end

end
