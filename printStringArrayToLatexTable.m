%FUNCTION PRINT STRAING ARRAY TO LATEX TABLE
%Print the string array into latex format to a file
%PARAMETERS:
    %string_array: 2 dimensional array of strings
    %file_name: the file to print the latex table onto
function printStringArrayToLatexTable(string_array, file_name)

    %get the size of the string array
    [n_row, n_column] = size(string_array);
    
    %start the latex code with a blank
    latex_code = '';
    
    %for each row
    for i_row = 1:n_row
        
        %for each column
        for i_column = 1:n_column
            
            %append the string in the array to the latex table
            latex_code = [latex_code, char(string_array(i_row,i_column))];
            
            %if this is not the last column, append '&'
            if i_column ~= n_column
                latex_code = strcat(latex_code, '&');
            end
            
        end
        
        %if this is not the last row
        if i_row ~= n_row
            %append double back slash
            latex_code = [latex_code, '\\ '];
        end
        
        %if this is the first row, append \hline
        if i_row == 1
            latex_code = [latex_code, '\hline '];
        end
    end

    %print the latex code to the file
    file = fopen(fullfile(cd,file_name),'w');
    fprintf(file,'%s', latex_code);
    fclose(file);

end

