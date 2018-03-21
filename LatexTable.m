%CLASS: LATEXTABLE For outputting  a table of x +/- err type data into a latex table
classdef LatexTable < handle
    
    %MEMBER VARIABLES
    properties (SetAccess = protected)
        mean_array; %matrix of mean values to be printed on a table
        error_array; %matrix of error values to be printed on a table
        n_row; %number of rows in the table (not including labels)
        n_col; %number of columns in the table (not including labels)
        row_label; %cell array of strings, labelling each row
        col_label; %cell array of strings, labelling each column
        n_sig; %number of significant figures used for the errors
        n_dec_for_no_error; %number of decimial places for values with no errors
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %mean_array: matrix of mean values to be printed on a table
            %error_array: matrix of error values to be printed on a table
            %row_label: cell array of strings, labelling each row
            %col_label: cell array of strings, labelling each column
        function this = LatexTable(mean_array, error_array, row_label, col_label)
            %assign member variables
            this.mean_array = mean_array;
            this.error_array = error_array;
            this.row_label = row_label;
            this.col_label = col_label;
            [n_row,n_col] = size(mean_array);
            this.n_row = n_row;
            this.n_col = n_col;
            %assign default values for the member variables
            this.n_sig = 1;
            this.n_dec_for_no_error = 3;
        end
        
        %METHOD: SET N DECIMIAL
        %Set the number of significant figures used for the errors
        %PARAMETERS:
            %n_sig
        function setNSig(this, n_sig)
            this.n_sig = n_sig;
        end
        
        %METHOD: SET N DECIMAL FOR NO ERROR
        %Set the number of decimial places for values with no errors
        %PARAMETERS:
            %n_decimal
        function setNDecimalForNoError(this, n_decimal)
            this.n_dec_for_no_error = n_decimal;
        end
        
        %METHOD: PRINT
        %Output the table of values into a latex table to a specified file
        %PARAMETERS:
            %file_name: the file to print the latex table to
        function print(this, file_name)
            
            %declare a string cell array, each cell represent a latex table cell
            string_array = cell(this.n_row+1, this.n_col+1);
            %top left corner is blank
            string_array{1} = '';
            %for each row
            for i_row = 1:this.n_row
                %put the row label into string_array
                string_array(i_row+1,1) = this.row_label(i_row);
            end
            %for each column
            for i_col = 1:this.n_col
                %put the column label into string array
                string_array(1,i_col+1) = this.col_label(i_col);
            end
            
            %for each column
            for i_col = 1:this.n_col
                %for each row
                for i_row = 1:this.n_row
                    %get the x +/- error as a text and save it to string_array
                    string_array{i_row+1, i_col+1} = this.quoteStdError(i_row,i_col);
                end
            end
            
            %convert string cell array to latex code and write the latex code to file_name
            this.printStringArrayToLatex(string_array, file_name);
            
        end
        
        %METHOD: QUOTE STD ERROR
        %Returns the mean and std error bar as a string in the form of (value +/- error) x 10 ^ E
        %PARAMETERS:
            %i_row: the row index of the array mean_array and error_array
            %i_col: the column index of the array mean_array and error_array
        %RETURN:
            %quote: string
        function quote = quoteStdError(this, i_row, i_col)

            %get the numerical value of the value and error
            value = this.mean_array(i_row,i_col);
            err = this.error_array(i_row,i_col);
            
            %E is the exponent of value
            E = floor(log10(abs(value)));

            %get the mantissa of the value and error
            err = err * 10^-E;
            value = value * 10^-E;

            %instanise a boolean, false if the error is not 0
            has_no_error = false;
            %get the number of decimial places of the least significant figure of the error
                %-floor(log10(err)) gets the exponent of the errors
                %error_sig_fig - 1 increases the number of decimial places according to the number of significant figures of the error
            dec_places = -floor(log10(err)) + this.n_sig - 1;
            
            %if it is less or equal to 0, set significant figures to 1
            if dec_places <= 0
                sig_fig = 1;
                dec_places = 0;
                
            %if the number of decimial places if infinite
            elseif isinf(dec_places)
                %then set the decimial places to be this.n_dec_for_no_error
                %set the boolean flat has_no_error to be true
                dec_places = this.n_dec_for_no_error;
                sig_fig = dec_places + 1;
                has_no_error = true;
                
            %else, set the number of signifiant figures to the number of decimial places add 1
                %add one for the digit to the left of the decimial place
            else
                sig_fig = dec_places + 1;
            end

            %round the value to the correct number of significant figures
            value = round(value,sig_fig,'significant');
            
            %if the error is zero, set the string of error to be 0 with a decimial place
            if has_no_error
                err = '0.';
            %else...
            else
                %round the error using dec_places number of decimial places
                err = round(err,dec_places,'decimals');
                %convert the error to string
                err = num2str(err);
            end

            %fill in missing decimial places with zeros
            if dec_places ~= 0
                while numel(err)<2+dec_places
                    err = [err,'0'];
                end
            end

            %convert the exponent to string
            E = num2str(E);

            %convert the value to string
            if sig_fig == 1
                value = num2str(value);
            else
                value = num2str(value * 10^(sig_fig-1));
                value = strcat(value(1),'.',value(2:end));
            end

            %export the quote value as a string
            if E == '0'
                quote = strcat('$',value,'\pm',err,'$');
            else %put brackets around the value in scientific notation
                quote = strcat('$(',value,'\pm',err,')\times 10^{',E,'}$');
            end

        end
        
        %METHOD: PRINT STRING ARRAY TO LATEX
        %Given a cell array of strings, convert it to latex code and print it to a specified file
        %PARAMETERS:
            %string_array: 2 dimensional array of strings
            %file_name: the file to print the latex table onto
        function printStringArrayToLatex(this, string_array, file_name)

            %start the latex code with a tabular
            latex_code = '\begin{tabular}{c';
            %get the verticle lines
            for i_col = 1:this.n_col
                latex_code = [latex_code,'|c'];
            end
            latex_code = [latex_code,'}'];

            %for each row
            for i_row = 1:(this.n_row+1)

                %for each column
                for i_column = 1:(this.n_col+1)

                    %append the string in the array to the latex table
                    latex_code = [latex_code, char(string_array(i_row,i_column))];

                    %if this is not the last column, append '&'
                    if i_column ~= (this.n_col+1)
                        latex_code = strcat(latex_code, '&');
                    end

                end

                %if this is not the last row
                if i_row ~= (this.n_row+1)
                    %append double back slash
                    latex_code = [latex_code, '\\ '];
                end

                %if this is the first row, append \hline
                if i_row == 1
                    latex_code = [latex_code, '\hline '];
                end
            end
            
            %end the tabular
            latex_code = [latex_code, '\end{tabular}'];

            %print the latex code to the file
            file = fopen(fullfile(cd,file_name),'w');
            fprintf(file,'%s', latex_code);
            fclose(file);

        end
    end
    
end

