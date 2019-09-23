%MIT License
%Copyright (c) 2019 Sherman Lo

%FUNCTION: SI UNCERTAINITY
%Convert value and error bar so that it can be used using \num[separate-uncertainty = true] in LaTeX
%
%PARAMETERS:
  %value: reported value
  %err: error bar
  %nTerm: number of significant figures for err
%NOTES:
  %Does not work with value = 0, err = 0 and nTerm = 0
  %Should be that value > err
function quote = siUncertainity(value, err, nTerm)
  
  if value < 0
    isNegative = true;
    value = -value;
  else
    isNegative = false;
  end
  
  %E is the exponent of value (and to be used for the final quote)
  E = floor(log10(abs(value)));
  %errE is the exponent of the error
  errE = floor(log10(abs(err)));
  
  %compare the exponent of the value and the error
  %if the error is bigger than the value, use the exponent of the error
  if errE > E
    E = errE;
    isErrBigger = true;
  else
    isErrBigger = false;
  end
  
  %get the mantissa of the value
  value = value * 10^-E;
  
  %get the 'mantissa' of the error
  err = err * 10^-E;
  
  %round the error using nTerm significant figures
  errE = floor(log10(abs(err))); %get the exponent of the 'mantissa' of the error
  err = err * 10^(-(errE - nTerm + 1)); %put nTerm figures before the decimal place
  err = round(err); %round it
  errString = num2str(err); %convert to string
  if ( (numel(errString) - nTerm) == 1 ) %eg 99.9 rounds to 100 for example so adjust accordingly
    errString(end) = [];
    errE = errE + 1;
  end
  
  %round the value according to the number of decimal places in the error
  value = value * 10^(-(errE - nTerm + 1));
  value = round(value);
  valueString = num2str(value);
  %if the error is one or more magnitude bigger than the value, leading zeros are required
  if (isErrBigger)
    leadingZero = char();
    for i = 1:(numel(errString) - numel(valueString))
      leadingZero(i) = '0';
    end
    valueString = strcat(leadingZero, valueString);
  end
  %if there are more than 1 value, then the mantissa needs a decimial place
  if (numel(valueString) > 1)
    valueString = strcat(valueString(1),'.',valueString(2:end));
  end
  
  %to be used with \SI[separate-uncertainty = true] in latex
  quote = strcat(valueString,'(',errString,')E',num2str(E));
  if (isNegative)
    quote = strcat('-',quote);
  end
  quote = strcat('\\num[separate-uncertainty = true]{',quote,'}');
end
