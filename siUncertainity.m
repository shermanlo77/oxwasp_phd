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

  %E is the exponent of value
  E = floor(log10(abs(value)));

  %get the mantissa of the value
  err = err * 10^-E;
  %get the 'mantissa' of the error
  value = value * 10^-E;
  
  %round the error using to nTerm significant figures
  errE = floor(log10(abs(err))); %get the exponent of the 'mantissa' of the error
  err = err * 10^(-(errE - nTerm + 1)); %put nTerm figures before the decimal place
  err = round(err); %round it
  err = num2str(err); %convert to string
  if ( (numel(err) - nTerm) == 1 ) %99.9 rounds to 100 for example so adjust accordingly
    err(end) = [];
    errE = errE + 1;
  end
  
  %round the value according to the number of decimal places in the error
  value = value * 10^(-(errE - nTerm + 1));
  value = round(value);
  value = num2str(value);
  value = strcat(value(1),'.',value(2:end));
  
  %to be used with \SI[separate-uncertainty = true] in latex
  quote = strcat(value,'(',err,')E',num2str(E));
  if (isNegative)
    quote = strcat('-',quote);
  end
  quote = strcat('\\num[separate-uncertainty = true]{',quote,'}');
end
