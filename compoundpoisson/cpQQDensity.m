%FUNCTION: COMPOUND POISSON QQ PLOT FOR DENSITIES
%Plots histograms of simulations and pdf (exact or approximate)
%Plots qq plot, simulated quantiles (p) vs theoretical quantiles (p) for an array of p
%PARAMETERS:
    %compound_poisson: CompoundPoisson object, contains simulated data
    %histo_file: directory to save histogram figure
    %qq_file: directory to save the qq figure
function cpQQDensity(compound_poisson, histo_file, qq_file)

    %number of points when plotting densities
    n_linspace = 500;
    %number of trapeziums to be used
    n_trapezium = 10000;

    %% SET UP

    %get data from compound poisson object
    X = compound_poisson.X;
    
    %get the index of data points with the value 0
    n = compound_poisson.n;
    zero_index = (X==0);
    %get the number of zeros
    n_0 = sum(zero_index);
    %get non-zero data
    X_no_0 = X(~zero_index);

    %get an array of xs, equally spaced out across the range of the observed simulated data
    x_array = linspace(min(X),max(X),n_linspace);
    %if the minimum of the data is 0, remove it
    if x_array(1)==0
        x_array(1) = [];
    end
    
    %get the pdf and mass at 0
    pdf_range = compound_poisson.getPdf(x_array);
    if compound_poisson.can_support_zero_mass
        p_0 = compound_poisson.getPdf(0);
    end

    %% DENSITY
    
    fig = figure_latexSub;
    
    %plot the histogram (frequency density axis on the left)
    %do not include zeros if support zero mass
    if compound_poisson.can_support_zero_mass
        if (n_0~=0)
            yyaxis left;
        end
        h = histogram(X_no_0,'Normalization','CountDensity');
    else
        h = histogram(X,'Normalization','CountDensity');
    end
    hold on;
    %plot the pdf
    plot(x_array,pdf_range*(n),'r-');
    ylabel('freq. density');
    
    %if this density can support zero mass and there are zeros
    if compound_poisson.can_support_zero_mass && (n_0~=0)
        %plot the zero frequency for both the simulation and the pmf (frequency axis on the right)
        yyaxis right;
        %plot the frequency in the first bin of the histogram
        scatter(h.BinWidth/2,n_0,50,'filled','b'); %plot the simulation as a circle
        scatter(h.BinWidth/4,n*p_0,50,'filled','r'); %plot the pmf as a circle
        plot([h.BinWidth/2,h.BinWidth/2],[0,n_0],'b'); %simulation line
        plot([h.BinWidth/4,h.BinWidth/4],[0,n*p_0],'r'); %pmf line
        ylabel('frequency');
    end

    %set the size of the x axis and label
    xlim([min(X),max(X)]);
    xlabel('support');
    if compound_poisson.can_support_zero_mass && (n_0~=0)
        legend('Simulation freq. density','Theoretical freq. density','Simulation freq.','Theoretical freq.');
    else
        legend('Simulation freq. density','Theoretical freq. density');
    end

    %save figure
    saveas(fig,histo_file,'png');
    
    %% QQPLOT
    
    fig = figure_latexSub;
    
    %get array of percentages
    p = ((1:n)'-0.5)/n;
    
    %x_min is the lower limit for numerical integration to get the cdf
    %if the density cannot support zero and there are zeros
    if ( (~compound_poisson.can_support_zero_mass) && (n_0~=0) )
        %x_min is between 0 and min(X_no_0), weighted averaged
        x_min = exp(-compound_poisson.lambda) * min(X_no_0) * 0.1 / (exp(-compound_poisson.lambda) * (1+compound_poisson.lambda));
    %else...
    else
        x_min = min(X);
    end
    %get the inverse cdf for the array of percentages
    x_theoretical = compound_poisson.getInvCdf(p, x_min, max(X), n_trapezium);
    
    %plot the simulated quantiles vs the exact quantiles
    scatter(x_theoretical,sort(X),'x');
    hold on;
    %plot straight line
    plot(x_theoretical,x_theoretical);
    xlabel('Theoretical quantiles');
    ylabel('Simulation quantiles');
    xlim([x_theoretical(1),x_theoretical(end)]);
    
    %save figure
    saveas(fig,qq_file,'png');
    
    %% CDF GRAPH
    %DEBUGGING PURPOSES
%     figure;
%     plot(x_theoretical,p);
    
end

