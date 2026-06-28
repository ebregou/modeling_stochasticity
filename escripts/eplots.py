# Purpose: plotting tools
# Author: Emily Bregou

# Standard packages
import matplotlib as mpl
from matplotlib import pyplot as plt
plt.style.use('/Users/eb35267/Desktop/code/mpl_style/estyle.mplstyle')
plt.style.use('/Users/eb35267/Desktop/code/mpl_style/notebook_style.mplstyle')
#plt.style.use('/Users/eb35267/Desktop/code/mpl_style/pres_style.mplstyle')
from matplotlib import ticker
from matplotlib import colors as mplcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox
from collections import OrderedDict
import numpy as np
import corner
import math
import zeus21
import pandas as pd
import dataframe_image as dfi
import emcee

# Local packages
from escripts import eMCMC
from escripts import cosmo_calc

color_cycler = mpl.rcParams['axes.prop_cycle']
colors = [c['color'] for c in color_cycler]
hex_colors = [mpl.colors.to_hex(color) for color in colors]
red = hex_colors[0]
turquoise = hex_colors[1]
yellow = hex_colors[2]
navy = hex_colors[3]
orange = hex_colors[4]
green = hex_colors[5]
periwinkle = hex_colors[6]
labels = ['red', 'turquoise', 'yellow', 'navy', 'orange', 'green', 'periwinkle']


def percent_diff(x, y0, y1, labels, other_data = None):
    """
    Make a plot showing 2 different datasets & their percent difference
    Inputs:
        x [1darray]: x values taken by both datasets
        y0 [1darray]: corresponding y values of the first dataset (this will be the dataset used for comparison)
        y1 [1darray]: corresponding y values of the second dataset
        labels [list]: list of labels
        other_data [list of 1darrays]: other data that will be plotted but whose percent difference will not be calculated
    Returns:
        fig, ax
    """
    percent_diff = np.abs(y1-y0) / y0

    fig, ax = plt.subplots(2, 1, sharex = True, height_ratios = (2, 1.5))
    plt.subplots_adjust(hspace = 0.1)

    ax[0].plot(x, y0, label = labels[0])
    ax[0].plot(x, y1, label = labels[1])
    if other_data is not None:
        assert len(other_data) > 0, 'other_data must be given as a list or array'
        [ax[0].plot(x, dat, label = lab, linestyle = 'dotted') for (dat, lab) in zip(other_data, labels[2:])]
        
    ax[0].legend()

    ax[1].plot(x, percent_diff, color = navy)
    ax[1].yaxis.set_major_formatter(ticker.FuncFormatter(custom_percent_formatter)) # Format percent difference with % symbols

    return fig

def make_corner(my_UVLF, backend_file = None, samples = None, true_vals = None, title = '', burn_in = None, include_params = None, 
                sample_color = red, truth_color = turquoise):
    """
    Plot a corner plot, adding true values if any are given.
    Inputs:
        [eMCMC UVLF object]: UVLF object used for its method for getting stored samples
        backend_file [str]: .h5 file containing previously saved chain
        samples [array]: samples from the MCMC chain
        labels [list of strs]: label for each sample
        true_vals [1darray]: optional true values for each parameter. Don't include true values for parameters in excluded_params
        title [str]: plot title
        burn_in [int]: number of samples to discard as burnin
        include_params [list]: use if you only want to plot certain parameters. Any parameter not listed will not be plotted
        sample_color [str]: color for the samples
        truth_color [str]: color to plot the best fit
    Returns:
        corner_plot [matplotlib figure]: corner plot showing the values and covariances of each parameter, and, optionally, their true values
    """
    
    # Get samples & parameter labels, excluding parameters that weren't fit
    if samples is None: # Use the stored samples if the user has not input any
        samples, _, _, labels = my_UVLF.get_fit(backend_file, exclude_unfit = True, burn_in = burn_in, include_params = include_params) 
    else: # Just get the names of the parameters if you've input samples
        fit_params = my_UVLF.param_data['fit'].values.astype(bool)
        if include_params is not None:
            include_mask = my_UVLF.param_data.index.isin(include_params)
            print(include_mask)
            include_samples = np.delete(include_mask, ~fit_params) # Samples will just not have the columns that weren't fit so we need to get the right dimensions
            print(include_samples)
            samples = samples[:,include_samples]
            if true_vals is not None:
                true_vals = np.array(true_vals)[include_samples]
            fit_params = fit_params & include_mask
        labels = my_UVLF.param_data['label'].iloc[fit_params].tolist()


    if true_vals is None:
        corner_plot = corner.corner(samples, labels=labels, color = sample_color, plot_contours = True, plot_datapoints = False, hist_kwargs={'linewidth': 2}, 
        label_kwargs={"fontsize": 15}, levels = (0.393, 0.864)) # Levels correspond to 1 & 2 sigma 
    else:
        corner_plot = corner.corner(samples, labels=labels, color = sample_color, truths = true_vals, 
                                    truth_color= truth_color, plot_contours = True, plot_datapoints = False, hist_kwargs={'linewidth': 2})

    corner_plot.suptitle(title, y = 1.02)
        
    return corner_plot

def evolving_UVLF_fit(my_UVLF, backend_file = None, z_plot = None, z_errs = None, MUV_dat = None, ylims = None, plot_from_chain = True, nsamples = 100, 
                      
                      comparison_fits = [], comparison_labels = [], 
                      ncols = 4, burn_in = None, show_chi2 = True, data_fmt = None, fit_colors = None):
    """
    Plot the UVLF at different redshifts
    Inputs:
        my_UVLF: eMCMC UVLF object
        backend_file [str]: .h5 file containing previously saved chain
        z_plot [list of floats]: redshifts to plot, or None if you want to plot all the redshifts for which there exists data
        z_errs [list of floats]: the uncertainty in redshift, or None if you want to use the uncertainty that corresponds to the data.
                                If you provide redshifts to plot but no redshift error, it will default to 0.5.
        MUV_dat [tuple of lists]: MUV bins & widths to plot. If None, defaults to the bins defined in my_UVLF.
        plot_from_chain [bool]: whether or not to plot the best fit and samples from the chain stored in my_UVLF. If False, only comparison_fits
                             parameter values will be plotted (so you can quickly check different fits this way).
        nsamples [int]: number of samples from the MCMC chain you want to appear in addition to the best fit
        comparison_fits [list of lists]: lists of parameter values that will be used to create comparison UVLFs
        comparison_labels [list of strs]: labels that correspond to the parameter values in comparison_fits, for the legend
        ncols [int]: number of columns to plot
        burn_in [int]: number of samples to discard as burnin. If None, the default value will be used (see UVLF.get_fit())
        show_chi2 [bool]: whether or not to show the chi squared of the fit(s). Does not apply if plot_from_chain is True
        data_fmt [list]: list of scatter plot styles
        fit_colors [list]: list of colors for the different models
    Outputs:
        Figure showing the UVLF at different redshfits
    """

    data_z = [dat[0] for dat in my_UVLF.data] # Get the redshifts that correspond to the input data

    if z_plot is None:
        z_plot = data_z
        z_errs = [dat[1] for dat in my_UVLF.data]
        dat = my_UVLF.data
    else:
        if z_errs is None:
            z_errs = np.full_like(z_plot, 0.5, dtype = float)
        if type(z_plot) == list:
            z_plot = np.array(z_plot)
        if type(z_errs) == list:
            z_errs = np.array(z_errs)

        idxs = np.concatenate([np.where(np.abs(data_z - z) < z_err)[0] for z, z_err in zip(z_plot, z_errs)])
        dat = [my_UVLF.data[i] for i in idxs]
    

    if len(comparison_fits) > 0:
        assert len(comparison_labels) > 0, 'If specifying comparison fits you must also give their labels'

    # Figure out how many subplots to make
    nz = len(z_plot)
    fig = plt.figure()
    gs = GridSpec(math.ceil((nz+1)/ncols), ncols, figure=fig)
    plt.subplots_adjust(wspace = 0, hspace = 0.3)
    
   
    # Get samples & best fit
    if plot_from_chain:
        samples, best_fit, _, _= my_UVLF.get_fit(backend_file = backend_file, exclude_unfit = True, burn_in = burn_in)
        print(best_fit)
    
    chi2_comparison = [-2*my_UVLF.log_like(fit, dat) for fit in comparison_fits]

    # Set the same y limits for all the plots
    # Flatten all ydat values from sorted_data
    all_ydat = np.concatenate([np.concatenate([ds[3] for ds in z_separated])  # ds[3] is ydat
                               for z_separated in my_UVLF.sorted_data])
    
    if ylims is None:
        ylo, yhi = np.log10(all_ydat.min()) - 1, np.log10(all_ydat.max()+3)
    else: 
        ylo, yhi = ylims


    # Flatten all xdat and xerr across all redshifts and datasets
    if MUV_dat is None:
        MUVcenters, MUVwidths = my_UVLF.get_all_MUV_bins()
    else:
        MUVcenters, MUVwidths = MUV_dat

    # Plotting details
    if data_fmt is None:
        data_fmt = ['o', 's', 'D', 'P']
    if fit_colors is None:
        fit_colors = [turquoise, yellow, orange, green]

    axs = []
    for i, (z, z_err) in enumerate(zip(z_plot, z_errs)): 

        ax = fig.add_subplot(gs[math.floor(i/ncols), i%ncols])
        axs.append(ax)

        if plot_from_chain:
            # Choose random samples from the MCMC chain
            inds = np.random.randint(len(samples), size=nsamples) # Choose nsamples from the chain

            # Plot each sample from the chain
            for ind in inds: 
                sample = samples[ind]
                ax.plot(MUVcenters, np.log10(my_UVLF.UVLF_wrapper(z,z_err, MUVcenters, MUVwidths,sample)), alpha=6/max(nsamples, 6), 
                           color = red, linestyle = '-', zorder = 0)
    
            # Plot best fit
            chi2 = -2* my_UVLF.log_like(best_fit, alt_data = dat)

            ax.plot(MUVcenters, np.log10(my_UVLF.UVLF_wrapper(z, z_err, MUVcenters, MUVwidths, best_fit)), color = red, linestyle = '-', 
                                           zorder = 0, label = fr'best fit, $\chi^2 = {chi2:.0f}$', lw = 5)
            
            ax.plot([0], [0], color = red, alpha = 0.1, label = f'{nsamples} samples from chain', linestyle = '-', lw = 5) # Create the label for the samples

        # Plot comparison fits
        for fit, chi2_fit, label, color, ls in zip(comparison_fits, chi2_comparison, comparison_labels, fit_colors, 
                                        ['solid', 'dashdot', 'dashed', 'dotted', 'solid']):
            if show_chi2:
                fit_label = fr'{label}, $\chi^2 = {chi2_fit:.0f}$'
            else:
                fit_label = f'{label}'
            ax.plot(MUVcenters, np.log10(my_UVLF.UVLF_wrapper(z,z_err,MUVcenters, MUVwidths,fit)), color = color, 
                       label = fit_label, linestyle = ls, zorder = 1, lw = 5)
        
        # Plot data
        all_dat_z = [dat[0][0] for dat in my_UVLF.sorted_data]
        idx = np.where(np.abs(all_dat_z - z) <= z_err)[0]
        if len(idx) > 0:
            for zbin in [my_UVLF.sorted_data[idx_i] for idx_i in idx]:
            #zbin = my_UVLF.sorted_data[idx[0]]

                for dat_z, fmt in zip(zbin, data_fmt): 
                    xdat, ydat, yerr_upper, yerr_lower, xerr, upper_lim_bool = dat_z[2], dat_z[3], dat_z[4], dat_z[5], dat_z[6], dat_z[7].astype('bool')
                    ax.vlines(xdat, np.clip(np.log10(ydat - yerr_lower), -100, 100), np.clip(np.log10(ydat + yerr_upper), -100, 100), ls = '-', colors = navy, linewidth =5)
                    
                    # Show upper limits with a unique plotting style
                    for x, y in zip(xdat[upper_lim_bool], ydat[upper_lim_bool]):
                        ax.vlines(x, np.log10(y)-0.75, np.log10(y), ls = '-', colors = navy, linewidth = 5, alpha = 0.5)
                        ax.scatter(x-0.015, np.log10(y)-0.75, marker = 'v', c = navy, s = 65) # down arrow

                    if dat_z[-1]: # This indicates whether the data was used in the MCMC likelihood or not
                        for x, y in zip(xdat[upper_lim_bool], ydat[upper_lim_bool]):
                            ax.scatter(x, np.log10(y), marker = fmt, label = dat_z[8], facecolors = 'white', edgecolors = navy, s = 125, 
                                zorder = 4, linewidth = 3)
                        for x, y in zip(xdat[np.invert(upper_lim_bool)], ydat[np.invert(upper_lim_bool)]): # Account for the fact that upper limits cannot currently be included in the MCMC likelihood
                            ax.scatter(x, np.log10(y), marker = fmt, label = dat_z[8], c = navy, s = 125)
                    else: 
                        ax.scatter(xdat, np.log10(ydat), marker = fmt, label = dat_z[8], facecolors = 'white', edgecolors = navy, s = 125, 
                                zorder = 4, linewidth = 3)

        ax.invert_xaxis() 
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # Make it so that only integers can be used in the  
                                                                                    # axis labels                                                                            
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # Make it so that only integers can be used in the  
                                                                                    # axis labels
        ax.set_title(fr'${round(z-z_err, 2)} \lesssim z \lesssim {round(z+z_err, 2)}$', fontsize = 20, pad = 8)


    # Text & labels
    ylabel = r'$\log_{10}(\phi_{\rm{UV}}$ [$\rm{mag}^{-1} \rm{Mpc}^{-3}$])'
    xlabel = r'$M_{\rm{UV}}$ [mag]'

    # Formatting, labels
    multicol(gs, axs, xlabel, ylabel, xlims = (np.max(MUVcenters)+0.25, np.min(MUVcenters)-0.25), ylims = (ylo, yhi), include_legend = True)
    
    return fig

def PMUV(my_UVLF, param_values, Mhtab = None, zs = None, z_errs = None, MUVcenters = None, MUVwidths = None, ncols = 3, logscale = False):
    """
    Plot P(MUV|Mh) for two different redshifts given a set of UVLF parameters
    Inputs:
        my_UVLF [eMCMC UVLF object]: UVLF object used for its methods such as applying time evolution to parameter values & wrapping them in the
                                        right format for zeus21
        param_values [1darray]: values of parameters to calculate MUV from Mh
        Mhtab [1darray]: log(Mh/M_odot) for plotting
        zs [1darray]: redshfits to plot (as separate subplots on a grid)
        z_errs [1darray]: redshift uncertainty (if None will default to 0.5)
        MUVcenters [1darray]: centers of MUV bins to plot
        MUVwidths [1darray]: widhts of MUV bins to plot. Defaults to 0.5 if none given
        ncols [int]: number of columns to plot the redshift bins over
        logscale [bool]: whether or not to use log scale for the y axis
    Outputs:
        fig, ax: Figure showing P(MUV|Mh) for different halo masses. Note that even without time-evolving parameters, the P(MUV|Mh) will look different at
        differnet redshifts due time-evolving mass accretion rates
    """

    if Mhtab is None:
        Mhtab = np.arange(9, 13.75, 0.75)

    if zs is None:
        zs = my_UVLF.zs
        z_errs = [dat[1] for dat in my_UVLF.data]
    if z_errs is None:
        z_errs = np.full_like(zs, 0.5, dtype = float)

    nz = len(zs)
    fig = plt.figure()
    gs = GridSpec(math.ceil(nz / ncols), ncols, figure=fig)
    plt.subplots_adjust(hspace = 0.3, wspace = 0)

    Mhtab_og = my_UVLF.HMFintclass.Mhtab # Save this to reset it later. There's probably a more elegant solution I could come up with at another time
    my_UVLF.HMFintclass.Mhtab = 10**Mhtab # Set the UVLF table of halo masses to the input value

    if MUVcenters is None: 
        # Get unique x values and their corresponding errors
        MUVcenters_lowres, MUVwidths_lowres = my_UVLF.get_all_MUV_bins() # Get coarse MUV binning
        MUVcenters, MUVwidths = np.linspace(MUVcenters_lowres.min(), MUVcenters_lowres.max(), 1000), np.full(1000, 0.5)
    elif MUVwidths is None:
        MUVwidths = np.full_like(MUVcenters, 0.5)

    if len(Mhtab) > 1:
        # Create gradient for color-coding curves based on corresponding halo mass
        cmap, sm, boundaries = create_custom_colorbar([periwinkle, red], Mhtab)
    
    axs = []
    all_weights = []
    for i, (z, z_err) in enumerate(zip(zs, z_errs)): # Plot curves for each redshift
        ax = fig.add_subplot(gs[math.floor(i/ncols), i%ncols])
        axs.append(ax)
        weights = my_UVLF.UVLF_wrapper(z,z_err,MUVcenters, MUVwidths, param_values, return_weights = True)

        for i, Mh in enumerate(Mhtab): # Plot different color-coded Gaussians for different halo masses
            if len(Mhtab) > 1:
                # Get color based on halo mass
                frac = (Mh - Mhtab.min()) / (Mhtab.max() - Mhtab.min()) 
                color = cmap(frac)
            else:
                color = periwinkle
            ax.plot(MUVcenters, weights[i], color = color, ls = 'solid')
            all_weights.append(weights[i])
        ax.set_title(fr'${round(z-z_err, 1)} \lesssim z \lesssim {round(z+z_err, 1)}$', fontsize = 20, pad = 8)
        ax.invert_xaxis()
        if logscale:
            ax.set_yscale('log')

    all_weights = np.concatenate(all_weights)

    # Color bar things:
    if len(Mhtab)>1:
        cbar = fig.colorbar(sm, ax = axs, boundaries=boundaries, ticks=Mhtab, aspect = 7*math.ceil(nz/ncols))
        cbar.set_label(r'$\log_{10}{M_h}$')

        


    xlabel = r'$M_{\rm{UV}}$ [mag]'
    ylabel = r'$p(M_{\rm{UV}}|M_h)$'
    multicol(gs, axs, xlabel, ylabel, ylims = (0.9*all_weights.min(), 1.1*all_weights.max()))
    multicol(gs, axs, ylims = (0.9*all_weights.min(), 1.1*all_weights.max()))

    my_UVLF.HMFintclass.Mhtab = Mhtab_og # Reset the Mhtab

    return fig 

def sfe_shape_diff_z(my_UVLF, param_values, zs = None, Mhtab = None, id_max = False):
    """
    Plot SFE vs. halo mass at different redshifts
    Inputs:
        my_UVLF [eMCMC UVLF object]: UVLF object used for its methods such as applying time evolution to parameter values & wrapping them in the
                                        right format for zeus21
        param_values [1darray]: values of parameters to calculate SFE
        zs [1darray]: redshifts to plot
        Mhtab [1darray]: log10(Mh) to plot over
        id_max [bool]: Whether or not to identify the maximum SFE at the highest and lowest redshift bin
    Returns:
        fig: A figure showing the SFE over the range of given halo masses & redshifts
    """

    if zs is None:
        zs = np.arange(4, 16, 2)

    if Mhtab is None: 
        Mhtab = np.linspace(9, 13, 200)

    fig = plt.figure(figsize=(6, 4.5))            
    ax_rect  = [0.10, 0.12, 0.78, 0.80]       # left, bottom, width, height (0-1)
    cax_rect = [0.895, 0.12, 0.04, 0.80]      # narrow right slot for cbar
    ax = fig.add_axes(ax_rect)

    if len(zs)>1:
        cmap, sm, boundaries = create_custom_colorbar([yellow, turquoise], zs)
    
    for z in zs:
        # Calculate parameters at this z
        params = my_UVLF.time_evolution(param_values, z)
        Astro_Parameters = my_UVLF.param_wrapper(params)
        # Calculate SFE
        SFEs = my_UVLF.CosmoParams.OmegaM/my_UVLF.CosmoParams.OmegaB * zeus21.sfrd.fstarofz(
            Astro_Parameters, my_UVLF.CosmoParams, z, 10**Mhtab)
        # Assign color
        if len(zs) > 1:
            frac = (z - min(zs)) / (max(zs) - min(zs))
            color = cmap(frac)
        else:
            color = yellow
        # Plot
        plt.plot(Mhtab, SFEs, color = color, ls = '-', zorder = 0)

        if ((z == max(zs) or z==min(zs)) and id_max): # Putting text on the plot indicating the maximum SFE for the highest & lowest redshifts
            max_idx = np.argmax(SFEs) # Maximum /= M_c unless the power law indices are the same!
            plt.scatter(Mhtab[max_idx], SFEs[max_idx], color = color, s = 100, zorder = 1)
            if z==min(zs): # Positioning the text
                x = (Mhtab[max_idx]-0.7)
                y = SFEs[max_idx]-0.01
            else:
                x = Mhtab[max_idx]
                y = 0.6*SFEs[max_idx]
            plt.text(x, y, fr'$f_\star = {round(SFEs[max_idx], 2)}$', color = color, ha = 'center', 
                         va = 'center', size = 17)

    if len(zs)>1:
    # Color bar things:
        cax = fig.add_axes(cax_rect)
        cbar = fig.colorbar(sm, cax=cax, boundaries=boundaries, ticks=zs, aspect = 14)
        cbar.set_label(r'$z$')

    # Labels
    ax.set_xlabel(r'$\log{M_h/M_{\odot}}$')
    ax.set_ylabel(r'$f_{\star} = \dot M_{\star} / \dot M_{\rm{gas}}$')
    ax.set_yscale('log')
        
        
    return fig 

def sfe_over_time(my_UVLF, param_values, Mhtab = None, zmax = 15, steps = 200):
    """
    Plot the SFE over time for a halo that will come to have a certain mass at z = 0
    Inputs:
        my_UVLF [eMCMC UVLF object]: UVLF object used for its methods such as applying time evolution to parameter values & wrapping them in the
                                        right format for zeus21
        param_values [1darray]: values of parameters to calculate SFE
        Mhtab [1darray]: log10(Mh) to plot over
        zmax [float]: maximum redshift to go up to (the plot will go down to z = 0 by default)
        steps [int]: number of steps in redshift (makes the plot smoother)
    Returns:
        fig: A figure showing the SFE for halos of different masses from zmax to 0
    """
    
    fig, ax = plt.subplots()
    zs = np.linspace(0, zmax, steps)

    if Mhtab is None:
        Mhtab = np.arange(9.5, 13.5, 0.5)

    cmap, sm, boundaries = create_custom_colorbar([periwinkle, red], Mhtab)

    for Mh in Mhtab:
        SFEs = np.zeros(steps)
        Mz = cosmo_calc.calc_Mz(zs, Mh, my_UVLF) # calculate the halo mass at each redshift
        for i, (z, Mz_i) in enumerate(zip(zs, Mz)):
            params = my_UVLF.time_evolution(param_values, z) # calculate the parameter values at the given redshift
            Astro_Parameters = my_UVLF.param_wrapper(params)
                           
            SFEs[i] = my_UVLF.CosmoParams.OmegaM/my_UVLF.CosmoParams.OmegaB * zeus21.sfrd.fstarofz(
                Astro_Parameters, my_UVLF.CosmoParams, z, 10**Mz_i) #calculate SFE at the given redshift
        
        frac = (Mh - Mhtab.min()) / (Mhtab.max() - Mhtab.min())
        color = cmap(frac)
        ax.plot(zs, SFEs, color = color, ls = '-')

    # Color bar things:
    cbar = fig.colorbar(sm, ax=ax, boundaries=boundaries, ticks=Mhtab, aspect = 14)
    cbar.set_label(r'$\log_{10}{M_h}$ at $z=0$')

    ax.set_ylabel(r'$f_{\star} = \dot{M}_{\star} / \dot{M}_{\rm{gas}}$')
    ax.set_xlabel('redshift')
    ax.set_yscale('log')

    return fig

def sigma_Mh(my_UVLF, param_values, zs = None, plot_Gelli = True):
    """
    Plot the relationship between sigma & Mh over different redshifts
    Inputs:
        my_UVLF [eMCMC UVLF object]: UVLF object used for its methods such as applying time evolution to parameter values & wrapping them in the
                                        right format for zeus21
        param_values [1darray]: values of parameters to calculate SFE
        zs [1darray]: redshifts to plot
        plot_Gelli [bool]: whether or not to plot the Gelli+24 line for comparison
    Returns:
        fig: A figure showing the relationship between sigma_UV & Mh at different redshifts, compared to the parameterization used in Gelli+24
    """
    
    fig = plt.figure(figsize=(5, 3))            
    ax_rect  = [0.10, 0.12, 0.78, 0.80]       # left, bottom, width, height (0-1)
    cax_rect = [0.895, 0.12, 0.04, 0.80]      # narrow right slot for cbar
    ax = fig.add_axes(ax_rect)

    if zs is None:
        zs = np.arange(4, 16, 2)
    
    Mhtab = np.log10(my_UVLF.HMFintclass.Mhtab)
    indices = np.where((Mhtab >= 8.5) & (Mhtab <= 12))[0]
    Mhtab = Mhtab[indices]
    
    if len(zs) > 1:
        cmap, sm, boundaries = create_custom_colorbar([yellow, turquoise], zs)
        for z in zs:
            sig_array = my_UVLF.time_evolution(param_values, z)[-3][indices]

            frac = (z - min(zs)) / (max(zs) - min(zs))
            color = cmap(frac)
            ax.plot(Mhtab, sig_array, ls = '-', color = color)

        # Color bar things:
        cax = fig.add_axes(cax_rect)
        cbar = fig.colorbar(sm, cax=cax, boundaries=boundaries, ticks=zs, aspect = 14)
        cbar.set_label(r'$z$')

    else:
        sig_array = my_UVLF.time_evolution(param_values, zs[0])[-1][indices]
        color = yellow
        ax.plot(Mhtab, sig_array, ls = '-', color = color)


    if plot_Gelli: # Compare to the Gelli+24 parameterization
        sig_gelli = (-0.34 * Mhtab) + 4.5
        ax.plot(Mhtab, sig_gelli, linestyle = '--', color = red, label = 'Gelli+24')
        ax.legend()

    # Plot min(sig)
    min_sig = param_values[11]
    ax.axhline(min_sig, zorder = 0, ls = 'dotted', color = navy, lw = 1, label = r'$\min(\sigma_{\rm{UV}})$')

    # Labels
    ax.set_xlabel(r'$\log{M_h/M_{\odot}}$')
    ax.set_ylabel(r'$\sigma_{\rm{UV}}$')
    
    return fig 

def walkers(my_UVLF, param_values, backend_file = None, thin = 8000, burn_in = None, ncols = 3, include_params = None):
    """
    Plot MCMC walkers along with burn in cutoff & best fit parameter values and the imposed upper/lower limits
    Inputs:
        my_UVLF [eMCMC UVLF object]: UVLF object used for its methods such as applying time evolution to parameter values & wrapping them in the
                                        right format for zeus21
        param_values [1darray]: fit parameter values for comparison to walkers
        thin [int]: number of steps to thin the chains by (to make it easier to see the path in the plot)
        burn_in [int]: number of steps removed from each chain before they are used to calculate parameters
        ncols [int]: number of columns to use in plotting
    Return:
        fig showing the paths of each walker for each parameter with the burn in and best fit value shown along with upper/lower limits on the parameters
    """
    # Get walkers with no burn in
    if backend_file is None:
        walkers = my_UVLF.sampler.get_chain(discard = 0)
    else:
        reader = emcee.backends.HDFBackend(backend_file)
        walkers = reader.get_chain(discard = 0)

    # Get the standard burn in (for plotting purposes)
    if burn_in is None:
        burn_in = my_UVLF.burn_in

    # Thin the chain
    steps = np.arange(0, len(walkers[:,0][:,0]), thin)
    walkers = walkers[0::thin]

    fit_params= my_UVLF.param_data['fit'].values.astype(bool)
    lowers = my_UVLF.param_data['lower'][fit_params]
    uppers = my_UVLF.param_data['upper'][fit_params]
    labels = my_UVLF.param_data['label'][fit_params]
    if include_params is not None:
        include = my_UVLF.param_data[fit_params].index.isin(include_params)
        walkers = walkers[:,:, include]
        lowers = lowers[include]
        uppers = uppers[include]
        labels = labels[include]

    # Figure stuff
    nparams = walkers.shape[-1]
    fig = plt.figure()
    plt.subplots_adjust(wspace = 0.4, hspace = 0.2)
    gs = GridSpec(math.ceil((nparams+1)/ncols), ncols, figure=fig)
    
    axs = []
    for i, (lower, upper, label) in enumerate(zip(lowers, uppers, labels)): # Iterate through the parameters
        ax = fig.add_subplot(gs[math.floor(i/ncols), i%ncols]) # Add an axis for each parameter
        axs.append(ax)
        for j in range(len(walkers[i])):
            ax.plot(steps, walkers[:,j][:,i], color = red, ls = '-', alpha = 0.3) # Plot each walker for the given parameter
        ax.axhline(param_values[i], color = turquoise, ls = 'dashed', label = 'best fit value') # Plot the best fit value
        ax.axhline(lower, color = yellow, label = 'upper/lower')
        ax.axhline(upper, color = yellow)
        
        # Set label, ticks, etc.
        ax.set_ylabel(fr'{label}', labelpad = -0.6)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.axvline(burn_in, color = navy, ls = '-', label = 'burn in cutoff')
    
    # Labels, etc.
    ax.plot(0,0.3, color = red, ls = '-', alpha = 0.6, label = f'walkers, every {thin} steps') # get label

    multicol(gs, axs, xlabel = 'step number', ylabel = '', include_legend = True)
    
    return fig


def PMh(my_UVLF, param_values, MUV_range = None, z_range = None, Mhtab=None, z_err=0.5, ncols=3):
    """
    Calculate & plot the probability distribution of host halo mass for galaxies of a given MUV
    
    :param my_UVLF: [eMCMC UVLF object] UVLF object used for its methods such as applying time evolution to parameter values & wrapping them in the
                                        right format for zeus21
    :param param_values: [1darray] of fit parameter values
    :param MUV_range: [1darray] list of MUVs to calculate the probability distribution for (separate from MUVcenters which is used as an underlying grid-- this should be a small number of MUVs). If None,
                    will default to a subsection of the MUVs in my_UVLF MUVcenters
    :param z_range: [1darray] Redshifts to plot (will appear as different subplots). If none are specified, will default to the redshifts contained in my_UVLF
    :param Mhtab: [1darray] Range of halo masses to consider (in log_10(M/M_odot)). If none are specified, will default to Mhtab contained within my_UVLF (which is a large range of masses)
    :param z_err: [float] uncertainty on redshift measurements
    :param ncols: [int] number of columns to plot the subplots over

    Returns:
        Figure showing the probability distribution of halo mass for a given galaxy MUV, given model parameters passed as param_values
    """

    MUVcenters, MUVwidths = my_UVLF.get_all_MUV_bins()

    if MUV_range is None:
        MUV_range = np.arange(int(MUVcenters.min()), int(MUVcenters.max())+1.5, 1.5)

    if z_range is None:
        z_range = my_UVLF.zs

    nz = len(z_range)
    fig = plt.figure()
    gs = GridSpec(math.ceil(nz / ncols), ncols, figure=fig)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Colormap
    print(MUV_range)
    if len(MUV_range) > 1:
        cmap, sm, boundaries = create_custom_colorbar([orange, navy], MUV_range)

    if Mhtab is not None:
        Mhtab = 10**Mhtab
        Mhtab_og = my_UVLF.HMFintclass.Mhtab # To be able to reset later
        my_UVLF.HMFintclass.Mhtab = Mhtab
    else:
        Mhtab_og = None
        Mhtab = my_UVLF.HMFintclass.Mhtab
    minMUV = my_UVLF.calc_min_MUV(Mhtab)

    axs = []


    for iz, z in enumerate(z_range): # Add the log likelihoods together for each redshift. The log likelihood is just a sum over all the points
        # anyways, so this makes sense

        ax = fig.add_subplot(gs[math.floor(iz/ncols), iz%ncols])
        axs.append(ax)

        ax.set_title(f'$z \simeq {z}$')

        weights = my_UVLF.UVLF_wrapper(z,z_err,MUVcenters, MUVwidths, param_values, return_weights = True)


        # HMF prior (normalized)
        hmf = my_UVLF.HMFintclass.HMF_int(Mhtab, z)
        PMh = hmf*Mhtab*np.log(10)/ np.trapz(hmf*Mhtab*np.log(10), np.log10(Mhtab)) # Looks weird because it's an integral over logMh. See notes from 2/9 where I derive this

        # Loop over MUV values for color-coded lines
        for MUV in MUV_range:

            iMUV = np.argmin(np.abs(MUVcenters-MUV))

            PMUV_given_Mh = weights[:, iMUV]
            
            # numerator of Bayes theorem
            numerator= PMUV_given_Mh * PMh 

            # normalize posterior
            posterior = numerator / np.trapz(numerator, np.log10(Mhtab))

            # Color
            if len(MUV_range) > 1:
                frac = (MUV - min(MUV_range)) / (max(MUV_range) - min(MUV_range))
                color = cmap(frac)
            else:
                color = navy

            ax.plot(np.log10(Mhtab), posterior, color=color, ls='-')

    multicol(gs, axs = axs, xlabel = r'$\log_{10} M_h / M_\odot$', ylabel = r'$p(M_h \mid M_{\rm UV})$')

    # Colorbar
    cbar = fig.colorbar(sm, ax=axs, boundaries=boundaries, ticks=MUV_range, aspect=4*math.ceil(nz/ncols))
    cbar.set_label(r'$M_{\rm{UV}}$')

    if Mhtab_og is not None:
        my_UVLF.HMFintclass.Mhtab = Mhtab_og # Reset Mhtab if it's been altered

    return fig

def bias(my_UVLF, compare_fits, fit_labels, ncols = 3, z_plot=None, z_errs=None, fit_colors = None):
    """
    Plot bias vs. bias data
    
    :param my_UVLF: [eMCMC UVLF object] UVLF object used for its methods such as applying time evolution to parameter values & wrapping them in the
                                        right format for zeus21
    :param compare_fits: [list of lists] different fits to compare the bias calculations for
    :param fit_labels: [list] labels for each of compare_fits
    :param ncols: [int] number of columns for plotting
    :param z_plot: [list] redshifts to plot. If None, will default to the redshifts of the data
    :param z_errs: [list] redshift uncertainty. If None, will default to the zerr of the data or 0.5
    """

    if z_plot is None:
        z_plot = [dat[0] for dat in my_UVLF.data] # Get the redshifts that correspond to the input data
        z_errs = [dat[1] for dat in my_UVLF.data] # Get redshift error from data
        dat = my_UVLF.data
    else:
        if z_errs is None:
            z_errs = np.full_like(z_plot, 0.5, dtype = float) # If the user provides z but not zerr, default to 0.5 for each redshift
        if type(z_plot) == list:
            z_plot = np.array(z_plot)
        if type(z_errs) == list:
            z_errs = np.array(z_errs)

    nz = len(z_plot)
    fig = plt.figure()
    gs = GridSpec(math.ceil((nz+1)/ncols), ncols, figure=fig)
    plt.subplots_adjust(wspace = 0, hspace = 0.3)

    MUVcenters, MUVwidths = my_UVLF.get_all_MUV_bins()

    # Hard code bias data for now
    bias_data = np.loadtxt('/Users/eb35267/Desktop/code/home/data/bias.txt', unpack = True)

    if fit_colors is None:
        fit_colors = [turquoise, yellow, orange, green]

    axs = []
    all_bias = []
    for i, (z, zerr) in enumerate(zip(z_plot, z_errs)):
        ax = fig.add_subplot(gs[math.floor(i/ncols), i%ncols])
        axs.append(ax)
        z_bias = []
        for (fit, label, color, ls) in zip(compare_fits, fit_labels, fit_colors, ['solid', 'dashdot', 'dotted', 'dashed']): 
            _, bias = my_UVLF.UVLF_wrapper(z, zerr, MUVcenters, MUVwidths, fit, get_bias = True)
            z_bias.append(bias)
            ax.plot(MUVcenters, bias, ls = ls, color = color, label = label)

        ax.set_title(fr'${round(z-zerr, 1)} \lesssim z \lesssim {round(z+zerr, 1)}$', fontsize = 20, pad = 8)
        ax.invert_xaxis()
        all_bias.append(np.concatenate(z_bias))
        bias_inds = ((bias_data[0] < z + zerr) & (bias_data[0] > z - zerr))
        ax.errorbar(bias_data[1][bias_inds], bias_data[2][bias_inds], yerr = bias_data[3][bias_inds], fmt = 'o', markeredgewidth=0, 
                    color = navy, label = 'Muñoz+23 effective bias', markersize = 9)

    xlabel = r'$M_{\rm{UV}}$ [mag]'
    ylabel = r'$b(M_{\rm{UV}})$'

    # Flatten all ydat values from sorted_data to get the min and max
    all_ydat = np.concatenate([np.concatenate([ds[3] for ds in z_separated])  # ds[3] is ydat
                                for z_separated in my_UVLF.sorted_data])

    all_bias = np.concatenate(all_bias)
    ylims = (np.min(all_bias)-0.25, np.max(all_bias)+0.25)
    
    # Formatting
    multicol(gs, axs, xlabel, ylabel, include_legend = True, ylims=ylims)

    return fig

def sigma_over_z(my_UVLF, fits, fit_labels, zs, Mh):
    """
    Inputs:
        my_UVLF [UVLF object]
        fits [list of 1darrays]: best fit parameters for different fits
        fit_labels [list of strings]: labels for each fit
        zs [1darray]: redshifts to plot
        Mh [float]: log10(M_halo) to calculate sigma(z) at
    Returns: 
        fig: shows sigma(z) and plots min(sigma)
    """
    # Temporarily set the UVLF HMF to be the value of Mh
    save_HMF = my_UVLF.HMFintclass.Mhtab
    my_UVLF.HMFintclass.Mhtab = [10**Mh]

    # Calculate sigma(z)
    all_sigs = [[my_UVLF.time_evolution(fit, z)[-3][0] for z in zs] for fit in fits]
    
    # Plotting
    fig, ax = plt.subplots()

    for sigs, fit_label in zip(all_sigs, fit_labels):
        ax.plot(zs, sigs, label = fit_label)

    # Get & plot min(sigma)
    if len(fits) == 1:
        if my_UVLF.param_data.loc['min_sig', 'fit'] == False:
            min_sig = my_UVLF.param_data.loc['min_sig', 'value']
        else:
            min_sig = fits[0][11]
        ax.axhline(min_sig, ls = 'dotted', lw = 1, color = 'black', zorder = 0)
        ax.text(zs[-3], 1.05*min_sig, r'$\min (\sigma_{\rm{UV}})=$'+f'{round(min_sig, 2)}', ha = 'center')

    # Labels
    ax.set_ylabel(r'$\sigma_{\rm{UV}}$')
    ax.set_xlabel('redshift')

    ax.legend()

    # Set the UVLF HMF back to its original value
    my_UVLF.HMFintclass.Mhtab = save_HMF

    return fig

def delta_chi2(my_UVLF, base, base_name, comparison_fits, labels):
    """
    Compare the chi2 of two fits at individual redshifts
    Inputs:
        my_UVLF [UVLF object]
        base [1darray]: baseline fit for comparison
        base_name [str]: name of the baseline fit for comparison
        comparison_fits [list of 1darrays]: fits for comparison
        labels [list of strings]: labels of the fits for comparison
    Returns:
        Fig showing the delta chi2 at each redshift as well as the cumulative delta chi2
    """
    chisq_base = -2*my_UVLF.log_like(base, return_by_z = True)
    chisqs_compare = [-2*my_UVLF.log_like(fit, return_by_z = True) for fit in comparison_fits]

    fig, ax = plt.subplots(2, 1, sharex = True)
    fig.set_size_inches(5, 8)
    plt.subplots_adjust(hspace = 0)

    for chisq_compare, label, color, ls in zip(chisqs_compare, labels, colors, ['solid', 'dashed', 'dotdash']):
        delta_chi2 = chisq_compare - chisq_base
        ax[0].plot(my_UVLF.zs, delta_chi2, color = color, ls = ls, label = label)
        ax[0].scatter(my_UVLF.zs, delta_chi2, color = color, s = 50)
        ax[1].plot(my_UVLF.zs, np.cumsum(delta_chi2), color = color, ls = ls)
        

    ax[0].axhline(0, color = 'black', ls = 'dotted', lw = 1, zorder = 0)
    ax[0].set_ylabel(r'$\Delta \chi^2$')
    ax[1].set_ylabel(r'Cumulative $\Delta \chi^2$')
    ax[1].set_xlabel(r'$z$')
    ax[0].legend()

    ax[0].set_title(f'Comparison to {base_name}')

    return fig



# Tables--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def make_table(best_fits, param_labels, fit_labels, bounds):
    """
    Make a table to compare best fit values of parameters
    Inputs:
        best_fits [list of lists]: list of best fit parameters
        param_labels [list of strs]: names of parameters (rows of the table)
        fit_labels [list]: names of each type of fit (columns of the table)
        bounds [list of Nx2 arrays]: list of arrays with col1 = lower bound, col2 = upper bound on the best fit parameters
    Outputs:
        dataframe table with labeled parameters for comparison
    """
    df_fill = []
    for best_fit, bound in zip(best_fits, bounds):
        has_bounds = bound is not None and ~np.all(np.isnan(bound), axis=0)  # bool array, True where bounds exist
        
        if has_bounds.any() and (any(bound[1][has_bounds] - best_fit[has_bounds] < 0) or 
                                any(bound[0][has_bounds] - best_fit[has_bounds] > 0)):
            print("Your best fit values are not within your 16th and 84th percentile upper and lower bounds. "
                "Take care when interpreting this table and consider broadening your priors.")
        
        row = []
        for j, (bf, lo, hi) in enumerate(zip(best_fit, bound[0], bound[1])):
            if has_bounds[j]:
                row.append(f'${round(bf, 3)}^{{+{max(round(hi-bf,2),0)}}}_{{{min(round(lo-bf,2),0)}}}$')
            else:
                row.append(bf)
        df_fill.append(row)

    df = pd.DataFrame(df_fill, columns = param_labels)
    df = df.fillna('—')
    df.index = fit_labels

    return df.T






# Helper functions--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def create_custom_colorbar(color_list, data):
    cmap = LinearSegmentedColormap.from_list("custom", color_list)
    dx = data[1]-data[0] # Separation between data in the list
    norm = BoundaryNorm(np.arange(min(data), max(data)+dx, dx), cmap.N)

    boundaries = np.arange(min(data)-(dx/2), max(data)+(1.5*dx), dx) # set bin edges between values
    norm = BoundaryNorm(boundaries, cmap.N)
    sm = ScalarMappable(cmap=cmap, norm=norm)

    return cmap, sm, boundaries

def custom_percent_formatter(x, pos):
    """
    Format a number as a percentage. Helper function for percent_diff_plot.
    """
    return f"{x * 100:.1f}%"

def plot_colors():
    """
    Plot the colors defined in my mplstyle sheet and print their names
    This gives a visual representation of all the colors to make it easier to choose the best one.
    """
    # Create a figure to visualize the colors
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_xlim(0, len(hex_colors))
    ax.set_ylim(0, 1)
    
    # Plot each color as a rectangle
    for i, (hex_color, label) in enumerate(zip(hex_colors, labels)):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=hex_color))
        ax.text(i + 0.5, -0.2, label, ha='center', va='top', fontsize=15, color='black', rotation=45)
        print(f'{label}: {hex_color}')
    
    # Formatting
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    plt.show()
    return fig

def interpolate_colors(hex_colors, total_steps):
    """
    Plot a step-wise gradient between each color in the colors list. There are [steps_between] colors between each color in the list.
    Inputs:
        hex_colors [list]: colors in hexadecimal format
        total_steps [int]: number of steps in the gradient
    Returns:
        interpolated_colors [list]: a list of all of the colors that make up the stepwise gradient
        fig [matplotlib figure]: visualization of the gradient
    """
    # Convert hex colors to RGB
    rgb_colors = [np.array(mplcolors.to_rgb(color)) for color in hex_colors]
    interpolated_colors = []

    steps_between = math.ceil((total_steps-len(hex_colors)) / (len(hex_colors)-1)) # calculate how many colors in between each specified color
    # to end up with the specified number of total colors (or more)
    
    # Interpolate between each pair of colors
    for i in range(len(rgb_colors) - 1):
        start = rgb_colors[i]
        end = rgb_colors[i + 1]
        for t in np.linspace(0, 1, steps_between + 2)[:-1]:  # omit the last to avoid duplicates
            interpolated = (1 - t) * start + t * end
            interpolated_colors.append(mplcolors.to_hex(interpolated))
    
    # Append the last color explicitly
    interpolated_colors.append(hex_colors[-1])

    fig, ax = plt.subplots(figsize=(12,2))
    ax.set_xlim(0, len(interpolated_colors))
    ax.set_ylim(0, 1)
    for i, color in enumerate(interpolated_colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
        ax.text(i + 0.5, -0.2, color, ha='center', va='top', fontsize=15, color='black', rotation=45)
        
    return interpolated_colors, fig

class HandlerStackedLine(HandlerLine2D): # Inherits from the HandlerLine2D class in matplotlib
    """
    Creates multiple horizontal lines to represent a single label in an axis. Initiate with a list of colors you want to include lines for
    """
    def __init__(self, colors, *args, **kwargs):
        self.colors = colors[::-1] # Fix the ordering for the way we plot later
        super().__init__(*args, **kwargs)
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Override the typical create_artists function in HandlerLine2D; use this one instead
        return [Line2D([xdescent, xdescent + width], [ydescent + (i-0.3) * 6] * 2, color=self.colors[i], transform=trans)
                for i in range(len(self.colors))]
    

def multicol(gs, axs, xlabel = '', ylabel = '', xlims = None, ylims = None, include_legend = False):
    """
    Helper function for formatting multi-column plots.
    
    :param gs: matplotlib GridSpec object
    :param axs: list of axes in the figure
    :param xlabel: x axis label text
    :param ylabel: y axis label text
    :param xlims: tuple with the upper and lower x limit to be imposed for each subplot
    :param ylims: tuple with the upper and lower y limit to be imposed for each subplot. 
    :param include_legend: boolean-- whether or not to include a legend. Relies on there being labels in the axes within axs
    """
    
    N = len(axs)
    if include_legend:
        N_inc_leg = N+1
    else:
        N_inc_leg = N

    # Sizing & spacing
    fig = gs.figure
    fig.set_size_inches(4*gs.ncols, 4*math.ceil(N_inc_leg/gs.ncols))

    # Formatting axis ticks
    for i, ax in enumerate(axs): 
        if xlims is not None:
            ax.set_xlim(*xlims)
        if ylims is not None:
            ax.set_ylim(*ylims)                                                                       
            if i%gs.ncols != 0:
                ax.axes.yaxis.set_ticklabels([]) # Only include y axis tick labels on the furthest left side if all axes will have the same y limits

    # Text & labels
    bbox = Bbox.union([ax.get_position() for ax in axs]) # Define the positions of all the axes
    xcenter = bbox.x0 + bbox.width / 2
    ycenter = bbox.y0 + bbox.height / 2
    fig.text(xcenter, bbox.y0 - gs.ncols*0.02 - 0.01, xlabel, ha='center', va='top', fontsize = 20)
    fig.text(bbox.x0 - 0.1 + (gs.ncols/75), ycenter, ylabel, ha='center', va='center', rotation='vertical', fontsize = 20)

    # Create legend & place it outside the axes
    if include_legend:
        legend = OrderedDict() # Avoids duplicates & ensures that the legend is in order of what people will see first
        for ax in axs:
            for h, l in zip(*ax.get_legend_handles_labels()):
                legend.setdefault(l, h)
        handles, labels = list(legend.values()), list(legend.keys())

        # Placement
        fig_ax = fig.add_subplot(gs[math.floor((N)/gs.ncols), (N)%gs.ncols])
        fig_ax.axis("off")  
        fig_ax.legend(handles, labels, loc='upper center')
    
    