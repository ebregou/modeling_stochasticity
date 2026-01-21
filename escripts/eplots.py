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

    return fig, ax

def make_corner(my_UVLF, backend_file = None, true_vals = None, title = '', burn_in = None, excluded_params = [], 
                sample_color = red, truth_color = turquoise):
    """
    Plot a corner plot, adding true values if any are given.
    Inputs:
        backend_file [str]: .h5 file containing previously saved chain
        samples [array]: samples from the MCMC chain
        labels [list of strs]: label for each sample
        true_vals [1darray]: optional true values for each parameter. Don't include true values for parameters in excluded_params
        title [str]: plot title
        burn_in [int]: number of samples to discard as burnin
        excluded_params [list]: list of parameters to exclude from the plot
        sample_color [str]: color for the samples
        truth_color [str]: color to plot the best fit
    Returns:
        corner_plot [matplotlib figure]: corner plot showing the values and covariances of each parameter, and, optionally, their true values
    """
    
    # Get samples & parameter labels, excluding parameters that weren't fit
    samples, _, _, labels = my_UVLF.get_fit(backend_file, exclude_unfit = True, burn_in = burn_in, excluded_params = excluded_params) 
    

    if true_vals is None:
        corner_plot = corner.corner(samples, labels=labels, color = sample_color, plot_contours = True) 
    else:
        corner_plot = corner.corner(samples, labels=labels, color = sample_color, truths = true_vals, 
                                    truth_color= truth_color, plot_contours = True)

    corner_plot.suptitle(title, y = 1.02)
        
    return corner_plot

def evolving_UVLF_fit(my_UVLF, backend_file = None, z_plot = None, z_errs = None, plot_from_chain = True, nsamples = 100, 
                      
                      comparison_fits = [], comparison_labels = [], 
                      ncols = 3, title = 'UVLF data vs. MCMC fit', burn_in = None):
    """
    Plot the UVLF at different redshifts
    Inputs:
        my_UVLF: eMCMC UVLF object
        backend_file [str]: .h5 file containing previously saved chain
        z_plot [list of floats]: redshifts to plot, or None if you want to plot all the redshifts for which there exists data
        z_errs [list of floats]: the uncertainty in redshift, or None if you want to use the uncertainty that corresponds to the data.
                                If you provide redshifts to plot but no redshift error, it will default to 0.5.
        plot_from_chain [bool]: whether or not to plot the best fit and samples from the chain stored in my_UVLF. If False, only comparison_fits
                             parameter values will be plotted (so you can quickly check different fits this way).
        nsamples [int]: number of samples from the MCMC chain you want to appear in addition to the best fit
        comparison_fits [list of lists]: lists of parameter values that will be used to create comparison UVLFs
        comparison_labels [list of strs]: labels that correspond to the parameter values in comparison_fits, for the legend
        ncols [int]: number of columns to plot
        title [str]: overarching title of the plot
        burn_in [int]: number of samples to discard as burnin
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

        idxs = np.concatenate([np.where(np.abs(data_z - z) < zerr)[0] for z, zerr in zip(z_plot, z_errs)])
        dat = [my_UVLF.data[i] for i in idxs]
        # if z_errs is None:
        #     z_errs = np.full_like(z_pot)
    

    if len(comparison_fits) > 0:
        assert len(comparison_labels) > 0, 'If specifying comparison fits you must also give their labels'

    # Figure out how many subplots to make
    nz = len(z_plot)
    fig = plt.figure()
    gs = GridSpec(math.ceil((nz+1)/ncols), ncols, figure=fig)
    
    # Adjust size & spacing
    fig.set_size_inches(4*ncols, 4*math.ceil((nz+1)/ncols))
    plt.subplots_adjust(wspace = 0, hspace = 0.3)

    # Get samples & best fit
    if plot_from_chain:
        samples, best_fit, _, _= my_UVLF.get_fit(backend_file = backend_file, exclude_unfit = True, burn_in = burn_in)
    
    chi2_comparison = [-2*my_UVLF.log_like(fit, dat) for fit in comparison_fits]

    # Set the same y limits for all the plots
    ylo, yhi = np.log10(min([min(dat[3]) for dat in my_UVLF.data]))-0.25, np.log10(max([max(dat[3]) for dat in my_UVLF.data]))+0.25 

    # Get the biggest possible grid of x data to plot over
    plot_xdat, inds = np.unique(np.concatenate([data[2] for data in my_UVLF.data]), return_index = True) 
    # Get the corresponding x error
    plot_xerr = np.concatenate([data[6] for data in my_UVLF.data])[inds]

    axs = []
    for i, (z, zerr) in enumerate(zip(z_plot, z_errs)): 

        ax = fig.add_subplot(gs[math.floor(i/ncols), i%ncols])
        axs.append(ax)

        if plot_from_chain:
            # Choose random samples from the MCMC chain
            inds = np.random.randint(len(samples), size=nsamples) # Choose nsamples from the chain

            # Plot each sample from the chain
            for ind in inds: 
                sample = samples[ind]
                ax.plot(plot_xdat, np.log10(my_UVLF.UVLF_wrapper(z,zerr, plot_xdat, plot_xerr,sample)), alpha=6/max(nsamples, 6), 
                           color = red, linestyle = '-', zorder = 0)
    
            # Plot best fit
            chi2 = -2* my_UVLF.log_like(best_fit, alt_data = dat)

            ax.plot(plot_xdat, np.log10(my_UVLF.UVLF_wrapper(z, zerr, plot_xdat, plot_xerr, best_fit)), color = red, linestyle = '-', 
                                           zorder = 0, label = fr'best fit, $\chi^2 = {chi2:.0f}$', lw = 5)
            
            ax.plot([0], [0], color = red, alpha = 0.1, label = 'sampled fits', linestyle = '-', lw = 5) # Create the label for the samples

        # Plot comparison fits
        for fit, chi2_fit, label, color, ls in zip(comparison_fits, chi2_comparison, comparison_labels, 
                                                   [turquoise, yellow, orange, green], 
                                        ['solid', 'dashdot', 'dashed', 'dotted']):
            ax.plot(plot_xdat, np.log10(my_UVLF.UVLF_wrapper(z,zerr,plot_xdat,plot_xerr, fit)), color = color, 
                       label = fr'{label}, $\chi^2 = {chi2_fit:.0f}$', linestyle = ls, zorder = 1)
        
        # Plot data
        idx = np.where(np.abs(data_z - z) <= zerr)[0]
        if len(idx) > 0:
            zbin = my_UVLF.sorted_data[idx[0]]
            for dat_z, fmt in zip(zbin, ['o', 'v', 's', 'D', '^']):
                xdat, ydat, yerr_upper, yerr_lower, xerr = dat_z[2], dat_z[3], dat_z[4], dat_z[5], dat_z[6]
                ax.scatter(xdat, np.log10(ydat), marker = fmt, label = dat_z[7], c = navy, s = 180)
                ax.vlines(xdat, np.clip(np.log10(ydat - yerr_lower), -100, 100), np.clip(np.log10(ydat + yerr_upper), -100, 100), ls = '-', 
                        colors = navy, linewidth =6)

        ax.invert_xaxis()
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(np.max(plot_xdat)+0.25, np.min(plot_xdat)-0.25)
        ax.set_title(fr'${round(z-zerr, 1)} \lesssim z \lesssim {round(z+zerr, 1)}$', fontsize = 20, pad = 8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # Make it so that only integers can be used in the  
                                                                                       # axis labels                                                                            
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # Make it so that only integers can be used in the  
                                                                                       # axis labels                                                                          
        if i%ncols != 0:
            ax.axes.yaxis.set_ticklabels([])

    # Create legend
    legend = OrderedDict() # Avoids duplicates & ensures that the legend is in order of what people will see first
    for ax in axs:
        for h, l in zip(*ax.get_legend_handles_labels()):
            legend.setdefault(l, h)
    handles, labels = list(legend.values()), list(legend.keys())


    # Text & labels
    ylabel = r'$\log_{10}(\phi_{\rm{UV}}$ [$\rm{mag}^{-1} \rm{Mpc}^{-3}$])'
    xlabel = r'$M_{\rm{UV}}$ [mag]'
    if math.ceil((nz+1)/ncols) > 1: # This means that there's more than one row of plots
        if (i+1)%ncols==0:
            fig.text(0.065, 0.5, ylabel, rotation = 'vertical')
            fig.text(0.5, 0.35, xlabel, ha = 'center')
        else:
            fig.supylabel(ylabel, x = 0.05)
            fig.supxlabel(xlabel, y = 0.07)
    else:
        if nz == 1:
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
        else:
            fig.text(0.065, 0.225, ylabel, rotation = 'vertical')
            fig.text(0.375, 0, xlabel, ha = 'center')

    # Put legend outside the last axis
    fig_ax = fig.add_subplot(gs[math.floor((i+1)/ncols), (i+1)%ncols])
    fig_ax.axis("off")  
    fig_ax.legend(handles, labels, loc='upper left')

    return fig

def MUV_distribution(my_UVLF, zs, param_values, Mhtab):
    """
    Plot P(MUV|Mh) for two different redshifts given a set of UVLF parameters
    Inputs:
        my_UVLF [eMCMC UVLF object]: UVLF object used for its methods such as applying time evolution to parameter values & wrapping them in the
                                        right format for zeus21
        zs [1darray]: <= 2 redshifts to plot for comparison
        param_values [1darray]: values of parameters to calculate MUV from Mh
        Mhtab [1darray]: log(Mh/M_odot) for plotting
    Outputs:
        fig, ax: Figure showing P(MUV|Mh) for different halo masses. Note that even without time-evolving parameters, the P(MUV|Mh) will look different at
        differnet redshifts due time-evolving mass accretion rates
    """
    assert(len(zs)) <= 2, 'This routine can currently only accommodate 2 redshifts at once'
    
    # Create figure
    fig = plt.figure(figsize=(12,8))            # keep same figsize for both figs
    ax_rect  = [0.10, 0.12, 0.78, 0.80]       # left, bottom, width, height (0-1)
    cax_rect = [0.895, 0.12, 0.04, 0.80]      # narrow right slot for cbar
    ax = fig.add_axes(ax_rect)
    ax.invert_xaxis()

    MUV_range = np.linspace(-15, -24, 500) # Define range of MUVs to examine
    my_UVLF.HMFintclass.Mhtab = 10**Mhtab # Set the UVLF table of halo masses to the input value

    if len(Mhtab) > 1:
        # Create gradient for color-coding curves based on corresponding halo mass
        cmap, sm, boundaries = create_custom_colorbar([periwinkle, red], Mhtab)
    
    for z, ls in zip(zs, ['-', '--']): # Plot curves for each redshift
        params = my_UVLF.time_evolution(param_values, z) # Calculate how parameters evolve with redshift
        astroparams = my_UVLF.param_wrapper(params)# Get the parameters in the right format for use with zeus21
        SFRlist = zeus21.sfrd.SFR_II(astroparams, my_UVLF.CosmoParams, my_UVLF.HMFintclass, 10**Mhtab, z, z) # Calculate SFR
        MUVbarlist = zeus21.UVLFs.MUV_of_SFR(SFRlist, astroparams._kappaUV) # Use SFR to calculate average MUV
        MUVbarlist = np.fmin(MUVbarlist, zeus21.constants._MAGMAX) # Make sure that MUV doesn't exceed a set value

        ax.plot(MUV_range[0], 0, ls = ls, color = 'black', label = f'$z={z}$') # Make invisible line for legend purposes

        for sigUV, MUV_bar, Mh in zip(params[-1], MUVbarlist, Mhtab): # Plot different color-coded Gaussians for different halo masses
            if len(Mhtab) > 1:
                # Get color based on halo mass
                frac = (Mh - Mhtab.min()) / (Mhtab.max() - Mhtab.min()) 
                color = cmap(frac)
            else:
                color = '#9689d5'
            P = (1/(sigUV * np.sqrt(2*np.pi)))*np.exp(-(MUV_range-MUV_bar)**2/(2*sigUV**2)) # Calculate Gaussian based on MUV_bar & sig_UV
            #ax.axvline(MUV_bar, color = turquoise, ls = 'dashed', lw = 3)
            ax.plot(MUV_range, P, color = color, ls = ls)

    # Color bar things:
    if len(Mhtab)>1:
        cax = fig.add_axes(cax_rect)
        cbar = fig.colorbar(sm, cax=cax, boundaries=boundaries, ticks=Mhtab, aspect = 14)
        cbar.set_label(r'$\log_{10}{M_h}$')

    # Figure labels
    ax.set_ylabel(r'$p(M_{\rm{UV}}|M_h)$')
    ax.set_xlabel(r'$M_{\rm{UV}}$ [mag]')
    ax.legend(loc = 'upper left')

    return fig, ax

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

    fig = plt.figure(figsize=(8, 6))            
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

def sigma_Mh(my_UVLF, param_values, zs = None):
    """
    Plot the relationship between sigma & Mh over different redshifts
    Inputs:
        my_UVLF [eMCMC UVLF object]: UVLF object used for its methods such as applying time evolution to parameter values & wrapping them in the
                                        right format for zeus21
        param_values [1darray]: values of parameters to calculate SFE
        zs [1darray]: redshifts to plot
    Returns:
        fig: A figure showing the relationship between sigma_UV & Mh at different redshifts, compared to the parameterization used in Gelli+24
    """
    
    fig = plt.figure(figsize=(8, 6))            
    ax_rect  = [0.10, 0.12, 0.78, 0.80]       # left, bottom, width, height (0-1)
    cax_rect = [0.895, 0.12, 0.04, 0.80]      # narrow right slot for cbar
    ax = fig.add_axes(ax_rect)

    if zs is None:
        zs = np.arange(4, 16, 2)

    cmap, sm, boundaries = create_custom_colorbar([yellow, turquoise], zs)
    
    Mhtab = np.log10(my_UVLF.HMFintclass.Mhtab)
    indices = np.where((Mhtab >= 8.5) & (Mhtab <= 12))[0]
    Mhtab = Mhtab[indices]
    
    for z in zs:
        sig_array = my_UVLF.time_evolution(param_values, z)[-1][indices]

        frac = (z - min(zs)) / (max(zs) - min(zs))
        color = cmap(frac)
        ax.plot(Mhtab, sig_array, ls = '-', color = color)

    # Compare to the Gelli+24 parameterization
    sig_gelli = (-0.34 * Mhtab) + 4.5
    ax.plot(Mhtab, sig_gelli, linestyle = '--', color = red, label = 'Gelli+24')

    # Color bar things:
    cax = fig.add_axes(cax_rect)
    cbar = fig.colorbar(sm, cax=cax, boundaries=boundaries, ticks=zs, aspect = 14)
    cbar.set_label(r'$z$')

    # Labels
    ax.set_xlabel(r'$\log{M_h/M_{\odot}}$')
    ax.set_ylabel(r'$\sigma_{\rm{UV}}$')

    ax.legend()
    
    return fig

def walkers(my_UVLF, param_values, backend_file = None, thin = 8000, burn_in = None, ncols = 3):
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
        burn_in = 8000

    # Thin the chain
    steps = np.arange(0, len(walkers[:,0][:,0]), thin)
    walkers = walkers[0::thin]

    # Figure stuff
    nparams = walkers.shape[-1]
    fig = plt.figure()
    fig.set_size_inches(3*ncols, 3*math.ceil((nparams+1)/ncols))
    plt.subplots_adjust(wspace = 0.4, hspace = 0.2)
    gs = GridSpec(math.ceil((nparams+1)/ncols), ncols, figure=fig)

    fit_params = my_UVLF.param_data['fit'].values.astype(bool)
    
    for i in range(nparams): # Iterate through the parameters
        ax = fig.add_subplot(gs[math.floor(i/ncols), i%ncols]) # Add an axis for each parameter
        for j in range(len(walkers[i])):
            ax.plot(steps, walkers[:,j][:,i], color = red, ls = '-', alpha = 0.3) # Plot each walker for the given parameter
        ax.axhline(param_values[i], color = turquoise, ls = 'dashed', label = 'best fit value') # Plot the best fit value
        ax.axhline(my_UVLF.param_data['lower'][fit_params].iloc[i], color = yellow, label = 'upper/lower')
        ax.axhline(my_UVLF.param_data['upper'][fit_params].iloc[i], color = yellow)
        
        # Set label, ticks, etc.
        ax.set_ylabel(fr'{my_UVLF.param_data['label'][fit_params].iloc[i]}', labelpad = -0.6)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.axvline(burn_in, color = navy, ls = '-', label = 'burn in cutoff')
    
    # Labels, etc.
    ax.plot(0,0, color = red, ls = '-', alpha = 0.6, label = f'walkers, every {thin} steps') # get label
    fig_ax = fig.add_subplot(gs[math.floor((i+1)/ncols), (i+1)%ncols])
    fig_ax.axis("off")  
    handles, labels = ax.get_legend_handles_labels() # Grab handles/labels from the real axes
    fig_ax.legend(handles, labels, loc='center')
    fig.supxlabel('step number', y = 0.05)
    
    return fig



# Tables--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def make_table(best_fits, param_labels, fit_labels, bounds, file_name = None):
    """
    Make a table to compare best fit values of parameters
    Inputs:
        best_fits [list of lists]: list of best fit parameters
        param_labels [list of strs]: names of parameters (rows of the table)
        fit_labels [list]: names of each type of fit (columns of the table)
        bounds [list of Nx2 arrays]: list of arrays with col1 = lower bound, col2 = upper bound on the best fit parameters
        file_name [str]: path to saved file if you'd like to save the table
    Outputs:
        dataframe table with labeled parameters for comparison
    """
    df_fill = []
    for best_fit, bound in zip(best_fits, bounds):
        if bound is not None:
            if any(bound[:,1]-best_fit < 0) or any(bound[:,0]-best_fit > 0): # Check that the bounds make sense
                print(
                    "Your best fit values are not within your 16th and 84th percentile upper and lower bounds. Take care when interpreting this table and consider broadening your priors."
                )
            df_fill.append([f'${bf}^{{+{max(round(bd[1]-bf,2),0)}}}_{{{min(round(bd[0]-bf,2),0)}}}$' for bf, bd in zip(best_fit, bound)])
        else:
            df_fill.append(best_fit)

    df = pd.DataFrame(df_fill, columns = param_labels)
    df.index = fit_labels

    if file_name is not None:
        dfi.export(df.T, file_name, table_conversion = 'matplotlib', use_mathjax = True, dpi = 200)

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
        ax.text(i + 0.5, -0.2, hex_color, ha='center', va='top', fontsize=15, color='black', rotation=45)
        print(f'{label}: {hex_color}')
    
    # Formatting
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    plt.show()
    return

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
    
    