# Purpose: plotting tools
# Author: Emily Bregou

# Standard packages
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colors
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
import numpy as np
import corner
import math

# Local packages
from escripts import estats
from escripts import eMCMC

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

def percent_diff_plot(x, y0, y1, labels, other_data = None):
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

def plot_corner(samples, labels, true_vals = None, title = ''):
    """
    Plot a corner plot, adding true values if any are given.
    Inputs:
        samples [array]: samples from the MCMC chain
        labels [list of strs]: label for each sample
        true_vals [1darray]: optional true values for each parameter. Don't include true values for parameters in excluded_params
        title [str]: plot title
    Returns:
        corner_plot [matplotlib figure]: corner plot showing the values and covariances of each parameter, and, optionally, their true values
    """
    
    if true_vals is None:
        corner_plot = corner.corner(samples, labels=labels, color = red) 
    else:
        corner_plot = corner.corner(samples, labels=labels, color = red, truths = true_vals, truth_color= turquoise)

    corner_plot.suptitle(title, y = 1.02)
        
    return corner_plot

def plot_evolving_UVLF_fit(my_UVLF, z_plot = None, plot_samples = True, nsamples = 1, comparison_fits = [], comparison_labels = [], 
                           title = 'UVLF data vs. MCMC fit'):
    """
    my_UVLF: eMCMC UVLF object
    z_plot [list of floats]: redshifts to plot, or leave as None if you want to plot all of them
    plot_samples [bool]: whether or not to plot the best fit and samples from my_UVLF. If False, only comparison_fits parameter values
                         will be plotted.
    nsamples [int]: number of samples from the MCMC chain you want to appear in addition to the best fit
    comparison_fits [list of lists]: lists of parameter values that will be used to create comparison UVLFs
    comparison_labels [list of strs]: labels that correspond to the parameter values in comparison_fits, for the legend
    title [str]: overarching title of the plot
    """
    if z_plot is None: # Plot every redshift if none are specified.
        z_plot = [dat[0] for dat in my_UVLF.data]

    # Figure out how many subplots to make
    nz = len(z_plot)
    fig, ax = plt.subplots(1, nz, sharey = True)
    if nz == 1:
        ax = [ax]  # Make ax iterable even if it's just one subplot
    fig.set_size_inches(3.5*nz, 3)
    plt.subplots_adjust(wspace = 0.1)

    # Get samples & best fit
    samples, best_fit, _ = my_UVLF.get_fit()
    chi2 = -2 * my_UVLF.log_like(best_fit)
    chi2_comparison = [-2*my_UVLF.log_like(fit) for fit in comparison_fits]

    # Set the same y limits for all the plots
    ylo, yhi = np.log10(min([min(dat[3]) for dat in my_UVLF.data]))-0.25, np.log10(max([max(dat[3]) for dat in my_UVLF.data]))+0.25 

    plot_xdat, plot_xerr = my_UVLF.data[0][2], my_UVLF.data[0][5], # Use the lowest redshift MUV centers & bins to calculate the theoretical
                                                                # UVLF (this will make it the smoothest)
    i = 0 # Tracks which axis you're on
    for zbin in my_UVLF.sorted_data:

        # Choose random samples from the MCMC chain
        inds = np.random.randint(len(samples), size=nsamples) # Choose nsamples from the chain
    
        zdat, zerr = zbin[0][0], zbin[0][1]

        if zdat not in z_plot:
            print(f'Skipping $z={zdat}$')
            continue

        if plot_samples:
            # Plot each sample from the chain
            for ind in inds: 
                sample = samples[ind]
                ax[i].plot(plot_xdat, np.log10(my_UVLF.UVLF_wrapper(zdat,zerr,plot_xdat, plot_xerr,sample)), alpha=nsamples/3500, color = red, 
                           linestyle = '-', zorder = 0)
    
            # Plot best fit
                chi2 = -2* my_UVLF.log_like(best_fit)
                ax[i].plot(plot_xdat, np.log10(my_UVLF.UVLF_wrapper(zdat, zerr, plot_xdat, plot_xerr, best_fit)), color = red, linestyle = '-', 
                                               zorder = 0, label = f'best fit, $\chi^2 = {chi2:.0f}$')
        # Plot comparison fits
        for fit, chi2_fit, label, color, ls in zip(comparison_fits, chi2_comparison, comparison_labels, [turquoise, yellow, orange, green], 
                                        ['solid', 'dashdot', 'dashed', 'dotted']):
            ax[i].plot(plot_xdat, np.log10(my_UVLF.UVLF_wrapper(zdat,zerr,plot_xdat,plot_xerr, fit)), color = color, 
                       label = f'{label}, $\chi^2 = {chi2_fit:.0f}$', linestyle = ls, zorder = 1)

        # Plot data
        for dat, fmt in zip(zbin, ['o', 'v', 's']):
            _, _, xdat, ydat, yerr_upper, yerr_lower, xerr = eMCMC.decompose_data(dat)
            yerr_lower, yerr_upper = estats.calc_log_error(ydat, yerr_lower), estats.calc_log_error(ydat, yerr_upper)
            ax[i].errorbar(xdat, np.log10(ydat), [yerr_lower, yerr_upper], fmt = fmt, capsize=0, markersize = 9, label = dat[7], 
                           markeredgecolor = 'none', markerfacecolor = navy, ecolor = navy)
        
        ax[i].invert_xaxis()
        ax[i].set_ylim(ylo, yhi)
        ax[i].set_xlim(np.max(plot_xdat)+0.25, np.min(plot_xdat)-0.25)
        ax[i].set_title(f'$z \simeq {round(zdat, 1)}$')
        ax[i].xaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # Make it so that only integers can be used in the  
                                                                                       # axis labels

        i+=1 # Move to the next axis
            
    
    ax[0].set_ylabel(r'$\log_{10}(\phi_{\rm{UV}}$ [$\rm{mag}^{-1} \rm{Mpc}^{-3}$])')
    fig.text(.5, -0.05, r'$M_{\rm{UV}}$ [mag]', ha = 'center')
    fig.text(.5, 1.1, title, ha = 'center', fontsize = 20)
    
    if len(ax) < 2: # Put the legend outside the box if there's only one redshift bin
        ax[-1].legend(bbox_to_anchor= (1, 1))
    else:
        ax[-1].legend()

    return fig

def interpolate_colors(hex_colors, total_steps, show_plot = False):
    """
    Create a step-wise gradient between each color in the colors list. There are [steps_between] colors between each color in the list.
    Inputs:
        hex_colors [list]: colors in hexadecimal format
        total_steps [int]: number of steps in the gradient
        show_plot [bool]: whether or not to show the gradient as a plot
    Returns:
        interpolated_colors [list]: a list of all of the colors that make up the stepwise gradient
        fig [matplotlib figure]: visualization of the gradient
    """
    # Convert hex colors to RGB
    rgb_colors = [np.array(colors.to_rgb(color)) for color in hex_colors]
    interpolated_colors = []

    steps_between = math.ceil((total_steps-len(hex_colors)) / (len(hex_colors)-1)) # calculate how many colors in between each specified color
    # to end up with the specified number of total colors (or more)
    
    # Interpolate between each pair of colors
    for i in range(len(rgb_colors) - 1):
        start = rgb_colors[i]
        end = rgb_colors[i + 1]
        for t in np.linspace(0, 1, steps_between + 2)[:-1]:  # omit the last to avoid duplicates
            interpolated = (1 - t) * start + t * end
            interpolated_colors.append(colors.to_hex(interpolated))
    
    # Append the last color explicitly
    interpolated_colors.append(hex_colors[-1])

    if show_plot:
        fig, ax = plt.subplots(figsize=(12,2))
        ax.set_xlim(0, len(interpolated_colors))
        ax.set_ylim(0, 1)
        for i, color in enumerate(interpolated_colors):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
            ax.text(i + 0.5, -0.2, color, ha='center', va='top', fontsize=15, color='black', rotation=45)
        
        return interpolated_colors, fig
    else:
        return interpolated_colors

def custom_percent_formatter(x, pos):
    """
    Format a number as a percentage. Helper function for percent_diff_plot.
    """
    return f"{x * 100:.1f}%"

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

