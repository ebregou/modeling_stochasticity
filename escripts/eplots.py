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

class Plotting():
    def __init__(self):
        color_cycler = mpl.rcParams['axes.prop_cycle']
        colors = [c['color'] for c in color_cycler]
        self.hex_colors = [mpl.colors.to_hex(color) for color in colors]
        self.red = self.hex_colors[0]
        self.turquoise = self.hex_colors[1]
        self.yellow = self.hex_colors[2]
        self.navy = self.hex_colors[3]
        self.orange = self.hex_colors[4]
        self.green = self.hex_colors[5]
        self.periwinkle = self.hex_colors[6]
        self.labels = ['red', 'turquoise', 'yellow', 'navy', 'orange', 'green', 'periwinkle']
        
    
    def plot_colors(self):
        """
        Plot the colors defined in my mplstyle sheet and print their names
        This gives a visual representation of all the colors to make it easier to choose the best one.
        """
        # Create a figure to visualize the colors
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.set_xlim(0, len(self.hex_colors))
        ax.set_ylim(0, 1)
        
        # Plot each color as a rectangle
        for i, (hex_color, label) in enumerate(zip(self.hex_colors, self.labels)):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=hex_color))
            ax.text(i + 0.5, -0.2, hex_color, ha='center', va='top', fontsize=15, color='black', rotation=45)
            print(f'{label}: {hex_color}')
        
        # Formatting
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        
        plt.show()
        return
    
    def percent_diff_plot(self, x, y0, y1, labels, other_data = None):
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
    
        ax[1].plot(x, percent_diff, color = self.navy)
        ax[1].yaxis.set_major_formatter(ticker.FuncFormatter(custom_percent_formatter)) # Format percent difference with % symbols
    
        return fig, ax

    def plot_corner(self, samples, UVLF, excluded_params = [], true_vals = None, title = ''):
        """
        Plot a corner plot, adding true values if any are given.
        Inputs:
            samples [array]: samples from the MCMC chain
            UVLF: UVLF object
            excluded_params [list of strings]: any parameters that you don't want plotted in the corner plot
            true_vals [1darray]: optional true values for each parameter. Don't include true values for parameters in excluded_params
            title [str]: plot title
        Returns:
            corner_plot [matplotlib figure]: corner plot showing the values and covariances of each parameter, and, optionally, their true values
        """
        plot_labels = np.array([param['label'] for name, param in UVLF.param_data.items() if param['fit'] is True])
                  
        # Keep only the desired columns (so that we only plot parameters not in excluded_params)
        names = [outer_key for outer_key, inner_dict in UVLF.param_data.items() if inner_dict.get('fit', True)]
        keep = [i for i, name in enumerate(names) if name not in excluded_params]
        cut_samples = np.array([[row[i] for i in keep] for row in samples])
        
        if true_vals is None:
            corner_plot = corner.corner(cut_samples, labels=plot_labels[keep], color = self.red) 
        else:
            corner_plot = corner.corner(cut_samples, labels=plot_labels[keep], color = self.red, truths = true_vals, truth_color=self.turquoise)

        corner_plot.suptitle(title, y = 1.02)
            
        return corner_plot

    def plot_evolving_UVLF_fit(self, samples, my_UVLF, nsamples = 100, truths = None, title = 'UVLF data vs. MCMC fit'):
        """
        samples [ndarray]: Results from MCMC
        my_UVLF: eMCMC UVLF object
        nsamples [int]: number of samples you want to appear alongside the best fit on the plot
        truths [ndarray]: true values of parameters, if they exist
        title [str]: overarching title of the plot
        """
        # Figure out how many subplots to make
        zs = [dat[0] for dat in my_UVLF.data]
        nz = len(zs)
        fig, ax = plt.subplots(1, nz, sharey = True)
        if nz == 1:
            ax = [ax]  # Make ax iterable even if it's just one subplot
        fig.set_size_inches(3.5*nz, 3)
        plt.subplots_adjust(wspace = 0.1)

        # Set the same y limits for all the plots
        ylo, yhi = np.log10(min([min(dat[3]) for dat in my_UVLF.data]))-0.25, np.log10(max([max(dat[3]) for dat in my_UVLF.data]))+0.25 

        plot_xdat, plot_xerr = my_UVLF.data[0][2], my_UVLF.data[0][5], # Use the lowest redshift MUV centers & bins to calculate the theoretical
                                                                    # UVLF (this will make it the smoothest)
        
        for i, (zbin, z) in enumerate(zip(my_UVLF.sorted_data, zs)):
    
            # Choose random samples from the MCMC chain
            inds = np.random.randint(len(samples), size=nsamples) # Choose nsamples from the chain
        
            zdat, zerr = zbin[0][0], zbin[0][1]
    
            # Plot each sample from the chain
            for ind in inds:
                sample = samples[ind]
                ax[i].plot(plot_xdat, np.log10(my_UVLF.UVLF_wrapper(zdat,zerr,plot_xdat, plot_xerr,sample)), alpha=nsamples/3500, color = self.red, 
                           linestyle = '-', zorder = 0)
        
            
            if truths is not None: # Plot the true curve if one is known. Otherwise, plot the best fit
                ax[i].plot(plot_xdat, np.log10(my_UVLF.UVLF_wrapper(zdat,zerr,plot_xdat,plot_xerr, truths)), color = self.red, label = 'true UVLF', 
                           linestyle = '-', zorder = 1)
            else:
                best_fit = np.average(samples, axis = 0)
                ax[i].plot(plot_xdat, np.log10(my_UVLF.UVLF_wrapper(zdat,zerr,plot_xdat,plot_xerr,best_fit)), color = self.red, 
                           label = f'best fit (average)', linestyle = '-', zorder = 1)
            
            ax[i].plot([0], [0], color = self.red, alpha = 0.5, label = 'sampled fits', linestyle = '-') # Create the label for the samples
            for dat, fmt in zip(zbin, ['o', 'v', 's']):
                _, _, xdat, ydat, yerr_upper, yerr_lower, xerr = eMCMC.decompose_data(dat)
                yerr_lower, yerr_upper = estats.calc_log_error(ydat, yerr_lower), estats.calc_log_error(ydat, yerr_upper)
                ax[i].errorbar(xdat, np.log10(ydat), [yerr_lower, yerr_lower], fmt = fmt, capsize=0, markersize = 9, label = dat[7], 
                               markeredgecolor = 'none', markerfacecolor = self.navy, ecolor = self.navy)
            
            ax[i].invert_xaxis()
            ax[i].set_ylim(ylo, yhi)
            ax[i].set_xlim(np.max(plot_xdat)+0.25, np.min(plot_xdat)-0.25)
            ax[i].set_title(f'$z \simeq {int(zdat)}$')
            ax[i].xaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # Make it so that only integers can be used in the  
                                                                                           # axis labels
    
        ax[-1].text(1.02, .95, 'Best fit parameter values:', transform=ax[-1].transAxes, va='top', fontsize = 10)
        for i, (label, val) in enumerate(zip([p['label'] for p in list(my_UVLF.param_data.values()) if p.get('fit', True)], best_fit)): 
            ax[-1].text(1.02, 0.95-(0.075*(i+1)), f'{label} : {round(val, 2)}', transform=ax[-1].transAxes, va='top', fontsize = 10)
                
        
        ax[0].set_ylabel(r'$\log_{10}(\phi_{\rm{UV}}$ [$\rm{mag}^{-1} \rm{Mpc}^{-3}$])')
        fig.text(.5, -0.05, r'$M_{\rm{UV}}$ [mag]', ha = 'center')
        fig.text(.5, 1.1, title, ha = 'center', fontsize = 20)
        ax[-1].legend()
    
        return fig

    def compare_fits(self, file_names, data_labels, best_fits, fit_labels, title = 'Fit comparison'):
        """
        Purpose: compare different best fits against each other
        Inputs:
            file_names [list of strs]: list of files where data is read from
            data_labels [list of strs]: legend keys for different datasets
            best_fits [list of lists]: list of best fit parameter values for different fits
            fit_labels [list of strings]: labels for these different fits (for the plot)
            title [str]: overall title for the plot
        Outputs: Comparison figure
        """
        my_UVLF = eMCMC.UVLF(file_names, data_labels)
        data = my_UVLF.sorted_data
        # Figure out how many subplots to make
        zs = [dat[0] for dat in data]
        nz = len(zs)
        fig, ax = plt.subplots(1, nz, sharey = True)
        if nz == 1:
            ax = [ax] # Make iterable
        fig.set_size_inches(3.5*nz, 3)
        plt.subplots_adjust(wspace = 0.1)

        # Set the same y limits for all the plots
        ylo, yhi = np.log10(min([min(dat[3]) for dat in my_UVLF.data]))-0.25, np.log10(max([max(dat[3]) for dat in my_UVLF.data]))+0.25

        plot_xdat, plot_xerr = my_UVLF.data[0][2], my_UVLF.data[0][5], # Use the lowest redshift MUV centers & bins to calculate the theoretical
                                                                    # UVLF (this will make it the smoothest)
    
        for i, (zbin, z) in enumerate(zip(data, zs)):
            zdat, zerr = zbin[0][0], zbin[0][1]
            
    
            # Plot best fit:
            for sample, label, color, ls in zip(best_fits, fit_labels, [self.red, self.turquoise, self.yellow, self.orange], 
                                                ['solid', 'dashdot', 'dashed', 'dotted']): 
                chi2 = -2*my_UVLF.log_like(sample, alt_data = my_UVLF.data)
                ax[i].plot(plot_xdat, np.log10(my_UVLF.UVLF_wrapper(zdat,zerr,plot_xdat,plot_xerr,sample)), 
                           label = f'{label}, $\chi^2 = {chi2:.0f}$', zorder = 1, color = color, linestyle = ls)

            for dat, fmt in zip(zbin, ['o', 'v', 's']):
                _, _, xdat, ydat, yerr_upper, yerr_lower, xerr = eMCMC.decompose_data(dat)
                yerr_lower, yerr_upper = estats.calc_log_error(ydat, yerr_lower), estats.calc_log_error(ydat, yerr_upper)
                ax[i].errorbar(xdat, np.log10(ydat), [yerr_lower, yerr_lower], fmt = fmt, capsize=0, markersize = 9, label = dat[7], 
                               markeredgecolor = 'none', markerfacecolor = self.navy, ecolor = self.navy)
            
            ax[i].invert_xaxis()
            ax[i].set_ylim(ylo, yhi)
            ax[i].set_xlim(np.max(plot_xdat)+0.25, np.min(plot_xdat)-0.25)
            ax[i].set_title(f'$z \simeq {int(zdat)}$')
            ax[i].xaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # Make it so that only integers can be used in the  
                                                                                           # axis labels
        ax[0].set_ylabel(r'$\log_{10}(\phi_{\rm{UV}}$ [$\rm{mag}^{-1} \rm{Mpc}^{-3}$])')
        fig.text(.5, -0.05, r'$M_{\rm{UV}}$ [mag]', ha = 'center')
        fig.text(.5, 1.1, title, ha = 'center', fontsize = 20)
        leg = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
        return fig

# ------------------------------------------------------------------------------------------------------------------------------------------------

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

