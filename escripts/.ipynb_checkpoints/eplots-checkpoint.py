# Purpose: plotting tools
# Author: Emily Bregou
# Depends on: matplotlib, numpy, corner, estats

# Standard packages
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import corner
# Local packages
from escripts import estats

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
    
    def custom_percent_formatter(x, pos):
        """
        Format a number as a percentage. Helper function for percent_diff_plot.
        """
        return f"{x * 100:.1f}%"

    def plot_corner(self, samples, my_UVLF, true_vals = None):
        """
        Plot a corner plot, adding true values if any are given.
        Inputs:
            samples [array]: samples from the MCMC chain
            param_data [list of dicts]: metadata about parameters
            true_vals [1darray]: optional true values for each parameter
        Returns:
            corner_plot [matplotlib figure]: corner plot showing the values and covariances of each parameter, and, optionally, their true values
        """
        param_data = my_UVLF.param_data
        labels = [p['label'] for p in param_data if p.get('fit', True)]
        if true_vals is None:
            corner_plot = corner.corner(samples, labels=labels, color = self.red) 
        else:
            corner_plot = corner.corner(samples, labels=labels, color = self.red, truths = true_vals, truth_color=self.turquoise)
            
        return corner_plot

    def plot_evolving_UVLF_fit(self, samples, my_UVLF, data_label = 'Bouwens+21', nsamples = 100, truths = None, 
                               title = 'UVLF data vs. MCMC fit'):
        """
        samples [ndarray]: Results from MCMC
        my_UVLF: eMCMC UVLF object
        data_label [str]: what you want to appear as the label for the data points w/ error bars
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
        
        for i, (dat, z) in enumerate(zip(my_UVLF.data, zs)):
    
            # Choose random samples from the MCMC chain
            inds = np.random.randint(len(samples), size=nsamples) # Choose nsamples from the chain
        
            # Decompose the data
            zdat = dat[0]
            zerr = dat[1]
            xdat = dat[2]
            ydat = dat[3]
            yerr = dat[4] 
            xerr = dat[5]   
    
            yerr = np.fmax(yerr, ydat* my_UVLF.MINRELERROR) # Make sure error bars aren't any smaller than the relative error set above
    
            # Plot each sample from the chain
            for ind in inds:
                sample = samples[ind]
                ax[i].plot(xdat, np.log10(my_UVLF.UVLF_wrapper(zdat,zerr,xdat,xerr, sample)), alpha=nsamples/3500, color = self.red, 
                           linestyle = '-', zorder = 0)
        
            
            if truths is not None: # Plot the true curve if one is known. Otherwise, plot the best fit
                ax[i].plot(xdat, np.log10(my_UVLF.UVLF_wrapper(zdat,zerr,xdat,xerr, truths)), color = self.red, label = 'true UVLF', 
                           linestyle = '-', zorder = 1)
            else:
                best_fit = np.average(samples, axis = 0)
                ax[i].plot(xdat, np.log10(my_UVLF.UVLF_wrapper(zdat,zerr,xdat,xerr, best_fit)), color = self.red, label = f'best fit (average)', 
                           linestyle = '-', zorder = 1)
            
            ax[i].plot([0], [0], color = self.red, alpha = 0.5, label = 'sampled fits', linestyle = '-') # Create the label for the samples
            ax[i].errorbar(xdat, np.log10(ydat), estats.calc_log_error(ydat, yerr), fmt = ".", capsize=0, markersize = 10, color = self.navy, 
                           label = data_label)
            
            ax[i].invert_xaxis()
            ax[i].set_ylim(np.log10(np.min(ydat))-1, np.log10(np.max(ydat))+2)
            ax[i].set_xlim(np.max(xdat)+0.25, np.min(xdat)-0.25)
            ax[i].set_title(f'$z \simeq {int(zdat)}$')
            ax[i].xaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # Make it so that only integers can be used in the axis labels
    
        ax[-1].text(1.02, .95, 'Best fit parameter values:', transform=ax[-1].transAxes, va='top', fontsize = 10)
        for i, (label, val) in enumerate(zip([p['label'] for p in my_UVLF.param_data if p.get('fit', True)], best_fit)): 
            ax[-1].text(1.02, 0.95-(0.075*(i+1)), f'{label} : {round(val, 2)}', transform=ax[-1].transAxes, va='top', fontsize = 10)
                
        
        ax[0].set_ylabel(r'$\log_{10}(\phi_{\rm{UV}}$ [$\rm{mag}^{-1} \rm{Mpc}^{-3}$])')
        fig.text(.5, -0.05, r'$M_{\rm{UV}}$ [mag]', ha = 'center')
        fig.text(.5, 1.1, title, ha = 'center', fontsize = 20)
        ax[-1].legend()
    
        return fig

    def compare_fits(self, data, best_fits, fit_labels, UVLF_objects, data_label = 'Bouwens+21', title = 'Fit comparison'):
        """
        Purpose: compare different best fits againste each other
        Inputs:
            data [Ndarray]: data to plot against the fits
            best_fits [Ndarray]: list of best fit parameter values for different fits
            fit_labels [list of strings]: labels for these different fits (for the plot)
            UVLF_objects: list of eMCMC UVLF objects that correspond to the best fits. This is needed so that this function has access to the 
                          parameter data.
            data_label [str]: label assigned to the plotted data
            title [str]: overall title for the plot
        Outputs: Comparison figure
        """
        # Figure out how many subplots to make
        zs = [dat[0] for dat in data]
        nz = len(zs)
        fig, ax = plt.subplots(1, nz, sharey = True)
        fig.set_size_inches(3.5*nz, 3)
        plt.subplots_adjust(wspace = 0.1)
    
        for i, (dat, z) in enumerate(zip(data, zs)):
        
            # Decompose the data
            zdat = dat[0]
            zerr = dat[1]
            xdat = dat[2]
            ydat = dat[3]
            yerr = dat[4] 
            xerr = dat[5]   
    
            yerr = np.fmax(yerr, ydat* UVLF_objects[0].MINRELERROR) # Make sure error bars aren't any smaller than the relative error
            ax[i].errorbar(xdat, np.log10(ydat), estats.calc_log_error(ydat, yerr), fmt = ".", capsize=0, markersize = 10, color = self.navy, 
                           label = data_label)
    
            # Plot best fit:
            for sample, currUVLF, label in zip(best_fits, UVLF_objects, fit_labels): #, self.hex_colors[:len(UVLF_objects)]:
                chi2 = -2*currUVLF.log_like(sample, alt_data = data)
                ax[i].plot(xdat, np.log10(currUVLF.UVLF_wrapper(zdat,zerr,xdat,xerr, sample)), label = f'{label}, $\chi^2 = {chi2:.0f}$', 
                           zorder = 1)
            
            ax[i].invert_xaxis()
            ax[i].set_ylim(np.log10(np.min(ydat))-1, np.log10(np.max(ydat))+2)
            ax[i].set_xlim(np.max(xdat)+0.25, np.min(xdat)-0.25)
            ax[i].set_title(f'$z \simeq {int(zdat)}$')
            ax[i].xaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # Make it so that only integers can be used in the axis labels
    
        ax[0].set_ylabel(r'$\log_{10}(\phi_{\rm{UV}}$ [$\rm{mag}^{-1} \rm{Mpc}^{-3}$])')
        fig.text(.5, -0.05, r'$M_{\rm{UV}}$ [mag]', ha = 'center')
        fig.text(.5, 1.1, title, ha = 'center', fontsize = 20)
        leg = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
        return fig

# ------------------------------------------------------------------------------------------------------------------------------------------------

def interpolate_colors(colors, steps_between):
    """
    Create a step-wise gradient between each color in the colors list. There are [steps_between] colors between each color in the list.
    Inputs:
        colors [list]: colors in hexadecimal format
        steps_between [int]: number of steps between each color and the next in the list
    Returns:
        interpolated_colors [list]: a list of all of the colors that make up the stepwise gradient
        fig [matplotlib figure]: visualization of the gradient
    """
    # Convert hex colors to RGB
    rgb_colors = [np.array(to_rgb(color)) for color in hex_colors]
    interpolated_colors = []
    
    # Interpolate between each pair of colors
    for i in range(len(rgb_colors) - 1):
        start = rgb_colors[i]
        end = rgb_colors[i + 1]
        for t in np.linspace(0, 1, steps_between + 2)[:-1]:  # omit the last to avoid duplicates
            interpolated = (1 - t) * start + t * end
            interpolated_colors.append(to_hex(interpolated))
    
    # Append the last color explicitly
    interpolated_colors.append(hex_colors[-1])
    
    # Plot
    fig, ax = plt.subplots(figsize=(12,2))
    ax.set_xlim(0, len(interpolated_colors))
    ax.set_ylim(0, 1)
    for i, color in enumerate(interpolated_colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
        ax.text(i + 0.5, -0.2, color, ha='center', va='top', fontsize=15, color='black', rotation=45)
    
    return interpolated_colors, fig
    


