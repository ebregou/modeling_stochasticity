# Purpose: plotting tools
# Author: Emily Bregou
# Depends on: matplotlib, numpy, corner


import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import corner

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
        ax[0].plot(x, y1, label = labels[1], linestyle = '--')
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

    def plot_corner(self, samples, labels, true_vals = None):
        if true_vals is None:
            corner_plot = corner.corner(samples, labels=labels, color = self.red) 
        else:
            corner_plot = corner.corner(samples, labels=labels, color = self.red, truths = true_vals, truth_color=self.turquoise)
            
        return corner_plot

