# Purpose: plotting tools
# Author: Emily Bregou
# Depends on: matplotlib, numpy


import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def plot_colors():
    """
    Plot the colors defined in my mplstyle sheet and print their names
    This gives a visual representation of all the colors to make it easier to choose the best one.
    """

    # Get the default color cycle from rcParams
    color_cycler = mpl.rcParams['axes.prop_cycle']
    colors = [c['color'] for c in color_cycler]
    
    # Create a figure to visualize the colors
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    
    # Plot each color as a rectangle
    for i, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
        hex_color = mpl.colors.to_hex(color)
        ax.text(i + 0.5, -0.2, hex_color.replace("#", ""),  # Remove '#' symbol
            ha='center', va='top', fontsize=10, color='black', rotation=45)
        print(hex_color)
    
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
    ax[0].plot(x, y1, label = labels[1], linestyle = '--')
    if other_data is not None:
        assert len(other_data) > 0, 'other_data must be given as a list or array'
        [ax[0].plot(x, dat, label = lab, linestyle = 'dotted') for (dat, lab) in zip(other_data, labels[2:])]
        
    ax[0].legend()

    ax[1].plot(x, percent_diff, color = '#294450')
    ax[1].yaxis.set_major_formatter(ticker.FuncFormatter(custom_percent_formatter)) # Format percent difference with % symbols

    return fig, ax

def custom_percent_formatter(x, pos):
    """
    Format a number as a percentage. Helper function for percent_diff_plot.
    """
    return f"{x * 100:.1f}%"

