# Purpose: plotting tools
# Author: Emily Bregou
# Depends on: matplotlib


import matplotlib as mpl
from matplotlib import pyplot as plt

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

