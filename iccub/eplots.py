
# def plot_residuals(residuals, x_grid, y_grid, linthresh=1e7, linscale=1, levels = 100, xlim=None, ylim=None, xlabel = 'x (kpc)', 
#                    ylabel = 'y (kpc)', cmap = 'PuOr'):
#     """
#     Plots density residuals when given mass residuals and a grid.
    
#     Parameters:
#         residuals [1darray]: binned mass residuals (mass_simulation - mass_model)
#         x_grid, y_grid [1darray, 1darray]: bins in x and y used for mass calculations
#         linthresh [float]: for SymLogNorm colorbar, the threshold for linear color scaling
#         linscale [float]: for SymLogNorm colorbar, the relative scale of the linear color scaling
#         levels [int]: how many contours to plot
#         xlim, ylim [float, float]: x and y limits for the plots -- will be set as (-xlim, xlim)
#         xlabel, ylabel (str): x and y axis labels
#         cmap (str): colormap for contour plot
#     Returns:
#         Figure showing residuals, matplotlib color normalization
#     """

#     # Calculate the area of a cell in the grid
#     box_len_x = x_grid[1]-x_grid[0]
#     box_len_y = y_grid[1]-y_grid[0]
#     box_area = box_len_y*box_len_x
    
#     # Get density residuals
#     density_residuals = residuals/box_area

#     # Calculate points in the center of each grid cell, for use when plotting
#     x_plot = (x_grid[1:]+x_grid[:-1])/2
#     y_plot = (y_grid[1:]+y_grid[:-1])/2

#     # Set colorbar parameters
#     vmin = np.min([np.min(density_residuals), -np.max(density_residuals)])
#     vmax = np.max([np.max(density_residuals), -np.min(density_residuals)])
#     norm = colors.SymLogNorm(linthresh = linthresh, linscale = linscale, vmin = vmin, vmax = vmax) 
#     # In the middle of the colorbar is a linear section whose extent is defined by linthresh. Then, the colorbar goes logarithmic at values more
#     # extreme. linscale sets how much space the linear part of the colorbar takes up as compared to the logarithmic part.
    
#     # Plot contours
#     fig, ax = plt.subplots()
#     ax.contourf(x_plot,y_plot,density_residuals.T, cmap=cmap, levels = levels, norm=norm)

#     # Add colorbar
#     cax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
#     cbar = fig.colorbar(colormap.ScalarMappable(norm=norm, cmap=cmap), cax, orientation = 'vertical')
#     cbar.set_label(label= r'Residuals, $\frac{M_{sim} - M_{model}}{area}$ ($\frac{M_\odot}{kpc^2}$)', size = 15)

#     # Set x and y limits
#     if xlim is not None:
#         ax.set_xlim(-xlim, xlim)
#     if ylim is not None:
#         ax.set_ylim(-ylim, ylim)

#     # Set axis labels
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)

#     return fig, norm

def summary_plot(grids, sim_mass_bins, model_mass_bins, galactic_pot, orientation = 'xy', residual_norm = None, contour_levels = 100, residual_levels = 50, 
                 linthresh=1e7, linscale=0.15, residual_bins = 20, limit = None):
    """
    
    """
    # """
    # 2x2 plot including simulation contours, model contours, residual histogram, and residual contours.

    # Parameters:
    #     grids [list of 3 1darrays]: positions of the bins of the 2dhistogram in x, y, z (each array corresponds to a coordinate)
    #     sim_mass_bins, model_mass_bins [list of 3 1darrays]: binned masses for each of the three coordinate pairs (x, y) (y, z) (x, z)
    #     orientation [str]: One of: 'xy', 'yz', 'xz' describing the orientation of the resulting plot
    #     residual_norm [matplotlib colormap]: normalization for plotting the residuals. It is highly encouraged to include this if you want
    #                                          control over the appearance of your plot. You can use eplots.plot_residuals to tweak residual_norm.
    #     contour_levels, residual_levels [int, int]: number of contour levels for the density plots and residual plot
    #     linthresh, linscale [float, float]: values defining the appearance of the colorbar (see matplotlib.colors.SymLogNorm)
    #     residual_bins [int]: number of bins for the 1d residual histogram
    #     limit [float]: x_lim and y_lim for plot
    # Returns:
    #     Summary plot including contours for the mass in the simulation and the model, histogram of the values of residuals, and contour 
    #     plot showing the spatial distribution of residuals.
    # """

    # assert orientation in ['xy', 'yz', 'xz'], 'orientation must be one of: xy, yz, zx'

    # # Assign variables depending on the orientation of the final plot
    # if orientation == 'xy':
    #     (coord_1, coord_2) = (0, 1)
    #     (x_label, y_label) = ('x (kpc)', 'y (kpc)')

    # elif orientation == 'yz':
    #     (coord_1, coord_2) = (1, 2)
    #     (x_label, y_label) = ('y (kpc)', 'z (kpc)')

    # else:
    #     (coord_1, coord_2) = (0, 2)
    #     (x_label, y_label) = ('x (kpc)', 'z (kpc)')

    # # Calculate residuals
    # residuals = sim_mass_bins[coord_1] - model_mass_bins[coord_1]

    # # Calculate residual norm if not given
    # if residual_norm is None:
    #     print('Plotting residuals since residual norm was not provided')
    #     residual_fig, residual_norm = plot_residuals(residuals, grids[coord_1], grids[coord_2])

    # # Calculate the area of each grid cell
    # box_len_x = grids[coord_1][1]-grids[coord_1][0]
    # box_len_y = grids[coord_2][1]-grids[coord_2][0]
    # box_area = box_len_y*box_len_x

    # # Calculate the center of boxes for plotting
    # x_plot = (grids[coord_1][1:]+grids[coord_1][:-1])/2
    # y_plot = (grids[coord_2][1:]+grids[coord_2][:-1])/2

    # # Plot
    # fig,ax = plt.subplots(2,2)
    # fig.set_size_inches(10,10)
    # plt.subplots_adjust(wspace=0.25, hspace=0.25)

    # # Plot the mass contours for the simulation and the model
    # contour_norm = colors.SymLogNorm(linthresh = linthresh, linscale = linscale, vmin = np.min(sim_mass_bins[coord_1]/box_area), 
    #                               vmax = np.max(sim_mass_bins[coord_1]/box_area)) # Set scale for colorbar
    # contour_plot = ax[0,0].contourf(x_plot,y_plot,(sim_mass_bins[coord_1]/box_area).T, cmap='cividis', levels = contour_levels, 
    #                                 norm = contour_norm)
    # ax[0,1].contourf(x_plot,y_plot,(model_mass_bins[coord_1]/box_area).T, cmap='cividis', levels = contour_levels, norm = contour_norm)
    
    # # Set axis labels
    # ax[0,1].set_ylabel(y_label)
    # ax[0,1].set_xlabel(x_label)
    # ax[0,1].set_title('Mass sampled from model')
    # ax[0,0].set_ylabel(y_label)
    # ax[0,0].set_xlabel(x_label)
    # ax[0,0].set_title('Mass from simulation')
    
    # # Make colorbar
    # cax_contour = fig.add_axes([0.93, 0.524, 0.016, 0.36])
    # cb_contour = plt.colorbar(colormap.ScalarMappable(norm=contour_norm, cmap='cividis'), cax = cax_contour)
    # cb_contour.set_label(label= r'Density ($\frac{M_\odot}{kpc^2}$)', size = 15)

    # # Plot the 1d residual histogram
    # residual_hist = ax[1,0].hist((residuals/box_area).reshape(-1,1), bins = residual_bins)
    # ax[1,0].set_yscale('log')
    # ax[1,0].set_title('Residuals')
    # ax[1,0].set_xlabel(r'Residuals, $\frac{M_{sim} - M_{model}}{area}$ ($\frac{M_\odot}{kpc^2}$)')

    # # Plot the residual contours
    # ax[1,1].contourf(x_plot,y_plot,(residuals/box_area).T, cmap='PuOr', norm=residual_norm, levels = residual_levels) # Set scale for colorbar
    # # Set axis labels
    # ax[1,1].set_ylabel(y_label)
    # ax[1,1].set_xlabel(x_label)
    # ax[1,1].set_title(r'Residual contours')
    # # Make colorbar
    # cax = fig.add_axes([0.93, 0.098, 0.016, 0.36])
    # cb = plt.colorbar(colormap.ScalarMappable(norm=residual_norm, cmap='PuOr'), cax, label = '')
    # cb.set_label(label= r'Residuals, $\frac{M_{sim} - M_{model}}{area}$ ($\frac{M_\odot}{kpc^2}$)', size = 15)

    # # Set x and y limits
    # if limit is not None:
    #     for ax_i in [ax[0,0], ax[0,1], ax[1,1]]:
    #         ax_i.set_xlim(-limit, limit)
    #         ax_i.set_ylim(-limit, limit)

    # return fig

def plot_enclosed_mass(radii, potential, particles, title = '', threshold = 0.05):
    """
    Makes a plot comparing the enclosed mass in a potential model with the true enclosed mass from the simulation. Calculate radius at which percent
    difference between the true and simulated mass is crossed (percent difference is typically greatest at the center)
    
    Parameters:
        radii [1darray]: Array of radii to check the enclosed mass at
        potential: Agama potential model
        particles [dict]: keys 'POS ' and 'MASS' corresponding to particle position & mass in the simulation.
        title [str]: Title of plot
        threshold [float]: Percent difference threshold between true and simulated enclosed mass. Function will return the radius at which this 
                           threshold is crossed
    Returns:
        Figure comparing the enclosed mass and the percent difference of enclosed mass
    """
    print('there is a more updated version of this under potential_visualization.ipynb')
    
    # Get model & real enclosed mass
    model_enc_mass = potential.enclosedMass(radii)
    real_enc_mass = [np.sum(particles['MASS'][np.where(np.linalg.norm(particles['POS '], axis = 1) <= radius)]) for radius in radii]
    
    # Plot enclosed mass
    fig, ax1 = plt.subplots()
    ax1.plot(radii, real_enc_mass, label = 'simulation')
    ax1.plot(radii, model_enc_mass, linestyle = '--', label = 'model')
    ax1.set_xscale('log')
    ax1.set_xlabel('radius (kpc)')
    ax1.set_ylabel(r'enclosed mass ($M_\odot$)')
    plt.legend()

    # Plot percent difference
    diff = 100*(real_enc_mass - model_enc_mass) / real_enc_mass
    ax2 = ax1.twinx() # Make this share the x axis
    color = 'tab:red'
    ax2.plot(radii, diff, color = color)
    ax2.set_ylabel('% difference', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Figure settings
    plt.title(title)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # Calculate where the percent difference crosses given threshold
    intersection_point = np.where(np.abs(diff) <  threshold)[0][0]
    intersection_radius = radii[intersection_point]

    return fig, intersection_radius

def plot_residual_percentage(sim_mass_bins, model_mass_bins, grids, title = '', colorbar_lim = 1, levels = 75, r = 100):
    """
    Plots surface density residuals (p_sim - p_model / p_sim) as a percentage.
    Paremeters:
        sim_mass_bins, model_mass_bins [2darray, 2darray, 2darray]: contain the binned masses from the simulation and model
        grid [1darray, 1darray, 1darray]: the x, y, and z grids that gave rise to the binned model mass
        title [str]: title of the plot
        colorbar_lim [float]: Minimum / maximum percentage for colorbar (will be set as (-colorbar_lim, colorbar_lim)
        levels [int]: levels for contour plot-- for readability, it's important to set levels high enough that a total of 6 colors are visible on
                      the colorbar.
        r (float): radius for circular plotting. Can be set to None if you don't want to do circular plotting.
    Returns:
        fig-- Contour plot, custom colorbar.
    """

    # Calculate percent difference; put nan where there is no data to ensure there are no divide by zero errors
    percent_differences = [np.divide((sim_mass_i-model_mass_i), sim_mass_i, out=np.full_like(sim_mass_i, np.nan), where=sim_mass_i!=0)
                          for (sim_mass_i, model_mass_i) in zip(sim_mass_bins, model_mass_bins)]
    
    # Workaround since the 'extend' option in the colorbars is not working with contourf as far as I can tell
    percent_differences = [np.where(percent_difference < -colorbar_lim, -colorbar_lim - 0.01, percent_difference) 
                           for percent_difference in percent_differences] 
    # So I'll just set the out of range values myself
    percent_differences = [np.where(percent_difference > colorbar_lim, colorbar_lim + 0.01, percent_difference) 
                           for percent_difference in percent_differences]
    
    # Plot contours
    fig, ax = plt.subplots(1, 3, subplot_kw = {'aspect':'equal'})
    fig.set_size_inches(12, 4)
    plt.subplots_adjust(wspace=0.35)

    # Colormap
    cmap = 'RdBu'
    norm = colors.TwoSlopeNorm(vcenter = 0, vmin = -colorbar_lim, vmax = colorbar_lim)

    # Plot each point in the center of the bins
    plot_grid = [(grid[1:] + grid[:-1])/2 for grid in grids]

    # Plot
    colorplot_xy = ax[0].contourf(plot_grid[0], plot_grid[1], percent_differences[0].T, cmap=cmap, levels = levels, norm = norm, extend = 'both')
    colorplot_yz = ax[1].contourf(plot_grid[1], plot_grid[2], percent_differences[1].T, cmap=cmap, levels = levels, norm = norm, extend = 'both')
    colorplot_zx = ax[2].contourf(plot_grid[0], plot_grid[2], percent_differences[2].T, cmap=cmap, levels = levels, norm = norm, extend = 'both')

    # Colorbar
    cax = fig.add_axes([0.925, 0.145, 0.02, 0.695])
    fig.colorbar(colorplot_xy, cax = cax, extend = 'both', label = r'$\Delta \rho / \rho$', 
                 format = PercentFormatter(xmax=1))

    # Mask out everything not within a circle of radius r from 0,0
    if r is not None:
        circles = [Circle((0, 0), r_i, facecolor='none') for r_i in [r, r, r]]
        [ax_i.add_patch(circle_i) for (ax_i, circle_i) in zip(ax, circles)]
        [colorplot.set_clip_path(circle_i) for (colorplot, circle_i) in zip([colorplot_xy, colorplot_yz, colorplot_zx], circles)]
    else:
        r = 100 # Default so that we set x and y limits without error.

    # Set x and y limits for each suplot
    xlim = 1.1*r
    ylim = 1.1*r
    [ax_i.set_xlim(-xlim, xlim) for ax_i in ax]
    [ax_i.set_ylim(-ylim, ylim) for ax_i in ax]

    # Labels & title
    ax[0].set_xlabel('x (kpc)')
    ax[0].set_ylabel('y (kpc)')
    ax[1].set_xlabel('y (kpc)')
    ax[1].set_ylabel('z (kpc)')
    ax[2].set_xlabel('x (kpc)')
    ax[2].set_ylabel('z (kpc)')
    ax[1].set_title(title, fontsize = 18, pad = 18)

    return fig

def order_vs_density_plot(snapshot, lmax_list, mmax_list, type, num_bins = 200, min_r = 1e-1, max_r = None, potentials = None, inc_title = True):
    """
    Create a figure similar to Figure 2 in Sanders et al., 2020-- https://doi.org/10.1093/mnras/staa3079 
    Parameters:
        snapshot [int]: simulation snapshot
        lmax_list, mmax_list [list, list]: list of maximum order of spherical, azimuthal expansion
        type [str]: type of model-- sph-- model everything spherically; disc_cyl-- model the disc cylindrically & everything else 
                    spherically; disc_gas_cyl: model the disc & cold gas cylindrically and everything else spherically
        num_bins [int]: number of bins for calculating the density
        max_r [float]: maximum radius for plot
        potentials [list of agama potential objects]: list of potentials to be plotted (optional)
        inc_title [bool]: whether to title the plot or not
    Returns:
        Figure-- true vs. model density in top panel; percent difference in bottom panel
    """

    print('this is now out of date & needs to be cross-compared with the new version in potential_visualization.ipynb')
    return
    
    # if potentials is None:
    #     # Create potentials for each lmax, mmax expansion pair
    #     potentials = [BFE.get_pot_model(snapshot, lmax = lmax, mmax = mmax, type = type) for lmax, mmax 
    #                   in zip(lmax_list, mmax_list)]

    # # Calculate simulation radial density
    # rot_DM, rot_stars, in_disc, rot_gas = read_snapshot.get_aligned_particles(snapshot) # Get particles
    # combined_particles = {'POS ': np.vstack((rot_DM['POS '], rot_stars['POS '], rot_gas['POS '])),
    #                      'MASS': np.concatenate((rot_DM['MASS'], rot_stars['MASS'], rot_gas['MASS']))} # Combine into one dictionary
    # radial_bins, sim_density = calc_sim_density(combined_particles, num_bins, min_r, max_r) # Calculate density

    # # Calculate model radial density for each potential model
    # N_part = len(combined_particles['MASS']) # Number of true particles (used to sample the model)
    # model_density = [calc_model_density(pot_model, N_part, radial_bins) for pot_model in potentials]

    # # Plot all expansions
    # avg_r = (radial_bins[1:] + radial_bins[:-1]) / 2 # Get average radius of spherical shells for plotting
    # num_exp = len(lmax_list)
    # fig = plt.figure(figsize=(5 * num_exp, 7))
    # ax_top = [plt.subplot2grid((2, num_exp), (0, i), fig=fig) for i in range(num_exp)] # Create top row of subplots
    # ax_btm = [plt.subplot2grid((2, num_exp), (1, i), fig=fig) for i in range(num_exp)] # Create bottom row of subplots

    # # Set maximum and minimum for the percent difference axes
    # # This is so that all of the subplots on the bottom row share bottom axes.
    # max = 0
    # min = 100
    # for i in range(num_exp):
    #     percent_diff = np.abs(sim_density - model_density[i])/sim_density
    #     if np.nanmax(percent_diff[np.isfinite(percent_diff)]) > max:
    #         max = np.nanmax(percent_diff[np.isfinite(percent_diff)])
    #     if np.nanmin(percent_diff[np.isfinite(percent_diff)]) < min: 
    #         min = np.nanmin(percent_diff[np.isfinite(percent_diff)])

    # for i in range(num_exp):
    #     # Plot density vs. radius on the top subplot of each column
    #     ax_top[i].plot(avg_r, sim_density, label='simulation')
    #     ax_top[i].plot(avg_r, model_density[i], linestyle = '--', label = 'model')

    #     # Plot percent difference between sim and model in the bottom subplot of each column
    #     ax_btm[i].plot(avg_r, np.abs(sim_density - model_density[i])/sim_density)
    #     ax_btm[i].set_ylim((min, max))
            

    #     # Set plot properties-----
    #     # Set logscale
    #     ax_top[i].set_xscale('log')
    #     ax_top[i].set_yscale('log')
    #     ax_btm[i].set_xscale('log')
    #     ax_btm[i].set_yscale('log')

    #     # Set x label
    #     ax_btm[i].set_xlabel(r'$r$ / kpc')

    #     if inc_title:
    #         # Set axis title
    #         ax_top[i].set_title(fr'$l_{{max}} = {lmax_list[i]}$, $m_{{max}} = {mmax_list[i]}$')

    # # Add y labels on the left
    # ax_top[0].set_ylabel(r'$\rho (r)$ / $M_\odot kpc^-3$')
    # ax_btm[0].set_ylabel(r'$| \Delta \rho / \rho |$')

    # # Add legend on the right
    # ax_top[-1].legend()

    # # Set overall title
    # if inc_title:
    #     halo = read_snapshot.read_halo(snapshot)
    #     plt.suptitle(fr'$z = {round(halo["z"], 2)}$; type = {type}', fontsize=18)

    # return fig

def calc_model_density(pot_model, N_part, radial_bins):
    """
    I think this should be rolled into calc_sim_density & calc_sim_density should be renamed
    """
    # """
    # Helper function to calculate the density as a function of radius from a potential model
    # Parameters:
    #     pot_model [agama potential]: potential model
    #     N_part [int]: number of particles to sample from model
    #     radial_bins [1darray]: list of radial bins to use in the histogram (get radial bins from read_subhalos.calc_sim_density to ensure they're 
    #                             the same)
    # Returns:
    #     model_density [1darray]: model density as a function of radius
    # """
    
    # # Sample the potential model to get masses
    # model_pos, model_mass = pot_model.sample(N_part)

    # # Calculate the radii of particles
    # model_radii = np.linalg.norm(model_pos, axis = 1)

    # # Create radial histogram of masses
    # model_mass_bins = np.histogram(model_radii, bins = radial_bins, weights = model_mass)[0]

    # # Calculate density
    # shell_volume = (4/3) * np.pi * (radial_bins[1:]**3 - radial_bins[:1]**3)
    # model_density = model_mass_bins / shell_volume

    # return model_density

def plot_orbit_3D(orbits, orbit_names, cmap = plt.cm.viridis):
    """
    Compare true and integrated orbit in 3D
    Parameters:
        orbits [list or dict]: list of dictionaries (or single dictionary) corresponding to orbits. Dictionaries should include 'TIME' (time in 
                               Gyr) and 'POS ' (position over time)
        orbit_names [list]: list of labels corresponding to the orbits
        cmap [matplotlib.cm colormap]: colormap to assign colors to orbits
    Returns:
        Figure comparing the orbits, titled to indicate how long the orbit integration was performed for
    """
    print('THERE IS A BETTER VERSION OF THIS IN SEMINAR FIGURES NOTEBOOK')
    # Formatting
    if type(orbits) != list:
        orbits = [orbits]
    
    # Figure settings
    fig, ax = plt.subplots(1, 3)
    plt.subplots_adjust(wspace = 0.35)
    fig.set_size_inches(12, 3.5)

    # Assign a different color to each orbit
    colors = [cmap(i) for i in np.linspace(0, 1, len(orbits))]

    # Assign a different linestyle to each type of orbit
    linestyles = ['solid', '--', 'dotted', 'dashdot']

    for i, (orbit, label, color) in enumerate(zip(orbits, orbit_names, colors)):
        # Set the linestyle to cycle around the list given
        linestyle = linestyles[i % len(linestyles)]
        
        # Plot orbit
        ax[0].plot(orbit['POS '][:,0], orbit['POS '][:,1], color = color, linestyle = linestyle)
        ax[1].plot(orbit['POS '][:,1], orbit['POS '][:,2], color = color, linestyle = linestyle)
        ax[2].plot(orbit['POS '][:,0], orbit['POS '][:,2], color = color, linestyle = linestyle, label = label)
    
    # Legend
    ax[2].legend()

    # Axis labels
    ax[0].set_xlabel('x(kpc)')
    ax[0].set_ylabel('y(kpc)')
    ax[1].set_xlabel('y(kpc)')
    ax[1].set_ylabel('z(kpc)')
    ax[2].set_xlabel('x(kpc)')
    ax[2].set_ylabel('z(kpc)')

    # Plot title to indicate how long orbit integration was performed for
    plt.suptitle(f'{round(float(orbits[0]["TIME"][-1]-orbits[0]["TIME"][0]), 2)} Gyr orbit integration')

    return fig

def plot_orbital_radius(all_orbits, orbit_names, plot_vr = True, cmap = plt.cm.viridis):
    """
    Plots radius over time for different types of orbits
    Parameters:
        all_orbits [list]: list of different types of orbits, e.g. [true_orbits, int_orbits]
        orbit_names [list]: list of strings describing the orbits
        plot_vr [bool]: whether or not to plot the virial radius along with the orbits
        cmap [matplotlib.cm]: colormap (for color-coding different orbits)
    Returns:
        Figure showing the radius over time for different types of orbits (each orbit has a different color & each type of orbit has a different
        linestyle.
    """
    print('THERE IS A BETTER VERSION OF THIS IN SEMINAR FIGURES NOTEBOOK')
    assert all([type(orbit) == list for orbit in all_orbits]), 'each orbit type must be given in the form of a list'

    # Set up plot
    fig, ax = plt.subplots()
    fig.set_size_inches(12,8)

    # Assign a different color to each orbit
    colors = [cmap(i) for i in np.linspace(0, 1, len(all_orbits[0]))]

    # Assign a different linestyle to each type of orbit
    linestyles = ['solid', '--', 'dotted', 'dashdot']

    # Plot radii over time, color-coding by orbit
    for j in range(len(all_orbits)):
        for i in range(len(all_orbits[0])):
            ax.plot(all_orbits[j][i]['TIME'], np.linalg.norm(all_orbits[j][i]['POS '], axis = 1), color = colors[i], 
                    linestyle = linestyles[j])

    # Legend (done in this weird way so that we can just show that solid means true & dashed means integrated without having a key on the
    # legend for each color)
    lines = [Line2D([0], [0], color = 'k', linestyle = ls) for ls in linestyles[:len(all_orbits)]]
    
    # Plot virial radius & include it in the legend
    if plot_vr:
        virial_radius = read_snapshot.get_virial_radius(all_orbits[0][0]['SNAP'])
        plt.plot(all_orbits[0][0]['TIME'], virial_radius, color = 'red')
        lines.append(Line2D([0],[0], color = 'red'))
        orbit_names.append('virial radius')
    
    plt.legend(lines, orbit_names)

    # Title based on how long orbit integration was performed
    int_time = all_orbits[0][0]['TIME'][-1] - all_orbits[0][0]['TIME'][0]
    plt.title(f'{round(float(int_time), 2)} Gyr orbit integration')

    # Axis labels
    ax.set_ylabel('radius (kpc)')
    ax.set_xlabel('time (Gyr)')

    return fig

def plot_velocity_mag(true_orbit, int_orbit):
    """
    Compare the magnitude of velocity over time between true & integrated orbits
    Parameters:
        true_orbit [dict]: orbit read from simulation data. Includes 'TIME' (time in Gyr) and 'POS ' (position over time in physical kpc.
        int_orbit [dict]: orbit reconstructed via orbit integration. Includes 'TIME' (time in Gyr) and 'POS ' (position over time in physical 
                          kpc.
    Returns:
        Figure with magnitude of velocity over time
    """
    # Compare magnitude of velocity over time
    fig, ax = plt.subplots()
    ax.plot(true_orbit['TIME'], np.linalg.norm(true_orbit['VEL '], axis = 1), label = 'simulation')
    ax.plot(int_orbit['TIME'], np.linalg.norm(int_orbit['VEL '], axis = 1), linestyle = '--', label = 'orbit integration')

    # Axis scales & labels
    ax.set_yscale('log')
    ax.set_ylabel('magnitude of velocity (km/s)')
    ax.set_xlabel('age of unvierse (Gyr)')

    # Legend
    ax.legend()

    return fig

def plot_vxyz(true_orbit, int_orbit):
    """
    Compare vx, vy, and vz over time for true and integrated orbits.
    Parameters:
        true_orbit [dict]: orbit read from simulation data. Includes 'TIME' (time in Gyr) and 'POS ' (position over time in physical kpc.
        int_orbit [dict]: orbit reconstructed via orbit integration. Includes 'TIME' (time in Gyr) and 'POS ' (position over time in physical 
                          kpc.
    Returns:
        Figure comparing vx, vy, vz over time for true and integrated orbits.
    """
    # Figure settings
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(12,3)
    plt.subplots_adjust(wspace=0.35, hspace=0.25)

    # Plot vx, vy, vz over time
    ax[0].plot(true_orbit['TIME'], true_orbit['VEL '][:,0])
    ax[0].plot(int_orbit['TIME'], int_orbit['VEL '][:,0], linestyle = '--')
    ax[1].plot(true_orbit['TIME'], true_orbit['VEL '][:,1])
    ax[1].plot(int_orbit['TIME'], int_orbit['VEL '][:,1], linestyle = '--')
    ax[2].plot(true_orbit['TIME'], true_orbit['VEL '][:,2], label = 'simulation')
    ax[2].plot(int_orbit['TIME'], int_orbit['VEL '][:,2], linestyle = '--', label = 'orbit integration')

    # Axis labels
    ax[0].set_ylabel('vx (km/s)')
    ax[1].set_ylabel('vy (km/s)')
    ax[2].set_ylabel('vz (km/s)')
    [ax_i.set_xlabel('age of universe (Gyr)') for ax_i in ax]

    # Legend
    ax[2].legend()
    
    return fig

def compare_mass_contours(grids, sim_mass_bins, model_mass_bins, title = '', levels=200, linthresh=1e8, linscale=0.25, num_bins = 100, 
                          limit = 60):
    """
    2-row plot including simulation contours and model contours for each of the three orientations.
    Parameters:
        grids [list of 3 1darrays]: positions of the bins of the 2dhistogram in x, y, z (each array corresponds to a coordinate)
        sim_mass_bins, model_mass_bins [list of 3 1darrays]: binned masses for each of the three coordinate pairs (x, y) (y, z) (x, z)
        title [str]: plot title
        levels [int]: how many contours to plot
        linthresh [float]: for SymLogNorm colorbar, the threshold for linear color scaling
        linscale [float]: for SymLogNorm colorbar, the relative scale of the linear color scaling
        num_bins [int]: number of bins for calculating the density
        limit [float]: x_lim and y_lim for plot
    """
    
    # For plotting and labels
    orientations = {
        'xy': ((0, 1), ('x (kpc)', 'y (kpc)')),
        'yz': ((1, 2), ('y (kpc)', 'z (kpc)')),
        'xz': ((0, 2), ('x (kpc)', 'z (kpc)'))
    }

    # Precompute all densities and determine the range for color normalization
    densities = {}
    for i, (orientation, (coords, _)) in enumerate(orientations.items()): # _ means disregard that variable (in this case, the label of the plot)
        coord_1, coord_2 = coords
        densities[orientation] = {
            'sim': calculate_projected_density(sim_mass_bins[i], grids, coord_1, coord_2),
            'model': calculate_projected_density(model_mass_bins[i], grids, coord_1, coord_2)
        }
        min_density = min(densities[orientation]['sim'].min(), densities[orientation]['model'].min())
        max_density = max(densities[orientation]['sim'].max(), densities[orientation]['model'].max())

    # Set the colorscale
    contour_norm = colors.SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=min_density, vmax=max_density)

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.75))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for i, (orientation, (coords, labels)) in enumerate(orientations.items()):
        coord_1, coord_2 = coords

        # Calculate the center of boxes for plotting
        x_plot = (grids[coord_1][1:] + grids[coord_1][:-1]) / 2
        y_plot = (grids[coord_2][1:] + grids[coord_2][:-1]) / 2
        
        # Plot simulation and model contours using helper function
        plot_contour(axes[0, i], x_plot, y_plot, densities[orientation]['sim'], levels, contour_norm, limit, labels, 
                      f'Simulation ({orientation})')
        plot_contour(axes[1, i], x_plot, y_plot, densities[orientation]['model'], levels, contour_norm, limit, labels, 
                      f'Model ({orientation})')

    # Add colorbar to the figure
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.09, 0.02, 0.8])
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=contour_norm, cmap='cividis'), cax=cbar_ax)
    cbar.set_label(r'Surface Density ($M_\odot/kpc^2$)', fontsize = 15, labelpad = 15)

    # Title
    plt.suptitle(title, fontsize = 18, y = 0.98)

    return fig

def calculate_projected_density(mass_bins, grids, coord_1, coord_2):
    """
    Calculate the density for given mass bins and grids.
    Parameters
        mass_bins [list of 3 1darrays]: binned masses for each of the three coordinate pairs (x, y) (y, z) (x, z)
        grids [list of 3 1darrays]: positions of the bins of the 2dhistogram in x, y, z (each array corresponds to a coordinate)
        coord_1, coord_2 [int, int]: integers for indexing corresponding to the desired coordinate (e.g. 0,1 -> x, y)
    Returns:
        Density in the desired projection
    """
    box_area = (grids[coord_1][1] - grids[coord_1][0]) * (grids[coord_2][1] - grids[coord_2][0])
    return mass_bins / box_area

# def plot_contour(ax, x_plot, y_plot, density, levels, contour_norm, limit, labels, title):
#     """
#     Helper function to plot mass contours.
#     Parameters:
#         ax [matplotlib axis]: axis to plot
#         x_plot, y_plot [1darray, 1darray]: x and y grids to plot the density on
#         density [2darray]: binned density for each node of the xy grid
#         contour_norm [matplotlib colormap]: normalized colormap for plotting the contours
#         limit [float]: x, y limit
#         labels [(str, str)]: x and y axis labels
#         titles [str]: axis title
#     Outputs:
#         Plots a density contour on the provided axis
#     """
#     # Set title
#     ax.set_title(title)

#     # Set axis labels
#     x_label, y_label = labels
#     ax.set_xlabel(x_label)
#     ax.set_ylabel(y_label)

#     # Set limits
#     ax.set_xlim(-limit, limit)
#     ax.set_ylim(-limit, limit)

#     # Plot
#     cmap = colormap.get_cmap('cividis')
#     ax.contourf(x_plot, y_plot, density.T, cmap=cmap, levels=levels, norm=contour_norm)

#     # Fill in the background with the darkest colormap color present in the data (in case there's not enough data to fill out the whole plot)
#     ax.set_facecolor(cmap(0))

def compare_integration_radial(true_orbit, integrated_orbits, orbit_names, true_label = 'simulation', cmap = plt.cm.viridis, plot_vr = True):
    """
    Compare one true orbit (radius over time) with orbits coming from an unlimited number of different integration methods
    Parameters:
        true_orbit[dict]: "true" orbit or orbit to compare to, with keys 'POS ', 'SNAP', and 'TIME'
        integrated_orbits [list or dict]: list of orbit dictionaries (or single orbit dictionary) with the same keys as above
        orbit_names [list]: list of names given to each of the integrated orbits
        true_label [str]: label for the "true" orbit or orbit which is being compared to
        cmap [matplotlib.cm]: colormap for plotting the different orbits
        plot_vr [bool] whether or not to plot the virial radius along with the orbits
    Returns:
        figure comparing the orbital radius over time between the true orbits & the various integrated ones
    """
    # Formatting
    if type(integrated_orbits) == dict:
        integrated_orbits = [integrated_orbits]
    
    # Set up figure   
    fig, ax = plt.subplots()
    fig.set_size_inches(12,8)

    # Assign a different color to each orbit
    colors = [cmap(i) for i in np.linspace(0, 1, len(integrated_orbits))]

    # Plot radii over time, color-coding by orbit
    for i in range(len(integrated_orbits)):
        ax.plot(integrated_orbits[i]['TIME'], np.linalg.norm(integrated_orbits[i]['POS '], axis = 1), color = colors[i], 
                label = orbit_names[i], zorder = 1)

    # Plot true orbit
    ax.plot(true_orbit['TIME'], np.linalg.norm(true_orbit['POS '], axis = 1), color = 'black', label = true_label, linewidth = 3, zorder = 0)
    
    # Plot virial radius
    if plot_vr:
        virial_radius = get_virial_radius(true_orbit['SNAP'])
        plt.plot(true_orbit['TIME'], virial_radius, color = 'black', linestyle = 'dotted', label = 'virial radius')

    plt.legend()

    # Title based on how long orbit integration was performed
    int_time = true_orbit['TIME'][-1] - true_orbit['TIME'][0]
    plt.title(f'{round(float(int_time), 2)} Gyr orbit integration')
    # Text to help with ID'ing the orbit later
    if 'ID  ' in true_orbit.keys():
        plt.text(0.99, 0.01, f'start: {true_orbit["SNAP"][0]}, ID: {true_orbit["ID  "]}', verticalalignment='bottom', horizontalalignment='right', 
             transform = ax.transAxes)

    # Axis labels
    ax.set_ylabel('radius (kpc)')
    ax.set_xlabel('time (Gyr)')

    return fig

def mass_bins(particles, pot_model, num_bins = 100, limit = None):
    """
    Samples the mass from agama potential model and returns a 2d histogram of the mass in the simulation and the mass in the model for each
    pair of coordinates.
    Parameters:
        particles [dict]: particle dictionary with keys ['POS '] and ['MASS']
        pot_model [agama potential model]: model of potential, created using agama with sim_mass and sim_pos
        num_bins [int]: number of bins in each of the sides of the 2d histogram (e.g. num_bins = 10 -> 10 bins per side; 100 bins total)
        limit [float]]: x, y, and z limits of the histograms
    Returns:
        grids: positions of the bins of the 2dhistogram in x, y, z
        sim_mass_bins, model_mass_bins: 2D grid of binned masses for the simulation and the model for each of the three coordinate pairs (x, y) 
                                        y, z) (x, z)
    """
        
    # Define grid for 2D histogram
    if limit is None:
        x_grid = np.linspace(np.min(particles['POS '][:,0]), np.max(particles['POS '][:,0]), num_bins)
        y_grid = np.linspace(np.min(particles['POS '][:,1]), np.max(particles['POS '][:,1]), num_bins)
        z_grid = np.linspace(np.min(particles['POS '][:,2]), np.max(particles['POS '][:,2]), num_bins)
    else:
        limit *=1.01 # The size of the x_grid, y_grid, and z_grid determines the extent of any contour plot. If we make them end directly on +/- 
        # limit, the color on the contour plot will not extend to the edge of the axis. So basically, we increase the limit to make the plots look 
        # good. It shouldn't change anything about the actual data.
        x_grid, y_grid, z_grid = (np.linspace(-limit, limit, num_bins), np.linspace(-limit, limit, num_bins), 
                                  np.linspace(-limit, limit, num_bins))
    grids = [x_grid, y_grid, z_grid]

    # Sample the potential model to get masses
    sampled_particles = BFE.sample_from_model(pot_model, len(particles['MASS']))

    # Get 2D mass histograms
    sim_mass_bins = [np.histogram2d(particles['POS '][:,coord1], particles['POS '][:,coord2], weights = particles['MASS'], 
                                    bins = [grids[coord1], grids[coord2]])[0] for (coord1,coord2) in zip([0, 1, 0], [1, 2, 2])]
    model_mass_bins = [np.histogram2d(sampled_particles['POS '][:,coord1], sampled_particles['POS '][:,coord2], weights = 
                                      sampled_particles['MASS'], bins = [grids[coord1], grids[coord2]])[0] for (coord1,coord2) in 
                       zip([0, 1, 0], [1, 2, 2])]

    return grids, sim_mass_bins, model_mass_bins

def simple_hist2d(particles, titles, num_bins = 100, rmax = None):
    """
    Plot a simple 2d histogram from 3 different angles in x,y, z from a collection of particles of different types
    Parameters:
        particles [list or dict]: (list of) dict(s) that contain key ['POS '] for x,y,z positions. Each dict represents a different type of 
                                  particles (e.g. simulation particles, sampled model particles)
        titles [list or str]: (list of) title(s) (one for each type of particle)
        num_bins [int]: number of bins to use for 2d histogram
        rmax [float]: maximum radius of particles to plot
    Returns:
        2D histogram figure 
    """
    # Formatting
    if type(particles) is not list:
        particles = [particles]
    if type(titles) is not list:
        titles = [titles]
    
    # log norm for colors
    norm = LogNorm()

    # Plot settings
    fig, ax = plt.subplots(len(particles),3)
    fig.set_size_inches(9, 2.75*len(particles))
    plt.subplots_adjust(wspace=0.45, hspace = 0.55)
    plot_dict = {'coords': [(0, 1), (1, 2), (0, 2)],
                 'axis_labels': [('x (kpc)', 'y (kpc'), ('y (kpc)', 'z (kpc'), ('x (kpc)', 'z (kpc')]} # Axis label tuples

    # Make sure this works even if there is only one row of plots
    if len(particles) == 1:
        ax = np.atleast_2d(ax) 

    
    # Plot
    for j, (part, title) in enumerate(zip(particles, titles)):
        
        # Particle inclusion / exclusion
        if rmax is None:
            r_filter = np.full_like(part['POS '][:,0], True, dtype = bool) # Include all particles
        else:
            r_filter = np.where(np.linalg.norm(part['POS '], axis = 1) <= rmax)[0] # Include only particles with r <= rmax

        
        for i, (coords, labels) in enumerate(zip(plot_dict['coords'], plot_dict['axis_labels'])):
            _ = ax[j, i].hist2d(part['POS '][:,coords[0]][r_filter], part['POS '][:,coords[1]][r_filter], bins = num_bins, norm = norm)
            ax[j, i].set_xlabel(labels[0]) # Picks the first in each tuple
            ax[j, i].set_ylabel(labels[1]) # Picks the second in each tuple
            ax[j, i].set_aspect('equal') # Ensure each subplot is square

        # Title each component
        ax[j, 1].set_title(title, fontsize = 15, pad = 10.0)

    return fig

def compare_dp_potential(particles, potential, title = '', num_bins = 200, rmin = 1e-1, rmax = None):
    """
    Plot radial density profile or real particles vs. fitted potential.
    Parameters:
        snapshot [int]: simulation snapshot
        particles [dict]: key ['POS '] for x,y,z positions, ['MASS']
        potential [agama potential]: potential model corresponding to particles
        title [str]: plot title
        num_bins [int]: number of bins for density profile
        rmin, rmax [float, float]: minimum (maximum) radii for the density profile
    Returns:
        Density profile comparison.
    """

    # Calculate densities
    particle_bins, particle_density = density_profiles.calc_sim_density(particles, num_bins, rmin, rmax)
    model_density = calc_model_density(potential, len(particles['MASS']), particle_bins)

    # Make plot
    avg_r = (particle_bins[1:] + particle_bins[:-1]) / 2 # Get average radius of spherical density shells for most accurate plotting
    fig, ax = plt.subplots()

    # Plot density vs. radius
    ax.plot(avg_r, particle_density*avg_r**2, label='simulation')
    ax.plot(avg_r, model_density*avg_r**2, linestyle = '--', label = 'model')

    # Set plot properties-----
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r$ / kpc')
    ax.set_ylabel(r'$\rho (r) * r^2$ / $M_\odot kpc^-1$')
    ax.legend()
    plt.title(title)

    return fig

def compare_dp_sampled_particles(input_particles, sampled_particles, title = '', num_bins = 200, rmin = None, rmax = None, pres_mode = False):
    """
    Plot radial density profile of real vs. sampled particles
    Parameters:
        input_particles, sampled_particles [dict, dict]: input simulation particles & sampled particles with keys 'POS ' (xyz position) &
                                                         'MASS'
        title [str]: plot title
        num_bins [int]: number of bins for density profile
        rmin, rmax [float, float]: minimum (maximum) radii for the density profile
        pres_mode [bool]: whether or not to use presentation dark mode
    Returns:
        Density profile comparison.
    """
    if pres_mode:
        apply_pres_mode()

    # Calculate densities
    rbins, input_density = density_profiles.calc_sim_density(input_particles, num_bins, rmin, rmax)
    r = (rbins[1:] + rbins[:-1]) / 2
    sample_bins, sampled_density = density_profiles.calc_sim_density(sampled_particles, num_bins, (rmin, rmax), rbins = rbins)

    # Make plot
    fig, ax = plt.subplots(2, 1, sharex = True, height_ratios = (2, 1))
    fig.set_size_inches(5,6)
    plt.subplots_adjust(hspace = 0.1)

    # Plot density vs. radius
    ax[0].plot(r, input_density*r**2, color = '#F8E620', label='true density')
    ax[0].plot(r, sampled_density*r**2, color = '#B5D9F6', linestyle = '--', label='sampled density')

    # Plot residuals
    residuals = np.abs((input_density - sampled_density) / input_density)
    ax[1].plot(r, residuals, color = '#E7B3FF')

    # Set plot properties-----
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_xscale('log')
    ax[0].set_ylabel(r'$\rho (r) * r^2$ / $M_\odot kpc^-1$')
    ax[1].set_xlabel(r'$r$ / kpc')
    ax[1].set_ylabel(r'$|\Delta \rho / \rho|$')
    ax[0].legend()
    plt.suptitle(title, y = 0.97, ha = 'center')

    plt.rcdefaults()

    return fig

def apply_pres_mode():
    """
    Apply "presentation mode" to plotting. This means making text colors white (to show up against a dark background) & making text sizes
    bigger than normal.
    It is important to use plt.rcdefaults() at the end of your function to reset these changes to the default.
    """
    # Create a dictionary with all of the parameters you want to modfiy
    pres_mode_style = {
            'axes.edgecolor': 'white',     # Edge color of the plot
            'axes.labelcolor': 'white',    # Label color
            'xtick.color': 'white',        # X-axis tick color
            'ytick.color': 'white',        # Y-axis tick color
            'text.color': 'white',         # Text color
            'lines.color': 'white',        # Line color
            'patch.edgecolor': 'white',    # Edge color for patches
            'legend.facecolor': 'none',    # Transparent legend face color
            'legend.edgecolor': 'white',   # White frame color
            'axes.titlesize': 24,            # Title font size
            'axes.labelsize': 18,            # Axis label font size
            'xtick.labelsize': 13,           # X-axis tick label font size
            'ytick.labelsize': 13,           # Y-axis tick label font size
            'legend.fontsize': 13,          # Legend font size
            'figure.titlesize': 26        # Suptitle font size
            
        }
    plt.rcParams.update(pres_mode_style) # Update standard matplotlib parameters to reflect these changes

    return

def sph_mass_bins(particles, orientation, limits, depth, npixels, nb = 32):
    """
    Calculate the sph smoothed surface density of particles. See https://github.com/alejandrobll/py-sphviewer/tree/master/sphviewer for more 
    info about how sph works.
    Inputs:
        particles [dict]: particle dictionary with keys 'POS ' & 'MASS'
        orientation [string]: One of 'xy', 'yz', 'xz'; represents which orientation to calculate surface density from.
        limits [tuple]: x, y, & z limits of density. The axis that is not represented with your choice of orientation will not matter.
        depth [float]: radial depth for the non-chosen axis.
        npixels [int]: number of bins (pixels) for density
        nb [int]: number of neighbors to use for smoothing (more = smoother & more computationally intensive) 
    Outputs:
        density [npixels x npixels array]: represents the density at each of the npixels**2 points you have chosen, spanning the grid created
                                           using limits.
    """
    assert orientation in ['xy', 'yz', 'xz'], f'Chosen orientation of {orientation} is not supported. Please choose one of "xy", "yz", or "xz"'

    # Define various rotations
    graphing = {
        'xy': (0, 1, 2),
        'yz': (1, 2, 0),
        'xz': (0, 2, 1)
    }

    # Reorder particle positions to get the right orientation
    rot_pos = particles['POS '][:,graphing[orientation]]
    # QuickView will automatically calculate the smoothed histogram for the xy orientation, so we need to reorder the particles to get
    # what we want

    # Set xlim, ylim, zlim (zlim isn't useful for anything due to the automatic priveliging of xy orientation.
    xlim, ylim, zlim = np.array(limits)[list(graphing[orientation])] 
    
    # Calculate surface density
    extent = np.array([-xlim,xlim,-ylim,ylim])
    z_slice = np.abs(rot_pos[:,2]) <= depth
    density = QuickView(rot_pos[z_slice], mass=particles['MASS'][z_slice],\
                    r='infinity', x=0, y=0, z=0, extent=list(extent), \
                    plot=False, logscale=True, xsize = npixels, ysize = npixels, nb = nb).get_image()
    
    return density

def plot_contour(ax, grids, density, cmap = 'viridis', contour_norm = LogNorm(), contour_lines = None, ncontours = 5):
    """
    Make contour plot.
    Parameters:
        ax [matplotlib axis]: axis where you want to plot the contours
        grids [tuple]: tuple of x and y bins (1darrays) that the density is defined over
        density [NxN array]: array that defines the density over the x & y grids
        cmap [str]: matplotlib colormap name
        contour_norm [matplotlib colormap']: color normalization
        contour_lines [NxN array]: optional extra-smoothed densities used to plot contour lines
        nlevels [int]: if contour_lines is defined, determiens the number of contour lines to plot
    Returns:
        plotting the densities on the given axis
    """
    # Set limits
    grid_x, grid_y = grids
    ax.set_xlim(grid_x[0], grid_x[-1])
    ax.set_ylim(grid_y[0], grid_y[-1])

    # Plot the filled contours
    extent = np.array([grid_x[0],grid_x[-1],grid_y[0],grid_y[-1]])
    ax.imshow(density, extent = extent, origin='lower', cmap = cmap, norm = contour_norm)

    # Plot the contour lines
    if contour_lines is not None:
        min_density = np.log10(density.min())
        max_density = np.log10(density.max())
        contour_levels = np.logspace(min_density, max_density, ncontours)
        ax.contour(grid_x, grid_y, contour_lines, levels=ncontours, colors = 'white')

def plot_residuals(ax, true_density, calculated_density, levels, grids, cax, colorbar_lim = None, cmap = 'RdBu', plot_colorbar = True):
    """
    Plot percent difference between two densities on a given axis
    Inputs:
        ax [matplotlib axis]: axis to plot the residuals
        true_density [NxN array]: true density defined over the x & y grids
        calculated_density [NxN array]: calcualted density for comparison defined over the x & y grids
        levels [int]: number of levels to use for the plot
        grids [tuple]: tuple of x and y bins (1darrays) that the density is defined over
        cax [matplotlib axis]: axis defined to contain the colorbar
        colorbar_lim [float]: optional limit for the error colorbar (errors outside of this range will just be represented as a dark color
                              & not specifically identified)
        cmap [str]: name of matplotlib colormap
        plot_colorbar [bool]: Whether or not to plot the colorbar associated with the residuals
    """
    # Calculate percent difference; put nan where there is no data to ensure there are no divide by zero errors
    percent_differences = np.divide((true_density-calculated_density), true_density, out=np.full_like(true_density, np.nan), 
                                    where=true_density!=0)

    # Make sure the colorbar limit makes sense
    if (colorbar_lim is None) or (np.max(np.abs(percent_differences)) < colorbar_lim):
        colorbar_lim = np.max(np.abs(percent_differences))

    # Workaround since the 'extend' option in the colorbars is not working with contourf as far as I can tell
    # So I'll just set the out of range values myself
    percent_differences = np.clip(percent_differences, -(1.1*colorbar_lim), 1.1*colorbar_lim)
    
    # Set color norm
    norm = colors.TwoSlopeNorm(vcenter = 0, vmin = -colorbar_lim, vmax = colorbar_lim)

    # Plot contours
    grid_x, grid_y = grids 
    # Define levels specifically rather than just as a number. Creates a symmetrical colorbar-- I don't know why it's necessary but it is.
    levels = np.linspace(-colorbar_lim, colorbar_lim, levels) 
    residual_contours = ax.contourf(grid_x, grid_y, percent_differences, cmap=cmap, levels = levels, norm = norm, extend = 'both')
    
    # Plot colorbar
    plt.colorbar(residual_contours, cax = cax, extend = 'both', label = r'$\Delta \rho / \rho$', format = PercentFormatter(xmax=1))