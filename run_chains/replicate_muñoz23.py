from escripts import eMCMC
from escripts import eplots
from escripts import edata
import os
import dataframe_image as dfi

# Name this run
run_name = 'replicate_muñoz23_hst'

# Get data
file_names = ['/Users/eb35267/Desktop/code/home/data/Bouwens2021_fixed.txt']
data_labels = ['Bouwens+21']
sorted_data = edata.get_sorted(file_names, data_labels)

# Initialize parameters
params = params = eMCMC.build_param_data({'dsigdlogM': {'fit':False, 'value':0}})

# Add Muñoz+23 fits for comparison
munoz_fit_vary_eps = [0.61, -0.01, -1.91, 0.08, 12.03, 0.03, '-', '-', 0.65, -0.03]
munoz_fit_vary_sig = [0.74, 0.03, -1.76, -0.02, 11.84, -0.02, -1.08, -0.07, '-', '-']


#---------------------------------------------------------------------------------------------------------------------------------------------

# Create backend file
folder = f'/Users/eb35267/Desktop/code/figures/{run_name}'
if not os.path.exists(folder):
    os.makedirs(folder)
else:
    print(f'Data in {folder} will be overwritten')
backend_filename = f'{folder}/samples.h5'

# Make the UVLF object
my_UVLF = eMCMC.UVLF(sorted_data, params, backend_filename = backend_filename)

# Run the chain
my_UVLF.run_MCMC()

# Make & save figures
_, best_fit, bounds, param_labels = my_UVLF.get_fit()

# Corner plot
corner = eplots.make_corner(my_UVLF, true_vals = best_fit)
corner.savefig(f'{folder}/corner.png', bbox_inches = 'tight')

# UVLF plot
ev_fig = eplots.evolving_UVLF_fit(my_UVLF)
ev_fig.savefig(f'{folder}/evolving_UVLF.png', bbox_inches = 'tight')

# Parameter table
table = eplots.make_table([best_fit, munoz_fit_vary_eps, munoz_fit_vary_sig], param_labels, ['best fit', r'Muñoz+23 piecewise $\epsilon_0$', 
                                                                                             r'Muñoz+23 piecewise $\sigma_{\rm{UV}}$'], 
                          [bounds])
dfi.export(table, f'{folder}/table.png', table_conversion = 'matplotlib', use_mathjax = True, dpi = 200)



