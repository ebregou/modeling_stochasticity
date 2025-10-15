import eMCMC
import eplots
import edata
import os
import dataframe_image as dfi

# Describe how this run has been modified from a standard one
"""
Only including redshifts 7,8,9 (line 20)
Got rid of dsigma/dz (line 24)
"""

# Name this run
run_name = 'z=789'

# Get data
file_names = ['/Users/eb35267/Desktop/code/home/data/Bouwens2021_fixed.txt', 
              '/Users/eb35267/Desktop/code/home/data/Donnan24_GAL.txt',
             '/Users/eb35267/Desktop/code/home/data/Mcleod_24.txt']
data_labels = ['Bouwens+21', 'Donnan+24', 'McLeod+24']
sorted_data = edata.get_sorted(file_names, data_labels, include_zs= [7, 8, 9])

# Initialize parameters
params = eMCMC.build_param_data({'dsigdz': {'fit': False, 'value': 0}, 'dalphadz': {'fit': False, 'value':0}, 'dbetadz': {'fit':False, 'value':0}, 'dlogedz': {'fit':False, 'value':0}, 'dlogMcdz':{'fit':False, 'value':0}})

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
table = eMCMC.make_table([best_fit], param_labels, ['best fit'], [bounds])
dfi.export(table, f'{folder}/table.png', table_conversion = 'matplotlib', use_mathjax = True, dpi = 200)



