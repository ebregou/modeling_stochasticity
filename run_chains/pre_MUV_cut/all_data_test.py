from escripts import eMCMC
from escripts import eplots
from escripts import edata
import os
import dataframe_image as dfi

# This run only included HST data up to z = 7

# Name this run
run_name = 'all_data_test'

# Get data
file_names = ['/Users/eb35267/Desktop/code/home/data/Bouwens2021_low_z.txt',
              '/Users/eb35267/Desktop/code/home/data/Donnan24_GAL.txt',
             '/Users/eb35267/Desktop/code/home/data/Mcleod_24.txt',
             '/Users/eb35267/Desktop/code/home/data/Adams+25.txt',
             '/Users/eb35267/Desktop/code/home/data/Weibel+25.txt']
data_labels = ['Bouwens+21', 'Donnan+24', 'McLeod+24', 'Adams+25', 'Weibel+25']
sorted_data = edata.get_sorted(file_names, data_labels)

# Initialize parameters
params = params = eMCMC.build_param_data({'dsigdlogM': {}})


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
table = eplots.make_table([best_fit], param_labels, ['best fit'], [bounds])
dfi.export(table, f'{folder}/table.png', table_conversion = 'matplotlib', use_mathjax = True, dpi = 200)



