from escripts import eMCMC
from escripts import eplots
from escripts import edata
import os
import dataframe_image as dfi

# Name this run
run_name = 'no_M_ev'

# Input data
file_names = ['/Users/eb35267/Desktop/code/home/data/Bouwens2021_low_z.txt',
              '/Users/eb35267/Desktop/code/home/data/Donnan24_limit_z.txt']
data_labels = ['Bouwens+21', 'Donnan+24']

# Initialize parameters
params = eMCMC.build_param_data({'beta': {'fit': False, 'value': -1.49}, 'dbetadz': {'fit': False, 'value': 0.22}, 
                                 'dsigdlogM':{'fit': False, 'value': 0}})

# Adjust ICs
lowers, uppers = None, None

# Run MCMC ---------------------------------------------------------------------------------------------------------------------------------------------

# Get the data
sorted_data = edata.get_sorted(file_names, data_labels)

# Create backend file
folder = f'/Users/eb35267/Desktop/code/home/figures/{run_name}'
if not os.path.exists(folder):
    os.makedirs(folder)
else:
    print(f'Data in {folder} will be overwritten')
backend_filename = f'{folder}/samples.h5'

# Make the UVLF object
my_UVLF = eMCMC.UVLF(sorted_data, params, backend_filename = backend_filename)

# Assign uppres & lowers if necessary
if lowers is not None:
    ICs = my_UVLF.generate_ICs(lowers, uppers)
else:
    ICs = None

# Run the chain
my_UVLF.run_MCMC(ICs = ICs) # If ICs is None this will assign them based on the upper & lower limits on the parameters

# Make & save figures ---------------------------------------------------------------------------------------------------------------------------------------------
walkers, best_fit, bounds, param_labels = my_UVLF.get_fit()

# Corner plot
corner = eplots.make_corner(my_UVLF, true_vals = best_fit)
corner.savefig(f'{folder}/corner.png', bbox_inches = 'tight')

# UVLF plot
ev_fig = eplots.evolving_UVLF_fit(my_UVLF)
ev_fig.savefig(f'{folder}/evolving_UVLF.png', bbox_inches = 'tight')

# SFE(Mh) for different z
sfe_over_mh = eplots.sfe_shape_diff_z(my_UVLF, best_fit)
sfe_over_mh.savefig(f'{folder}/sfe_over_mh.png', bbox_inches = 'tight')

# SFE(z) for different Mh at z = 0
sfe_over_z = eplots.sfe_over_time(my_UVLF, best_fit)
sfe_over_z.savefig(f'{folder}/sfe_over_z.png', bbox_inches = 'tight')

# Plot walkers & burn-in
walker_fig = eplots.walkers(my_UVLF, best_fit)
walker_fig.savefig(f'{folder}/walkers.png', bbox_inches='tight')

# Plot sigmaUV vs. Mh at different redshifts
sigMh_fig = eplots.sigma_Mh(my_UVLF, best_fit)
sigMh_fig.savefig(f'{folder}/sigvsMh.png', bbox_inches='tight')

# Plot P(MUV|Mh) at a central redshifts
PMUV, _ = eplots.PMUV(my_UVLF, best_fit)
PMUV.savefig(f'{folder}/PMUV.png', bbox_inches='tight')

# Plot P(Mh|MUV) at different redshifts
PMh = eplots.PMh(my_UVLF, best_fit)
PMh.savefig(f'{folder}/PMh.png', bbox_inches='tight')

# Parameter table
table = eplots.make_table([best_fit], param_labels, ['best fit'], [bounds])
dfi.export(table, f'{folder}/table.png', table_conversion = 'matplotlib', use_mathjax = True, dpi = 200)



