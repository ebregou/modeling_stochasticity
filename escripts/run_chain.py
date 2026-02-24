from escripts import eMCMC
from escripts import eplots
from escripts import edata
import os
import dataframe_image as dfi
import yaml 
import sys

# Get the information from the YAML file
config_path = sys.argv[1]
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

run_name = cfg["run_name"]
file_names = cfg["file_names"]
data_labels = cfg["data_labels"]
include_zs = cfg["include_zs"]
param_input = cfg["param_input"]
lowers = cfg.get("lowers")
uppers = cfg.get("uppers")

# Run MCMC ---------------------------------------------------------------------------------------------------------------------------------------------

# Get the data
sorted_data = edata.get_sorted(file_names, data_labels, include_zs = include_zs)

# Store the data in a backend .h5 file
backend_filename = f'/Users/eb35267/Desktop/code/temp/samples_{run_name}.h5'

# Save the parameter data
params = eMCMC.build_param_data(param_input)

# Make the UVLF object
my_UVLF = eMCMC.UVLF(sorted_data, params, backend_filename = backend_filename)

# Assign uppres & lowers if necessary
if lowers is not None:
    ICs = my_UVLF.generate_ICs(lowers, uppers)
else:
    ICs = None

# Run the chain
my_UVLF.run_MCMC(ICs = ICs) # If ICs is None this will assign them based on the upper & lower limits on the parameters

# Save chain information
walkers, best_fit, bounds, param_labels = my_UVLF.get_fit()

# Make & save figures ---------------------------------------------------------------------------------------------------------------------------------------------

# Corner plot
corner = eplots.make_corner(my_UVLF) #, true_vals = best_fit)
corner.savefig('corner.png')

# UVLF plot
ev_fig = eplots.evolving_UVLF_fit(my_UVLF)
ev_fig.savefig('evolving_UVLF.png')

# SFE(Mh) for different z
sfe_over_mh, _ = eplots.sfe_shape_diff_z(my_UVLF, best_fit)
sfe_over_mh.savefig('sfe_over_mh.png')

# SFE(z) for different Mh at z = 0
sfe_over_z = eplots.sfe_over_time(my_UVLF, best_fit)
sfe_over_z.savefig('sfe_over_z.png')

# Plot walkers & burn-in
walker_fig = eplots.walkers(my_UVLF, best_fit)
walker_fig.savefig('walkers.png')

# Plot sigmaUV vs. Mh at different redshifts
sigMh_fig = eplots.sigma_Mh(my_UVLF, best_fit)
sigMh_fig.savefig('sigvsMh.png')

# Plot P(MUV|Mh) at a central redshifts
PMUV, _ = eplots.PMUV(my_UVLF, best_fit)
PMUV.savefig('PMUV.png')

# Plot P(Mh|MUV) at different redshifts
PMh = eplots.PMh(my_UVLF, best_fit)
PMh.savefig('PMh.png')

# Parameter table
table = eplots.make_table([best_fit], param_labels, ['best fit'], [bounds])
dfi.export(table, 'table.png', table_conversion = 'matplotlib', use_mathjax = True, dpi = 200)

os.replace(backend_filename, 'samples.h5') # Move the h5 file into current directory once the run is done (not before since I have that directory
                                            # backed up with Box which causes issues)



