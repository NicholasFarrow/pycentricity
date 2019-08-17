import python_code.reweight as rwt

import glob
import argparse

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.special import logsumexp


fontparams = {'mathtext.fontset': 'stix',
             'font.family': 'serif',
             'font.serif': "Times New Roman",
             'mathtext.rm': "Times New Roman",
             'mathtext.it': "Times New Roman:italic",
             'mathtext.sf': 'Times New Roman',
             'mathtext.tt': 'Times New Roman'}
rcParams.update(fontparams)

# Set up the argument parser
parser = argparse.ArgumentParser("Rebuild the eccentricity distribution based on weights given in output files")
parser.add_argument("-r", "--results-folder", help="Location of the results files")
args = parser.parse_args()

result_files = list(glob.glob(args.results_folder + '/result_*_master_output_store.txt'))
eccentricities = []
log_weights = []

for single_file in result_files:
    disregard = False
    with open(single_file, 'r') as s_f:
        for line in s_f:
            if 'e' not in line:
                if 'None' not in line:
                    split_line = line.split('\t\t')
                    eccentricities.append(float(split_line[1]))
                    log_weights.append(float(split_line[3]))
                else:
                    disregard = True

storage_file = args.results_folder + '/eccentricity_histogram_data.txt'
with open(storage_file, 'w') as f:
    for eccentricity in eccentricities:
        f.write(str(eccentricity) + '\n')

maximum_eccentricity = np.quantile(eccentricities, 0.9)
minimum_eccentricity = np.quantile(eccentricities, 0.1)

print('e_max: ' + str(maximum_eccentricity))
print('e_min: ' + str(minimum_eccentricity))

squared_log_weights = np.multiply(log_weights, 2)
log_sum_weights = logsumexp(log_weights)
log_sum_squared_weights = logsumexp(squared_log_weights)
log_neff = 2 * log_sum_weights - log_sum_squared_weights
print('log_neff: ' + str(log_neff))
print('neff: ' + str(np.exp(log_neff)))
print('efficiency: ' + str(np.exp(log_neff) / len(log_weights)))
ln_B = log_sum_weights - np.log(len(log_weights))
print('log B: ' + str(ln_B))

fig = plt.figure()
plt.hist(eccentricities, bins=np.logspace(-4, np.log10(0.2), 20), alpha=0.5)
plt.axvline(minimum_eccentricity, color='r', alpha=0.5)
plt.axvline(maximum_eccentricity, color='r', alpha=0.5)
plt.axvline(0.1, color='g', alpha=0.5)
plt.grid(False)
plt.xscale('log')
plt.xlabel('eccentricity, $e$')

output_file = args.results_folder + '/eccentricity_histogram.pdf'
plt.savefig(output_file, bbox_inches='tight')
