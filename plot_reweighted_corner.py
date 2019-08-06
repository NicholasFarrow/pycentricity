import bilby as bb
import python_code.utils as utils
import numpy as np
import argparse


# Set up the argument parser
parser = argparse.ArgumentParser("")
parser.add_argument("-e", "--event", "The event name, used as a label or a tag")
parser.add_argument("-r", "--results-file", help="The path to the original results file")
args = parser.parse_args()

# Read in the arguments
event = args.event
result = bb.result.read_in_result(args.results_file)
posterior = result.posterior
number_of_samples = len(posterior['luminosity_distance'])

weights_file_name = event + '_recalculated_log_weights.txt'
weights = [0] * number_of_samples

with open(weights_file_name, 'r') as f:
    for line in f:
        split_line = line.replace('\n', '').split('\t\t')
        index = int(split_line[0])
        log_weight = float(split_line[1])
        weights[index] = np.exp(log_weight)

utils.plot_reweighted_posteriors(posterior, weights, label=event)



