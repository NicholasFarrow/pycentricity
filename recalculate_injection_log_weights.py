import bilby as bb
import python_code.reweight as rwt
import python_code.waveform as wf
import python_code.utils as utils
import numpy as np
import glob
import argparse


# Set up the argument parser
parser = argparse.ArgumentParser("")
parser.add_argument("-e", "--event", "The event name, used as a label or a tag")
parser.add_argument("-w", "--weights-directory", help="The location of the weights files")
args = parser.parse_args()

# Read in the arguments
event = args.event
weights_directory = args.weights_directory

# Read in the result and convert them to new weights
result_files = list(glob.glob(weights_directory + '/result_*_*_eccentricity_result.txt'))

# Parameters to search for in the file
injection_parameters = dict(
    mass_1=None,
    mass_2=None,
    eccentricity=None,
    luminosity_distance=None,
    theta_jn=None,
    psi=None,
    phase=None,
    geocent_time=None,
    ra=None,
    dec=None,
    chi_1=None,
    chi_2=None,
)

# Set up the basic properties of the runs
maximum_frequency = 1024
post_trigger_duration = 2
deltaT = 0.2

# Read event-specific properties from the utils file
sampling_frequency = utils.sampling_frequency[event]
minimum_frequency = utils.minimum_frequency[event]
duration = utils.event_duration[event]
detectors = utils.event_detectors[event]
trigger_time = utils.trigger_time[event]

# Generate the comparison waveform generator
comparison_waveform_generator = wf.get_IMRPhenomD_comparison_waveform_generator(
    minimum_frequency, sampling_frequency, duration
)
# Frequency array
frequency_array = comparison_waveform_generator.frequency_array

# Set up the interferometers with event data
interferometers = bb.gw.detector.InterferometerList(detectors)
start = trigger_time + post_trigger_duration - duration
end = start + duration
channel_dict = {
    key: key + ":" + utils.event_channels[event][key] for key in detectors
}
for ifo in interferometers:
    ifo.set_strain_data_from_csv(
        '/home/isobel.romero-shaw/public_html/PYCENTRICITY/pycentricity/submissions/'
        + args.event + '/event_data/' + ifo.name + '_time_domain_strain_data.csv'
    )
    ifo.power_spectral_density = bb.gw.detector.PowerSpectralDensity.from_power_spectral_density_file(
        psd_file=utils.event_psd_file_path[args.event][ifo.name]
    )
    ifo.minimum_frequency = minimum_frequency
    ifo.maximum_frequency = maximum_frequency

output_file = open(event + '_recalculated_log_weights.txt', 'w')
output_file.write("i\t\tlog_w\n")
log_weights = []

for single_file in result_files:
    disregard = False
    index_strings = single_file.split('/')[-1].replace(
        'result_', ''
    ).replace(
        '_eccentricity_result.txt', ''
    ).split('_')
    subset_number = float(index_strings)[0]
    increment_number = float(index_strings)[1]
    index = (subset_number * 10) + increment_number

    with open(single_file, 'r') as s_f:
        read_results_from_here = False
        eccentric_log_likelihood = []
        for line in s_f:
            if read_results_from_here:
                if 'None' not in line:
                    split_line = line.split('\t\t')
                    eccentric_log_likelihood.append(float(split_line[1]))
                else:
                    disregard = True
            else:
                for key in injection_parameters.keys():
                    if key in line:
                        split_line = line.replace('\n', '').split(': ')
                        injection_parameters[key] = float(split_line[-1])
            if 'e\t\tlog_L\t\tmaximised_overlap' in line:
                read_results_from_here = True
        if not disregard:
            # Recalculate the log likelihood for the fiducial model
            comparison_strain = comparison_waveform_generator.frequency_domain_strain(
                injection_parameters
            )
            circular_log_likelihood = rwt.log_likelihood_ratio(
                comparison_strain, interferometers, injection_parameters, duration
            )
            log_weight = rwt.calculate_log_weight(
                eccentric_log_likelihood, circular_log_likelihood
            )
            print(log_weight)
            log_weights.append(log_weight)
            output_file.write(str(log_weight) + '\n')
        else:
            print('disregarding file ' + single_file)

output_file.close()

weights = np.exp(log_weight)
n_eff = np.power(np.sum(weights), 2) / np.sum(np.power(weights, 2))
print(event + ' has ' + str(n_eff) + ' effective samples')
print('As a fraction, this is ' + str(n_eff) + '/' + str(len(weights)) + '=' + str(n_eff / len(weights)))

bayes_factor = np.sum(weights) / len(weights)
print('Bayes factor: ' + str(bayes_factor))
print('log Bayes factor: ' + str(np.log(bayes_factor)))
print('log 10 Bayes factor: ' + str(np.log10(bayes_factor)))
print('done!')