#!/home/nicholas.farrow/anaconda3/bin/python3 -u
print("Importing bilby...")
import bilby as bb
print("Imported bilby")
import pandas as pd

import python_code.utils as utils
import python_code.reweight as rwt
import python_code.waveform as wf

import json
import argparse

print("Imported modules.")

# Set up the argument parser
parser = argparse.ArgumentParser("Produce weights for one subset of a results file")
parser.add_argument("-e", "--event", help="GW-name of the event")
parser.add_argument("-s", "--sub-result", help="Path of the sub-result file to use")
args = parser.parse_args()

# Access the samples
json_data = json.load(open(args.sub_result))
samples = json_data["samples"]
samples = {key: pd.DataFrame(samples[key]) for key in samples.keys()}
log_likelihoods = json_data["log_likelihoods"]

# Set up the basic properties of the runs
maximum_frequency = 2048 
post_trigger_duration = 2
deltaT = 0.2
wf_minimum_frequency = 30 #utils.minimum_frequency[args.event]

# Read event-specific properties from the utils file
sampling_frequency = 8192 #utils.sampling_frequency[args.event]
ifo_minimum_frequency = 30 #utils.minimum_frequency[args.event]
duration = 128 #utils.event_duration[args.event]
detectors = ['L1', 'V1'] #utils.event_detectors[args.event]
trigger_time = 1240215503.0171 #utils.trigger_time[args.event]

# Generate the comparison waveform generator
waveform_generator = wf.get_comparison_waveform_generator(
    wf_minimum_frequency, sampling_frequency, duration, maximum_frequency, 30, 'IMRPhenomPv2'
)

# Frequency array
frequency_array = waveform_generator.frequency_array

# Set up the interferometers with event data
interferometers = bb.gw.detector.InterferometerList(detectors)
start = trigger_time + post_trigger_duration - duration
end = start + duration

psd_dict = {'L1': '/home/nicholas.farrow/public_html/gwInference/newROQ/GW190425/glitch_median_PSD_forLI_L1.dat', 'V1': '/home/nicholas.farrow/public_html/gwInference/newROQ/GW190425/glitch_median_PSD_forLI_V1.dat'}

for ifo in interferometers:
    psd_file = psd_dict[ifo.name]
    ifo.set_strain_data_from_csv(
        '/home/nicholas.farrow/public_html/gwInference/pycentricity/submissions/'
        + args.event + '/event_data/' + ifo.name + '_time_domain_strain_data.csv'
    )
    
    ifo.power_spectral_density = bb.gw.detector.PowerSpectralDensity.from_power_spectral_density_file(
        psd_file=psd_file
    )
    ifo.minimum_frequency = ifo_minimum_frequency
    ifo.maximum_frequency = maximum_frequency

print("Loaded PSD and Strain")

# Output to the folder with all of the result subsets
folder_list = args.sub_result.split("/")
folder = ""
for string in folder_list[0:-2]:
    folder += string + "/"
number_of_eccentricity_bins = 10 
folder += "weights_{}/".format(number_of_eccentricity_bins)
bb.core.utils.check_directory_exists_and_if_not_mkdir(folder)
label = folder_list[-1].split(".")[0]
print("Reweighting samples...")
output = rwt.reweight_by_eccentricity(
    samples,
    log_likelihoods,
    sampling_frequency,
    waveform_generator,
    interferometers,
    duration,
    folder,
    maximum_frequency,
    label=label,
    number_of_eccentricity_bins=number_of_eccentricity_bins
)
print("Results weighted for file " + args.sub_result + " for event " + args.event)
