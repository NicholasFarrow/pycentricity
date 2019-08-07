#!/home/isobel.romero-shaw/anaconda3/bin/python3.6
import bilby as bb
import pandas as pd

import python_code.utils as utils
import python_code.reweight as rwt
import python_code.waveform as wf

import json
import argparse


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
maximum_frequency = 1024
post_trigger_duration = 2
deltaT = 0.2
wf_minimum_frequency = utils.minimum_frequency[args.event]

# Read event-specific properties from the utils file
sampling_frequency = utils.sampling_frequency[args.event]
ifo_minimum_frequency = utils.minimum_frequency[args.event]
duration = utils.event_duration[args.event]
detectors = utils.event_detectors[args.event]
trigger_time = utils.trigger_time[args.event]

# Generate the comparison waveform generator
waveform_generator = wf.get_IMRPhenomD_comparison_waveform_generator(
    wf_minimum_frequency, sampling_frequency, duration
)
# Frequency array
frequency_array = waveform_generator.frequency_array

# Set up the interferometers with event data
interferometers = bb.gw.detector.InterferometerList(detectors)
start = trigger_time + post_trigger_duration - duration
end = start + duration
channel_dict = {
    key: key + ":" + utils.event_channels[args.event][key] for key in detectors
}
for ifo in interferometers:
    ifo.set_strain_data_from_csv(
        '/home/isobel.romero-shaw/public_html/PYCENTRICITY/pycentricity/submissions/'
        + args.event + '/event_data/' + ifo.name + '_time_domain_strain_data.csv'
    )
    ifo.power_spectral_density = bb.gw.detector.PowerSpectralDensity.from_power_spectral_density_file(
        psd_file=utils.event_psd_file_path[args.event][ifo.name]
    )
    ifo.minimum_frequency = ifo_minimum_frequency
    ifo.maximum_frequency = maximum_frequency

# Output to the folder with all of the result subsets
folder_list = args.sub_result.split("/")
folder = ""
for string in folder_list[0:-2]:
    folder += string + "/"
folder += "weights/"
bb.core.utils.check_directory_exists_and_if_not_mkdir(folder)
label = folder_list[-1].split(".")[0]
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
)
print("Results weighted for file " + args.sub_result + " for event " + args.event)
