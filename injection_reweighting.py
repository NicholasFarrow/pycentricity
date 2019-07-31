#!/home/isobel.romero-shaw/anaconda3/bin/python3.6
import bilby as bb
import python_code.waveform as wf
import python_code.reweight as rwt
import numpy as np
import json
import pandas as pd
import argparse

# Set up the argument parser
parser = argparse.ArgumentParser("")
parser.add_argument("-i", "--index", help="")
parser.add_argument("-s", "--subsets_dir", help="")
parser.add_argument("-d", "injection_data", help="")
parser.add_argument("--H1-strain-data", help="")
parser.add_argument("--L1-strain-data", help="")
args = parser.parse_args()

np.random.seed(5432)
result_dir = args.subsets_dir
injection_data_dict = {
    'H1': args.H1_strain_data,
    'L1': args.L1_strain_data
}

# Frequency settings
minimum_frequency = 20
maximum_frequency = 1024
sampling_frequency = 4096
duration = 8
geocent_time = 0
start_time = geocent_time - duration + 2
deltaT = 0.2

# Comparison waveform
comparison_waveform_generator = wf.get_IMRPhenomD_comparison_waveform_generator(
    minimum_frequency, sampling_frequency, duration
)


# Interferometers
interferometers = bb.gw.detector.InterferometerList(["H1", "L1"])
for ifo in interferometers:
    data_file = injection_data_dict[ifo.name]
    ifo.strain_data._times_and_frequencies = \
        bb.core.series.CoupledTimeAndFrequencySeries(duration=duration,
                                                     sampling_frequency=sampling_frequency,
                                                     start_time=start_time)
    frequency = []
    strain = []
    with open(data_file, 'r') as data:
        for line in data:
            split_line = line.replace('\n', '').split(',')
            f = float(split_line[0])
            s = np.complex(split_line[1])
            frequency.append(f)
            strain.append(s)
    ifo.set_strain_data_from_frequency_domain_strain(
            np.asarray(strain), start_time=start_time, frequency_array=np.asarray(frequency))
    ifo.minimum_frequency = minimum_frequency
    ifo.maximum_frequency = maximum_frequency

# Access the samples
sub_result = result_dir + 'result_' + str(args.index) + '.json'
json_data = json.load(open(sub_result))
samples = json_data["samples"]
samples = {key: pd.DataFrame(samples[key]) for key in samples.keys()}
log_likelihoods = json_data["log_likelihoods"]

# Output to the folder with all of the result subsets
folder_list = sub_result.split("/")
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
        comparison_waveform_generator,
        interferometers,
        duration,
        folder,
        maximum_frequency,
        label=label,
)
print("Results weighted for file " + sub_result)